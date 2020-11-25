from typing import Dict, List

from overrides import overrides
import torch

from allennlp.common.util import pad_sequence_to_length
from allennlp.data.tokenizers.token import Token
from allennlp.data.token_indexers.token_indexer import TokenIndexer, IndexedTokenList
from allennlp.data.vocabulary import Vocabulary

from pypinyin import pinyin, Style

def _make_bos_eos(
    character: int,
    padding_character: int,
    beginning_of_word_character: int,
    end_of_word_character: int,
    max_word_length: int,
):
    char_ids = [padding_character] * max_word_length
    char_ids[0] = beginning_of_word_character
    char_ids[1] = character
    char_ids[2] = end_of_word_character
    return char_ids


class BPMFCharacterMapper:
    """
    Maps individual tokens to sequences of character ids, compatible with BPMF.
    To be consistent with previously trained models, we include it here as special of existing
    character indexers.

    We allow to add optional additional special tokens with designated
    character ids with `tokens_to_add`.
    """

    max_word_length = 6

    # char ids 0-255 come from utf-8 encoding bytes
    # assign 256-300 to special chars
    bpmf_id_offset = 261
    beginning_of_sentence_character = 256  # <begin sentence>
    end_of_sentence_character = 257  # <end sentence>
    beginning_of_word_character = 258  # <begin word>
    end_of_word_character = 259  # <end word>
    padding_character = 260  # <padding>

    beginning_of_sentence_characters = _make_bos_eos(
        beginning_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )
    end_of_sentence_characters = _make_bos_eos(
        end_of_sentence_character,
        padding_character,
        beginning_of_word_character,
        end_of_word_character,
        max_word_length,
    )

    bopomofo = "ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦˊˇˋ˙\'"

    def __init__(self,
                 bos_token = "<S>",
                 eos_token = "</S>",
                 tokens_to_add: Dict[str, int] = None) -> None:
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.tokens_to_add = tokens_to_add or {}

    @staticmethod
    def convert_char_to_id(char: str) -> int:
        try:
            return BPMFCharacterMapper.bpmf_id_offset + BPMFCharacterMapper.bopomofo.index(char)
        except ValueError:
            return char.encode("utf-8", "ignore")[0]

    def convert_word_to_char_ids(self, word: str) -> List[int]:
        if word in self.tokens_to_add:
            char_ids = [BPMFCharacterMapper.padding_character] * BPMFCharacterMapper.max_word_length
            char_ids[0] = BPMFCharacterMapper.beginning_of_word_character
            char_ids[1] = self.tokens_to_add[word]
            char_ids[2] = BPMFCharacterMapper.end_of_word_character
        elif word == self.bos_token:
            char_ids = BPMFCharacterMapper.beginning_of_sentence_characters
        elif word == self.eos_token:
            char_ids = BPMFCharacterMapper.end_of_sentence_characters
        else:
            bpmf_ids = [BPMFCharacterMapper.convert_char_to_id(zhuyin)
                        for zhuyin in pinyin(word, style=Style.BOPOMOFO)[0][0]]
            word_encoded = bpmf_ids[
                : (BPMFCharacterMapper.max_word_length - 2)
            ]
            char_ids = [BPMFCharacterMapper.padding_character] * BPMFCharacterMapper.max_word_length
            char_ids[0] = BPMFCharacterMapper.beginning_of_word_character
            for k, chr_id in enumerate(word_encoded, start=1):
                char_ids[k] = chr_id
            char_ids[len(word_encoded) + 1] = BPMFCharacterMapper.end_of_word_character

        # +1 one for masking
        return [c + 1 for c in char_ids]

    def __eq__(self, other) -> bool:
        if isinstance(self, other.__class__):
            return self.__dict__ == other.__dict__
        return NotImplemented


@TokenIndexer.register("bpmf")
class BPMFTokenCharactersIndexer(TokenIndexer):
    """
    Convert a token to an array of character ids to compute BPMF representations.

    Registered as a `TokenIndexer` with name "bpmf_characters".

    # Parameters

    namespace : `str`, optional (default=`bpmf_characters`)
    tokens_to_add : `Dict[str, int]`, optional (default=`None`)
        If not None, then provides a mapping of special tokens to character
        ids. When using pre-trained models, then the character id must be
        less then 261, and we recommend using un-used ids (e.g. 1-32).
    token_min_padding_length : `int`, optional (default=`0`)
        See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        namespace: str = "bpmf_characters",
        tokens_to_add: Dict[str, int] = None,
        token_min_padding_length: int = 0,
        bos_token = "<S>",
        eos_token = "</S>",
    ) -> None:
        super().__init__(token_min_padding_length)
        self._namespace = namespace
        self._mapper = BPMFCharacterMapper(bos_token, eos_token, tokens_to_add)

    @overrides
    def count_vocab_items(self, token: Token, counter: Dict[str, Dict[str, int]]):
        pass

    @overrides
    def get_empty_token_list(self) -> IndexedTokenList:
        return {"bpmf_tokens": []}

    @overrides
    def tokens_to_indices(
        self, tokens: List[Token], vocabulary: Vocabulary
    ) -> Dict[str, List[List[int]]]:
        # TODO(brendanr): Retain the token to index mappings in the vocabulary and remove this

        # https://github.com/allenai/allennlp/blob/master/allennlp/data/token_indexers/wordpiece_indexer.py#L113

        return {
            "bpmf_tokens": [self._mapper.convert_word_to_char_ids(t.ensure_text()) for t in tokens]
        }

    @overrides
    def as_padded_tensor_dict(
        self, tokens: IndexedTokenList, padding_lengths: Dict[str, int]
    ) -> Dict[str, torch.Tensor]:
        # Overriding this method only because we need a different padding token than the default.
        tensor_dict = {}

        def padding_token():
            return [0] * BPMFCharacterMapper.max_word_length

        tensor_dict["bpmf_tokens"] = torch.LongTensor(
            pad_sequence_to_length(
                tokens["bpmf_tokens"], padding_lengths["bpmf_tokens"], default_value=padding_token
            )
        )
        return tensor_dict
