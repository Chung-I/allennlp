from collections import Counter
import math
from typing import Iterable, Tuple, Dict, List, Set

from overrides import overrides
import torch
import torch.distributed as dist

from allennlp.common.util import is_distributed
from allennlp.training.metrics.metric import Metric


@Metric.register("wer")
class WER(Metric):
    """
    Bilingual Evaluation Understudy (BLEU).

    BLEU is a common metric used for evaluating the quality of machine translations
    against a set of reference translations. See
    [Papineni et. al., "BLEU: a method for automatic evaluation of machine translation", 2002][1].

    # Parameters

    ngram_weights : `Iterable[float]`, optional (default = `(0.25, 0.25, 0.25, 0.25)`)
        Weights to assign to scores for each ngram size.
    exclude_indices : `Set[int]`, optional (default = `None`)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.

    # Notes

    We chose to implement this from scratch instead of wrapping an existing implementation
    (such as `nltk.translate.bleu_score`) for a two reasons. First, so that we could
    pass tensors directly to this metric instead of first converting the tensors to lists of strings.
    And second, because functions like `nltk.translate.bleu_score.corpus_bleu()` are
    meant to be called once over the entire corpus, whereas it is more efficient
    in our use case to update the running precision counts every batch.

    This implementation only considers a reference set of size 1, i.e. a single
    gold target sequence for each predicted sequence.


    [1]: https://www.semanticscholar.org/paper/8ff93cfd37dced279134c9d642337a2085b31f59/
    """

    def __init__(
        self,
        exclude_indices: Set[str] = None,
        non_word_final_suffix: str = None,
        is_char_level_input: bool = False,
    ) -> None:
        self._exclude_indices = exclude_indices or set()
        self._edit_distances = 0
        self._reference_lengths = 0
        self._char_edit_distances = 0
        self._char_reference_lengths = 0
        self._non_word_final_suffix = non_word_final_suffix
        self._is_char_level_input = is_char_level_input

    @overrides
    def reset(self) -> None:
        self._edit_distances = 0
        self._reference_lengths = 0
        self._char_edit_distances = 0
        self._char_reference_lengths = 0

    def filter_exclude_indices(self, predictions: List[List[str]]) -> List[List[str]]:
        return [list(filter(lambda word: word not in self._exclude_indices, prediction)) 
                for prediction in predictions]                                     

    def compose_chars_to_sent(self, chars: List[str]) -> str:
        sent = ""
        for char in chars:
            if char.endswith(self._non_word_final_suffix):
                sent += char[:len(self._non_word_final_suffix)]
            else:
                sent += char + " "
        return sent

    @overrides
    def __call__(
        self,  # type: ignore
        predictions: List[List[str]],
        gold_targets: List[List[str]],
    ) -> None:
        """
        Update precision counts.

        # Parameters

        predictions : `torch.LongTensor`, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        references : `torch.LongTensor`, required
            Batched reference (gold) translations with shape `(batch_size, max_gold_sequence_length)`.

        # Returns

        None
        """
        predictions = self.filter_exclude_indices(predictions)
        gold_targets = self.filter_exclude_indices(gold_targets)
        if self._is_char_level_input:
            predictions = list(map(self.compose_chars_to_sent, predictions))
            gold_targets = list(map(self.compose_chars_to_sent, gold_targets))
        else:
            predictions = list(map(lambda words: " ".join(words), predictions))
            gold_targets = list(map(lambda words: " ".join(words), gold_targets))

        from nltk.metrics import distance

        _edit_distances = sum([distance.edit_distance(prediction.split(), gold_target.split()) for prediction, gold_target
                               in zip(predictions, gold_targets)])
        _reference_lengths = sum(map(lambda sent: len(sent.split()), gold_targets))

        self._edit_distances += _edit_distances
        self._reference_lengths += _reference_lengths

        _char_edit_distances = sum([distance.edit_distance(prediction, gold_target) for prediction, gold_target
                               in zip(predictions, gold_targets)])
        _char_reference_lengths = sum(map(len, gold_targets))
        self._char_edit_distances += _char_edit_distances
        self._char_reference_lengths += _char_reference_lengths

    @overrides
    def get_metric(self, reset: bool = False) -> Dict[str, float]:

        wer = self._edit_distances / self._reference_lengths
        cer = self._char_edit_distances / self._char_reference_lengths
        if reset:
            self.reset()
        return {"WER": wer, "CER": cer}
