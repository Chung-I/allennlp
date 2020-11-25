import numpy as np
import pytest

from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.batch import Batch
from allennlp.data.token_indexers import BPMFTokenCharactersIndexer
from allennlp.data.fields import ListField, TextField


class TestBPMFTokenCharactersIndexer(AllenNlpTestCase):
    def test_bos_to_char_ids(self):
        indexer = BPMFTokenCharactersIndexer()
        indices = indexer.tokens_to_indices([Token("你")], Vocabulary())
        expected_indices = [
            259,
            268,
            283,
            300,
            260,
            261,
        ]
        assert indices == {"bpmf_tokens": [expected_indices]}
