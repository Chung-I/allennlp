from typing import Any, Dict, List, Tuple, Union
import math
from collections import Counter

import torch
from torch.testing import assert_allclose

from allennlp.common.testing import (
    AllenNlpTestCase,
    multi_device,
    global_distributed_metric,
    run_distributed_test,
)
from allennlp.training.metrics import WER
from allennlp.training.util import ngrams, get_valid_tokens_mask


class WERTest(AllenNlpTestCase):
    def setup_method(self):
        super().setup_method()
        self.metric = WER(exclude_indices={"@@PADDING@@"})

    def test_wer_computed_correctly(self):
        self.metric.reset()
        predictions = [["bow", "wow", "@@PADDING@@"], ["cody", "jody"]]
        gold_targets = [["bow", "bow", "cow"], ["jody", "cody"]]
        self.metric(predictions, gold_targets)
        assert self.metric.get_metric()["WER"] == 0.8
        assert self.metric.get_metric()["CER"] == 0.3

        self.metric.reset()
        predictions = [["bow", "wow", "@@PADDING@@"], ["cody", "jody"]]
        gold_targets = [["bowwow"], ["jody", "cody"]]
        self.metric(predictions, gold_targets)
        assert self.metric.get_metric()["WER"] == (4/3)
        assert self.metric.get_metric()["CER"] == 0.2
