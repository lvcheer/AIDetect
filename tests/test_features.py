import unittest
from types import SimpleNamespace

import torch

from aidetect.features import (
    calculate_burstiness_feature,
    calculate_perplexity_feature,
)


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self):
        self.last_call = None

    def __call__(self, text, **kwargs):
        self.last_call = (text, kwargs)
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))


class FakeCausalModel:
    device = "cpu"

    def __call__(self, input_ids, labels):
        if not torch.equal(input_ids, labels):
            raise AssertionError("causal language-model labels must match input IDs")
        return SimpleNamespace(loss=torch.log(torch.tensor(40.0)))


class PerplexityFeatureTests(unittest.TestCase):
    def test_preserves_existing_perplexity_transform(self):
        tokenizer = FakeTokenizer()
        feature = calculate_perplexity_feature(
            "example",
            tokenizer=tokenizer,
            model=FakeCausalModel(),
        )

        self.assertAlmostEqual(feature.perplexity, 40.0, places=5)
        self.assertAlmostEqual(feature.heuristic_score, 50.0, places=5)
        self.assertEqual(tokenizer.last_call[1]["max_length"], 512)
        self.assertTrue(tokenizer.last_call[1]["truncation"])


class BurstinessFeatureTests(unittest.TestCase):
    def test_requires_three_sentences_longer_than_five_characters(self):
        self.assertIsNone(calculate_burstiness_feature("short. also short."))

    def test_equal_sentence_lengths_have_zero_variation(self):
        feature = calculate_burstiness_feature(
            "abcdef. ghijkl. mnopqr."
        )

        self.assertEqual(feature.coefficient_of_variation, 0.0)
        self.assertAlmostEqual(feature.heuristic_score, 93.503083, places=6)

    def test_supports_chinese_and_english_sentence_boundaries(self):
        feature = calculate_burstiness_feature(
            "第一句长度足够。This sentence is longer!第三句也有足够长度？"
        )

        self.assertIsNotNone(feature)
        self.assertGreater(feature.coefficient_of_variation, 0.0)
        self.assertGreaterEqual(feature.heuristic_score, 0.0)
        self.assertLessEqual(feature.heuristic_score, 100.0)


if __name__ == "__main__":
    unittest.main()
