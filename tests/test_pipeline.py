import unittest
from types import SimpleNamespace

import torch

from aidetect.pipeline import detect_text, generate_explanation


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __call__(self, _text, **_kwargs):
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))


class FakeClassifier:
    def __call__(self, **_inputs):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))


class BrokenClassifier:
    def __call__(self, **_inputs):
        raise RuntimeError("classifier failed")


class FakeCausalModel:
    device = "cpu"

    def __call__(self, input_ids, labels):
        if not torch.equal(input_ids, labels):
            raise AssertionError("causal language-model labels must match input IDs")
        return SimpleNamespace(loss=torch.log(torch.tensor(40.0)))


class PipelineTests(unittest.TestCase):
    def test_classifier_only_record_matches_existing_score_scale(self):
        record = detect_text(
            "short text",
            tokenizer=FakeTokenizer(),
            model=FakeClassifier(),
            device="cpu",
            ai_label_index=1,
        )

        self.assertEqual(record.raw_classifier_score, 88.08)
        self.assertTrue(record.raw_classifier_is_ai)
        self.assertEqual(record.fused_score, 88.08)
        self.assertIsNone(record.perplexity_value)
        self.assertIsNone(record.burstiness_cv)

    def test_optional_features_are_fused_into_the_record(self):
        record = detect_text(
            "abcdef. ghijkl. mnopqr.",
            tokenizer=FakeTokenizer(),
            model=FakeClassifier(),
            device="cpu",
            ai_label_index=1,
            perplexity_tokenizer=FakeTokenizer(),
            perplexity_model=FakeCausalModel(),
        )

        self.assertEqual(record.perplexity_value, 40.0)
        self.assertEqual(record.perplexity_heuristic_score, 50.0)
        self.assertEqual(record.burstiness_cv, 0.0)
        self.assertEqual(record.burstiness_heuristic_score, 93.5)
        self.assertEqual(record.fused_score, 66.32)

    def test_classifier_error_is_recorded_without_losing_the_result(self):
        record = detect_text(
            "short text",
            tokenizer=FakeTokenizer(),
            model=BrokenClassifier(),
            device="cpu",
            ai_label_index=1,
        )

        self.assertEqual(record.raw_classifier_score, 0.0)
        self.assertFalse(record.raw_classifier_is_ai)
        self.assertEqual(record.fused_score, 0.0)
        self.assertEqual(record.error, "classifier failed")

    def test_explanation_preserves_existing_score_boundaries(self):
        self.assertTrue(generate_explanation(29.99).startswith("文本符合人类写作特征"))
        self.assertTrue(generate_explanation(30.0).startswith("文本疑似混合生成"))
        self.assertTrue(generate_explanation(69.99).startswith("文本疑似混合生成"))
        self.assertTrue(generate_explanation(70.0).startswith("文本高度疑似AI生成"))


if __name__ == "__main__":
    unittest.main()
