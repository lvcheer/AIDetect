import unittest
from types import SimpleNamespace

import torch

from aidetect.inference import infer_raw_score


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


class FakeModel:
    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float32)

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=self.logits)


class RawInferenceTests(unittest.TestCase):
    def test_returns_uncalibrated_ai_score(self):
        tokenizer = FakeTokenizer()
        result = infer_raw_score(
            "example",
            tokenizer=tokenizer,
            model=FakeModel([0.0, 2.0]),
            device="cpu",
            ai_label_index=1,
        )

        self.assertEqual(result.text, "example")
        self.assertAlmostEqual(sum(result.scores), 1.0, places=6)
        self.assertAlmostEqual(result.ai_score, 0.880797, places=6)
        self.assertTrue(result.is_ai)
        self.assertEqual(tokenizer.last_call[1]["max_length"], 512)
        self.assertTrue(tokenizer.last_call[1]["truncation"])

    def test_rejects_ai_index_outside_model_classes(self):
        with self.assertRaisesRegex(ValueError, "outside 2 model classes"):
            infer_raw_score(
                "example",
                tokenizer=FakeTokenizer(),
                model=FakeModel([1.0, 0.0]),
                device="cpu",
                ai_label_index=2,
            )


if __name__ == "__main__":
    unittest.main()
