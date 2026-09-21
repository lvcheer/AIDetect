import tempfile
import unittest
from pathlib import Path

from aidetect.models import MODEL_REGISTRY, find_ai_label_index, local_model_path


class ModelRegistryTests(unittest.TestCase):
    def test_registry_preserves_the_five_gui_models(self):
        self.assertEqual(len(MODEL_REGISTRY), 5)
        self.assertEqual(
            MODEL_REGISTRY["英文通用（OpenAI Detector）"],
            "roberta-base-openai-detector",
        )

    def test_local_model_path_matches_download_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                local_model_path(directory, "owner/model"),
                Path(directory) / "owner__model",
            )


class LabelMappingTests(unittest.TestCase):
    def test_finds_ai_label_with_integer_keys(self):
        self.assertEqual(find_ai_label_index({0: "Human", 1: "AIGC"}), 1)

    def test_accepts_string_indices(self):
        self.assertEqual(find_ai_label_index({"0": "AI-generated", "1": "Human"}), 0)

    def test_preserves_label_one_fallback(self):
        self.assertEqual(find_ai_label_index({0: "LABEL_0", 1: "LABEL_1"}), 1)


if __name__ == "__main__":
    unittest.main()
