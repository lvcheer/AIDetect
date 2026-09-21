import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from aidetect.models import (
    MODEL_REGISTRY,
    PERPLEXITY_MODEL_ID,
    find_ai_label_index,
    load_classifier,
    load_perplexity_model,
    local_model_path,
    resolve_model_source,
)


class FakeModel:
    def __init__(self, id2label=None):
        self.config = SimpleNamespace(id2label=id2label or {0: "Human", 1: "AI"})
        self.device = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeTorchWithCuda:
    class cuda:
        @staticmethod
        def is_available():
            return True


class FakeTorchWithoutCuda:
    class cuda:
        @staticmethod
        def is_available():
            return False


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

    def test_model_source_prefers_existing_local_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            local_path = local_model_path(directory, "owner/model")
            local_path.mkdir()
            self.assertEqual(
                resolve_model_source(directory, "owner/model"),
                local_path,
            )

    def test_model_source_falls_back_to_remote_id(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(
                resolve_model_source(directory, "owner/model"),
                "owner/model",
            )


class LabelMappingTests(unittest.TestCase):
    def test_finds_ai_label_with_integer_keys(self):
        self.assertEqual(find_ai_label_index({0: "Human", 1: "AIGC"}), 1)

    def test_accepts_string_indices(self):
        self.assertEqual(find_ai_label_index({"0": "AI-generated", "1": "Human"}), 0)

    def test_preserves_label_one_fallback(self):
        self.assertEqual(find_ai_label_index({0: "LABEL_0", 1: "LABEL_1"}), 1)


class ModelLoadingTests(unittest.TestCase):
    def test_classifier_loader_prepares_model_and_label_mapping(self):
        loaded_sources = []
        model = FakeModel({0: "AI-generated", 1: "Human"})

        def load_tokenizer(source):
            loaded_sources.append(("tokenizer", source))
            return "tokenizer"

        def load_model(source):
            loaded_sources.append(("model", source))
            return model

        with tempfile.TemporaryDirectory() as directory:
            loaded = load_classifier(
                "owner/model",
                directory,
                tokenizer_loader=load_tokenizer,
                model_loader=load_model,
                torch_module=FakeTorchWithCuda,
            )

        self.assertEqual(loaded_sources, [
            ("tokenizer", "owner/model"),
            ("model", "owner/model"),
        ])
        self.assertEqual(loaded.tokenizer, "tokenizer")
        self.assertEqual(loaded.device, "cuda")
        self.assertEqual(loaded.ai_label_index, 0)
        self.assertEqual(model.device, "cuda")
        self.assertTrue(model.eval_called)

    def test_classifier_loader_falls_back_to_cpu(self):
        model = FakeModel()
        with tempfile.TemporaryDirectory() as directory:
            loaded = load_classifier(
                "owner/model",
                directory,
                tokenizer_loader=lambda _source: "tokenizer",
                model_loader=lambda _source: model,
                torch_module=FakeTorchWithoutCuda,
            )

        self.assertEqual(loaded.device, "cpu")
        self.assertEqual(model.device, "cpu")

    def test_perplexity_loader_uses_existing_model_id_and_device(self):
        loaded_sources = []
        model = FakeModel()

        def load_tokenizer(source):
            loaded_sources.append(("tokenizer", source))
            return "tokenizer"

        def load_model(source):
            loaded_sources.append(("model", source))
            return model

        loaded = load_perplexity_model(
            device="cpu",
            tokenizer_loader=load_tokenizer,
            model_loader=load_model,
        )

        self.assertEqual(loaded.source, PERPLEXITY_MODEL_ID)
        self.assertEqual(loaded_sources, [
            ("tokenizer", PERPLEXITY_MODEL_ID),
            ("model", PERPLEXITY_MODEL_ID),
        ])
        self.assertEqual(model.device, "cpu")
        self.assertTrue(model.eval_called)


if __name__ == "__main__":
    unittest.main()
