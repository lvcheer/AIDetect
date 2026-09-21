import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import torch

from aidetect.benchmark_runner import main, run_record, verify_frozen_manifest
from aidetect.manifest import assign_splits, load_schema, write_jsonl


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "dataset_manifest_schema.json"
)


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    truncation_side = "right"
    model_max_length = 512

    def __call__(self, _text, **kwargs):
        if kwargs.get("return_tensors") == "pt":
            return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))
        return {"input_ids": [1, 2, 3, 4, 5]}


class FakeClassifierModel:
    config = SimpleNamespace(id2label={0: "Human", 1: "AI"})

    def __call__(self, **_inputs):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))


class BrokenClassifierModel:
    config = SimpleNamespace(id2label={0: "Human", 1: "AI"})

    def __call__(self, **_inputs):
        raise RuntimeError("classifier failed")


class FakeCausalModel:
    device = "cpu"

    def __call__(self, input_ids, labels):
        return SimpleNamespace(loss=torch.log(torch.tensor(40.0)))


def make_record(root, document_id, source_id, text, generator=None):
    text_directory = root / "texts"
    text_directory.mkdir(exist_ok=True)
    relative_path = Path("texts") / f"{document_id}.txt"
    text_bytes = text.encode("utf-8")
    (root / relative_path).write_bytes(text_bytes)
    is_ai = generator is not None
    return {
        "schema_version": "0.1",
        "document_id": document_id,
        "source_id": source_id,
        "language": "en",
        "domain": "general",
        "human_or_ai": "ai" if is_ai else "human",
        "generator": generator,
        "generator_revision": "v1" if is_ai else None,
        "prompt_id": f"prompt-{document_id}" if is_ai else None,
        "temperature": 0.7 if is_ai else None,
        "editing_condition": "original",
        "author_group": None,
        "license": "CC-BY-4.0",
        "provenance_uri": f"https://example.test/{document_id}",
        "text_path": str(relative_path),
        "text_sha256": hashlib.sha256(text_bytes).hexdigest(),
        "split": "train",
        "evaluation_partition": "in_distribution",
        "near_duplicate_cluster_id": None,
    }


class BenchmarkRecordTests(unittest.TestCase):
    def test_perplexity_remains_a_separate_feature(self):
        classifier = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeClassifierModel(),
            device="cpu",
            ai_label_index=1,
        )
        perplexity = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeCausalModel(),
        )
        manifest_record = {
            "document_id": "doc-1",
            "source_id": "source-1",
            "language": "en",
            "domain": "general",
            "human_or_ai": "human",
            "generator": None,
            "generator_revision": None,
            "editing_condition": "original",
            "split": "test",
            "evaluation_partition": "in_distribution",
            "text_sha256": "a" * 64,
        }

        result = run_record(
            manifest_record,
            "abcdef. ghijkl. mnopqr.",
            classifier=classifier,
            run_context={"run_id": "run-1"},
            max_length=512,
            perplexity=perplexity,
        )

        self.assertEqual(result["perplexity_value"], 40.0)
        self.assertEqual(result["perplexity_heuristic_score"], 50.0)
        self.assertIsNotNone(result["burstiness_heuristic_score"])
        self.assertNotIn("fused_score", result)

    def test_classifier_failure_is_null_not_zero(self):
        classifier = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=BrokenClassifierModel(),
            device="cpu",
            ai_label_index=1,
        )
        manifest_record = {
            "document_id": "doc-1",
            "source_id": "source-1",
            "language": "en",
            "domain": "general",
            "human_or_ai": "human",
            "generator": None,
            "generator_revision": None,
            "editing_condition": "original",
            "split": "test",
            "evaluation_partition": "in_distribution",
            "text_sha256": "a" * 64,
        }
        clock_values = iter([1.0, 1.1, 1.2, 1.3])

        result = run_record(
            manifest_record,
            "Example text",
            classifier=classifier,
            run_context={"run_id": "run-1"},
            max_length=3,
            clock=lambda: next(clock_values),
        )

        self.assertIsNone(result["raw_ai_score"])
        self.assertIsNone(result["classifier_class_scores"])
        self.assertEqual(result["status"], "failed")
        self.assertIn("classifier failed", result["classifier_error"])
        self.assertEqual(result["input_token_length"], 5)
        self.assertEqual(result["effective_token_length"], 3)
        self.assertTrue(result["truncated"])

    def test_frozen_manifest_hash_must_match_split_metadata(self):
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            verify_frozen_manifest(
                "manifest-hash",
                "schema-hash",
                {
                    "output_manifest_sha256": "different-hash",
                    "schema_sha256": "schema-hash",
                    "held_out_generators": ["generator-b"],
                },
                [],
            )


class BenchmarkRunnerCliTests(unittest.TestCase):
    def test_cli_writes_auditable_results_without_model_downloads(self):
        schema, schema_hash = load_schema(SCHEMA_PATH)
        del schema
        classifier = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeClassifierModel(),
            device="cpu",
            ai_label_index=1,
            source="owner/model",
            resolved_revision="b" * 40,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "frozen.jsonl"
            split_metadata_path = root / "split_metadata.json"
            output_path = root / "results.jsonl"
            run_metadata_path = root / "run_metadata.json"
            records = [
                make_record(root, "human-1", "source-1", "Human text one."),
                make_record(root, "human-2", "source-2", "Human text two."),
                make_record(root, "ai-1", "source-3", "AI text one.", "generator-a"),
                make_record(root, "ai-2", "source-4", "AI text two.", "generator-b"),
            ]
            assigned = assign_splits(records, 2026, 0.6, 0.2, ["generator-b"])
            manifest_hash = write_jsonl(manifest_path, assigned)
            split_metadata_path.write_text(
                json.dumps(
                    {
                        "output_manifest_sha256": manifest_hash,
                        "schema_sha256": schema_hash,
                        "held_out_generators": ["generator-b"],
                    }
                ),
                encoding="utf-8",
            )

            with patch(
                "aidetect.benchmark_runner.load_classifier",
                return_value=classifier,
            ) as mocked_loader:
                with redirect_stdout(io.StringIO()):
                    exit_code = main(
                        [
                            "--manifest",
                            str(manifest_path),
                            "--split-metadata",
                            str(split_metadata_path),
                            "--schema",
                            str(SCHEMA_PATH),
                            "--output",
                            str(output_path),
                            "--run-metadata-output",
                            str(run_metadata_path),
                            "--run-id",
                            "test-run",
                            "--code-commit",
                            "a" * 40,
                            "--model",
                            "owner/model",
                            "--model-revision",
                            "b" * 40,
                            "--ai-label-index",
                            "1",
                            "--max-length",
                            "3",
                            "--include-split",
                            "train",
                            "--include-split",
                            "calibration",
                            "--include-split",
                            "test",
                        ]
                    )

            results = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            output_hash = hashlib.sha256(output_path.read_bytes()).hexdigest()

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(results), 4)
        self.assertAlmostEqual(results[0]["raw_ai_score"], 0.880797, places=6)
        self.assertEqual(results[0]["classifier_class_count"], 2)
        self.assertEqual(results[0]["input_token_length"], 5)
        self.assertEqual(results[0]["effective_token_length"], 3)
        self.assertTrue(results[0]["truncated"])
        self.assertEqual(results[0]["status"], "success")
        self.assertNotIn("text", results[0])
        self.assertNotIn("fused_score", results[0])
        self.assertEqual(
            results[0]["configuration_sha256"],
            run_metadata["configuration_sha256"],
        )
        self.assertEqual(run_metadata["results_sha256"], output_hash)
        self.assertEqual(run_metadata["records_by_status"], {"success": 4})
        mocked_loader.assert_called_once_with(
            "owner/model",
            ANY,
            revision="b" * 40,
            ai_label_index=1,
        )


if __name__ == "__main__":
    unittest.main()
