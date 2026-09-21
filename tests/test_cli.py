import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from aidetect.cli import detect_batch, main, read_input, resolve_model_id, write_output
from aidetect.models import MODEL_REGISTRY


class FakeBatch(dict):
    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __call__(self, _text, **_kwargs):
        return FakeBatch(input_ids=torch.tensor([[1, 2, 3]]))


class FakeClassifierModel:
    def __call__(self, **_inputs):
        return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))


class CliInputTests(unittest.TestCase):
    def test_reads_unicode_jsonl_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(
                '{"document_id":"zh-1","text":"中文文本"}\n'
                '{"document_id":"en-1","text":"English text"}\n',
                encoding="utf-8",
            )

            self.assertEqual(
                read_input(path),
                [
                    {"document_id": "zh-1", "text": "中文文本"},
                    {"document_id": "en-1", "text": "English text"},
                ],
            )

    def test_reads_csv_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["document_id", "text"])
                writer.writeheader()
                writer.writerow({"document_id": "doc-1", "text": "Example text"})

            self.assertEqual(
                read_input(path),
                [{"document_id": "doc-1", "text": "Example text"}],
            )

    def test_rejects_duplicate_document_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text(
                '{"document_id":"dup","text":"first"}\n'
                '{"document_id":"dup","text":"second"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate document_id"):
                read_input(path)


class CliOutputTests(unittest.TestCase):
    def test_batch_output_contains_traceability_and_stable_scores(self):
        classifier = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeClassifierModel(),
            device="cpu",
            ai_label_index=1,
            source="owner/model",
        )
        outputs = detect_batch(
            [{"document_id": "doc-1", "text": "Example text"}],
            classifier=classifier,
            model_id="owner/model",
        )

        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]["document_id"], "doc-1")
        self.assertEqual(outputs[0]["model_id"], "owner/model")
        self.assertEqual(outputs[0]["device"], "cpu")
        self.assertEqual(outputs[0]["raw_classifier_score"], 88.08)
        self.assertEqual(outputs[0]["fused_score"], 88.08)
        self.assertEqual(outputs[0]["score_scale"], "percent_0_100")

    def test_writes_jsonl_and_csv_outputs(self):
        records = [{"document_id": "doc-1", "text": "中文", "fused_score": 50.0}]
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "output.jsonl"
            csv_path = Path(directory) / "output.csv"
            write_output(jsonl_path, records)
            write_output(csv_path, records)

            jsonl_record = json.loads(jsonl_path.read_text(encoding="utf-8"))
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                csv_record = next(csv.DictReader(handle))

        self.assertEqual(jsonl_record, records[0])
        self.assertEqual(csv_record["document_id"], "doc-1")
        self.assertEqual(csv_record["text"], "中文")

    def test_resolves_registry_name_or_keeps_raw_model_id(self):
        display_name = next(iter(MODEL_REGISTRY))
        self.assertEqual(resolve_model_id(display_name), MODEL_REGISTRY[display_name])
        self.assertEqual(resolve_model_id("owner/model"), "owner/model")


class CliMainTests(unittest.TestCase):
    def test_main_runs_jsonl_batch_without_real_model_downloads(self):
        classifier = SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeClassifierModel(),
            device="cpu",
            ai_label_index=1,
            source="owner/model",
        )
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "input.jsonl"
            output_path = Path(directory) / "output.jsonl"
            input_path.write_text(
                '{"document_id":"doc-1","text":"Example text"}\n',
                encoding="utf-8",
            )

            with patch("aidetect.cli.load_classifier", return_value=classifier):
                with redirect_stdout(io.StringIO()):
                    exit_code = main([
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                        "--model",
                        "owner/model",
                    ])

            output_record = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(output_record["document_id"], "doc-1")
        self.assertEqual(output_record["model_id"], "owner/model")
        self.assertEqual(output_record["fused_score"], 88.08)


if __name__ == "__main__":
    unittest.main()
