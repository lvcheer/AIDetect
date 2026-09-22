import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from aidetect.local_generation import (
    build_preflight,
    run_stage_zero,
    validate_prompt_records,
    validate_receipts,
)


NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


def prompt(language="en", length_band="short"):
    return {
        "request_id": f"{language}-{length_band}-1",
        "prompt_family_id": f"family-{language}-{length_band}",
        "language": language,
        "domain": "general",
        "length_band": length_band,
        "system_prompt": "Write the requested passage.",
        "user_prompt": "Explain a neutral topic.",
    }


def metadata():
    return {
        "model_id": "Qwen/Qwen3-1.7B",
        "model_revision": "0123456789abcdef",
        "runtime": "fake-local-runtime",
        "runtime_version": "1.0",
        "generation_config": {"max_tokens": 256},
        "now": NOW,
    }


class PromptValidationTests(unittest.TestCase):
    def test_rejects_duplicate_request_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate request_id"):
            validate_prompt_records([prompt(), prompt()])

    def test_rejects_unexpected_fields(self):
        record = prompt()
        record["detector_score"] = 0.99
        with self.assertRaisesRegex(ValueError, "unexpected fields detector_score"):
            validate_prompt_records([record])


class StageZeroTests(unittest.TestCase):
    def test_retains_exact_first_response_and_calls_generator_once(self):
        calls = []
        text = "word " * 150

        def fake_generator(prompt_payload, config):
            calls.append((prompt_payload, config))
            return {
                "text": text,
                "prompt_tokens": 12,
                "generation_tokens": 150,
                "finish_reason": "stop",
                "elapsed_seconds": 1.5,
                "tokens_per_second": 100.0,
                "peak_memory_gb": 2.0,
            }

        receipts = run_stage_zero([prompt()], fake_generator, **metadata())

        self.assertEqual(len(calls), 1)
        self.assertEqual(receipts[0]["output_text"], text)
        self.assertEqual(receipts[0]["status"], "success")
        self.assertTrue(receipts[0]["length_in_band"])
        self.assertNotIn("detector_score", receipts[0])

    def test_preserves_failure_without_fabricating_output(self):
        def failing_generator(_prompt_payload, _config):
            raise RuntimeError("out of memory")

        receipt = run_stage_zero(
            [prompt()], failing_generator, **metadata()
        )[0]

        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["error_type"], "RuntimeError")
        self.assertEqual(receipt["error_message"], "out of memory")
        self.assertIsNone(receipt["output_text"])
        self.assertIsNone(receipt["output_sha256"])

    def test_success_and_failure_receipts_match_schema(self):
        schema_path = (
            Path(__file__).parents[1]
            / "benchmark"
            / "local_generation_receipt_schema.json"
        )
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        success = run_stage_zero(
            [prompt()], lambda *_args: {"text": "word " * 150}, **metadata()
        )
        failure = run_stage_zero(
            [prompt("zh")],
            lambda *_args: (_ for _ in ()).throw(ValueError("failed")),
            **metadata(),
        )

        validate_receipts(success + failure, schema)


class PreflightTests(unittest.TestCase):
    def test_reports_disk_and_runtime_failures_without_mutating_target(self):
        usage = shutil_usage(total=100, used=90, free=10)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models" / "qwen"
            with patch("aidetect.local_generation.shutil.disk_usage", return_value=usage):
                with patch(
                    "aidetect.local_generation.importlib.util.find_spec",
                    return_value=None,
                ):
                    report = build_preflight(target, minimum_free_gib=20 / (1024**3))

            self.assertFalse(target.exists())

        self.assertFalse(report["ready"])
        self.assertFalse(report["checks"]["disk_headroom"])
        self.assertFalse(report["checks"]["runtime_available"])


def shutil_usage(total, used, free):
    return type("DiskUsage", (), {"total": total, "used": used, "free": free})()


if __name__ == "__main__":
    unittest.main()
