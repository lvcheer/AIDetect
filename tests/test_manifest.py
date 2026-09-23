import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from aidetect.manifest import (
    assign_dry_run,
    assign_splits,
    load_schema,
    main,
    validate_manifest_records,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "dataset_manifest_schema.json"
)


def make_record(root, document_id, source_id, text, generator=None, **updates):
    text_directory = root / "texts"
    text_directory.mkdir(exist_ok=True)
    relative_path = Path("texts") / f"{document_id}.txt"
    text_bytes = text.encode("utf-8")
    (root / relative_path).write_bytes(text_bytes)
    is_ai = generator is not None
    record = {
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
    record.update(updates)
    return record


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.schema, _schema_hash = load_schema(SCHEMA_PATH)

    def test_validates_schema_lineage_and_text_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = make_record(root, "doc-1", "source-1", "Human text one.")
            child = make_record(
                root,
                "doc-2",
                "source-1",
                "AI rewrite two.",
                generator="generator-a",
                parent_document_id="doc-1",
            )

            validate_manifest_records([parent, child], self.schema, root)

    def test_rejects_missing_schema_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = make_record(root, "doc-1", "source-1", "Text one.")
            del record["license"]

            with self.assertRaisesRegex(ValueError, "license"):
                validate_manifest_records([record], self.schema, root)

    def test_rejects_exact_duplicate_text_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = make_record(root, "doc-1", "source-1", "Same text.")
            second = make_record(root, "doc-2", "source-2", "Same text.")

            with self.assertRaisesRegex(ValueError, "exact duplicate"):
                validate_manifest_records([first, second], self.schema, root)


class ManifestSplitTests(unittest.TestCase):
    def test_dry_run_only_accepts_only_pipeline_dry_run_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dry_run = make_record(
                root,
                "dry-1",
                "source-dry",
                "Dry run text.",
                split="dry_run",
                evaluation_partition="pipeline_dry_run",
            )

            assigned = assign_dry_run([dry_run])
            self.assertEqual(assigned, [dry_run])

            formal = make_record(root, "formal-1", "source-formal", "Formal text.")
            with self.assertRaisesRegex(ValueError, "only dry_run records"):
                assign_dry_run([formal])

    def test_rejects_too_few_non_held_out_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                make_record(root, "human-1", "source-1", "Human text one."),
                make_record(root, "ai-1", "source-2", "AI text one.", "generator-a"),
                make_record(root, "ai-2", "source-3", "AI text two.", "generator-b"),
            ]

            with self.assertRaisesRegex(ValueError, "at least three"):
                assign_splits(records, 2026, 0.6, 0.2, ["generator-b"])

    def test_split_is_deterministic_and_leakage_groups_stay_together(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                make_record(
                    root,
                    "human-1",
                    "source-1",
                    "Human text one.",
                    near_duplicate_cluster_id="cluster-1",
                ),
                make_record(
                    root,
                    "human-2",
                    "source-2",
                    "Human text two.",
                    near_duplicate_cluster_id="cluster-1",
                ),
                make_record(root, "ai-1", "source-3", "AI text one.", "generator-a"),
                make_record(root, "ai-2", "source-4", "AI text two.", "generator-b"),
                make_record(root, "human-3", "source-8", "Human text three."),
            ]

            assigned = assign_splits(records, 2026, 0.6, 0.2, ["generator-b"])
            reversed_assigned = assign_splits(
                list(reversed(records)), 2026, 0.6, 0.2, ["generator-b"]
            )
            by_id = {record["document_id"]: record for record in assigned}
            reversed_by_id = {
                record["document_id"]: record for record in reversed_assigned
            }

        self.assertEqual(by_id["human-1"]["split"], by_id["human-2"]["split"])
        self.assertEqual(by_id["ai-2"]["split"], "test")
        self.assertEqual(
            by_id["ai-2"]["evaluation_partition"], "generator_held_out"
        )
        self.assertEqual(
            {
                record["split"]
                for record in assigned
                if record["evaluation_partition"] == "in_distribution"
            },
            {"train", "calibration", "test"},
        )
        self.assertEqual(
            {key: value["split"] for key, value in by_id.items()},
            {key: value["split"] for key, value in reversed_by_id.items()},
        )

    def test_preserves_isolated_dry_run_and_rejects_mixed_component(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dry_run = make_record(
                root,
                "dry-1",
                "source-dry",
                "Dry run text.",
                split="dry_run",
                evaluation_partition="pipeline_dry_run",
            )
            formal = make_record(
                root,
                "formal-1",
                "source-formal",
                "Formal text.",
                "generator-b",
            )
            supporting_records = [
                make_record(root, "human-1", "source-1", "Human text one."),
                make_record(root, "human-2", "source-2", "Human text two."),
                make_record(root, "ai-1", "source-3", "AI text one.", "generator-a"),
            ]
            assigned = assign_splits(
                [dry_run, formal, *supporting_records],
                2026,
                0.6,
                0.2,
                ["generator-b"],
            )
            self.assertEqual(assigned[0]["split"], "dry_run")

            formal["source_id"] = "source-dry"
            with self.assertRaisesRegex(ValueError, "mixes dry_run"):
                assign_splits(
                    [dry_run, formal, *supporting_records],
                    2026,
                    0.6,
                    0.2,
                    ["generator-b"],
                )

    def test_cli_writes_hashed_manifest_and_split_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidate.jsonl"
            output_path = root / "frozen.jsonl"
            metadata_path = root / "split_metadata.json"
            records = [
                make_record(root, "human-1", "source-1", "Human text one."),
                make_record(root, "human-2", "source-2", "Human text two."),
                make_record(root, "ai-1", "source-3", "AI text one.", "generator-a"),
                make_record(root, "ai-2", "source-4", "AI text two.", "generator-b"),
            ]
            input_content = "".join(
                json.dumps(record, ensure_ascii=False) + "\n" for record in records
            )
            input_path.write_text(input_content, encoding="utf-8")

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--metadata-output",
                    str(metadata_path),
                    "--schema",
                    str(SCHEMA_PATH),
                    "--seed",
                    "2026",
                    "--train-fraction",
                    "0.6",
                    "--calibration-fraction",
                    "0.2",
                    "--held-out-generator",
                    "generator-b",
                ]
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            output_bytes = output_path.read_bytes()

        self.assertEqual(exit_code, 0)
        self.assertEqual(metadata["record_count"], 4)
        self.assertEqual(
            metadata["input_manifest_sha256"],
            hashlib.sha256(input_content.encode("utf-8")).hexdigest(),
        )
        self.assertEqual(
            metadata["output_manifest_sha256"],
            hashlib.sha256(output_bytes).hexdigest(),
        )
        self.assertEqual(metadata["held_out_generators"], ["generator-b"])

    def test_cli_writes_dry_run_only_manifest_without_formal_split_options(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidate.jsonl"
            output_path = root / "frozen.jsonl"
            metadata_path = root / "split_metadata.json"
            records = [
                make_record(
                    root,
                    "dry-1",
                    "source-dry-1",
                    "Dry run human text.",
                    split="dry_run",
                    evaluation_partition="pipeline_dry_run",
                ),
                make_record(
                    root,
                    "dry-2",
                    "source-dry-2",
                    "Dry run AI text.",
                    "generator-a",
                    split="dry_run",
                    evaluation_partition="pipeline_dry_run",
                ),
            ]
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--metadata-output",
                    str(metadata_path),
                    "--schema",
                    str(SCHEMA_PATH),
                    "--dry-run-only",
                ]
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            frozen_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(metadata["manifest_mode"], "dry_run_only")
        self.assertEqual(metadata["assignment_method"], "validated_dry_run_only_v1")
        self.assertEqual(metadata["held_out_generators"], [])
        self.assertIsNone(metadata["seed"])
        self.assertEqual({record["split"] for record in frozen_records}, {"dry_run"})


if __name__ == "__main__":
    unittest.main()
