import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

from aidetect.candidate_metadata import (
    build_summary,
    main,
    screen_candidate,
    validate_candidate_records,
)
from aidetect.manifest import load_schema


SCHEMA_PATH = (
    Path(__file__).resolve().parents[1]
    / "benchmark"
    / "candidate_metadata_schema.json"
)


def make_candidate(candidate_id="candidate-1", **updates):
    record = {
        "schema_version": "0.1",
        "candidate_id": candidate_id,
        "source": "example_repository",
        "record_uri": f"https://example.test/records/{candidate_id}",
        "title": "中文学术候选记录",
        "language": "zh",
        "subject": "linguistics",
        "document_type": "preprint",
        "first_deposit_date": "2021-04-10",
        "version_date": "2021-04-10",
        "version_id": "v1",
        "is_first_version": True,
        "license": "CC-BY-4.0",
        "license_evidence_uri": f"https://example.test/records/{candidate_id}#license",
        "peer_review_status": "not_peer_reviewed",
        "publisher_pdf_flag": False,
        "related_record_id": None,
        "screening_decision": "pending",
        "exclusion_reasons": [],
    }
    record.update(updates)
    return record


class CandidateMetadataTests(unittest.TestCase):
    def setUp(self):
        self.schema, _schema_hash = load_schema(SCHEMA_PATH)

    def test_accepts_valid_metadata_and_marks_it_eligible(self):
        record = make_candidate()
        validate_candidate_records([record], self.schema)

        screened = screen_candidate(record)

        self.assertEqual(screened["screening_decision"], "eligible")
        self.assertEqual(screened["exclusion_reasons"], [])
        validate_candidate_records([screened], self.schema)

    def test_records_every_applicable_exclusion_reason(self):
        record = make_candidate(
            language="en",
            first_deposit_date="2023-01-01",
            version_date="2023-02-01",
            is_first_version=False,
            license="CC-BY-NC-SA-4.0",
            publisher_pdf_flag=True,
        )

        screened = screen_candidate(record)

        self.assertEqual(screened["screening_decision"], "excluded")
        self.assertEqual(
            screened["exclusion_reasons"],
            [
                "non_chinese",
                "first_deposit_after_cutoff",
                "version_after_cutoff",
                "not_first_version",
                "license_not_allowed",
                "publisher_pdf",
            ],
        )

    def test_rejects_duplicate_record_versions(self):
        first = make_candidate("candidate-1")
        second = make_candidate("candidate-2", record_uri=first["record_uri"])

        with self.assertRaisesRegex(ValueError, "duplicate record_uri/version_id"):
            validate_candidate_records([first, second], self.schema)

    def test_builds_auditable_summary(self):
        eligible = screen_candidate(make_candidate())
        excluded = screen_candidate(
            make_candidate("candidate-2", language="en", license="other")
        )

        summary = build_summary([eligible, excluded], date(2022, 11, 30))

        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["decision_counts"], {"eligible": 1, "excluded": 1})
        self.assertEqual(
            summary["exclusion_reason_counts"],
            {"license_not_allowed": 1, "non_chinese": 1},
        )
        self.assertEqual(summary["eligible_by_subject"], {"linguistics": 1})

    def test_cli_writes_full_audit_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "candidates.jsonl"
            output_path = root / "screened.jsonl"
            summary_path = root / "summary.json"
            candidates = [
                make_candidate(),
                make_candidate("candidate-2", publisher_pdf_flag=True),
            ]
            input_path.write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in candidates),
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                    "--schema",
                    str(SCHEMA_PATH),
                ]
            )
            screened = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(screened), 2)
        self.assertEqual(screened[1]["exclusion_reasons"], ["publisher_pdf"])
        self.assertEqual(summary["decision_counts"], {"eligible": 1, "excluded": 1})


if __name__ == "__main__":
    unittest.main()
