"""Validate and screen benchmark candidate metadata without acquiring text."""

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from aidetect.manifest import load_jsonl, load_schema


DEFAULT_CUTOFF = date(2022, 11, 30)
ALLOWED_LICENSES = {"CC-BY-4.0", "CC0-1.0"}


def validate_candidate_records(records, schema):
    """Validate metadata records and identifiers without reading source text."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    candidate_ids = set()
    version_keys = set()

    for record_number, record in enumerate(records, 1):
        errors = sorted(
            validator.iter_errors(record),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            error = errors[0]
            field = ".".join(str(part) for part in error.path) or "record"
            raise ValueError(f"record {record_number} ({field}): {error.message}")

        candidate_id = record["candidate_id"]
        if candidate_id in candidate_ids:
            raise ValueError(f"duplicate candidate_id: {candidate_id!r}")
        candidate_ids.add(candidate_id)

        version_key = (record["record_uri"], record["version_id"])
        if version_key in version_keys:
            raise ValueError(
                "duplicate record_uri/version_id: "
                f"{record['record_uri']!r}, {record['version_id']!r}"
            )
        version_keys.add(version_key)


def screen_candidate(record, cutoff=DEFAULT_CUTOFF):
    """Return an audited copy with a deterministic eligibility decision."""
    reasons = []
    if record["language"] != "zh":
        reasons.append("non_chinese")
    if date.fromisoformat(record["first_deposit_date"]) > cutoff:
        reasons.append("first_deposit_after_cutoff")
    if date.fromisoformat(record["version_date"]) > cutoff:
        reasons.append("version_after_cutoff")
    if not record["is_first_version"]:
        reasons.append("not_first_version")
    if record["license"] not in ALLOWED_LICENSES:
        reasons.append("license_not_allowed")
    if record["publisher_pdf_flag"]:
        reasons.append("publisher_pdf")

    screened = dict(record)
    screened["screening_decision"] = "excluded" if reasons else "eligible"
    screened["exclusion_reasons"] = reasons
    return screened


def screen_records(records, cutoff=DEFAULT_CUTOFF):
    return [screen_candidate(record, cutoff) for record in records]


def build_summary(records, cutoff):
    decisions = Counter(record["screening_decision"] for record in records)
    reasons = Counter(
        reason for record in records for reason in record["exclusion_reasons"]
    )
    eligible = [record for record in records if record["screening_decision"] == "eligible"]
    return {
        "screening_version": "0.1",
        "cutoff_date": cutoff.isoformat(),
        "record_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "exclusion_reason_counts": dict(sorted(reasons.items())),
        "eligible_by_document_type": dict(
            sorted(Counter(record["document_type"] for record in eligible).items())
        ),
        "eligible_by_subject": dict(
            sorted(Counter(record["subject"] for record in eligible).items())
        ),
    }


def _write_jsonl(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    Path(path).write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate and screen metadata-only benchmark candidates."
    )
    parser.add_argument("--input", required=True, help="Candidate JSONL input")
    parser.add_argument("--output", required=True, help="Audited JSONL output")
    parser.add_argument("--summary-output", required=True, help="Summary JSON output")
    parser.add_argument("--schema", required=True, help="Candidate metadata JSON schema")
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF.isoformat())
    args = parser.parse_args(argv)

    try:
        cutoff = date.fromisoformat(args.cutoff)
    except ValueError as exc:
        parser.error(f"invalid --cutoff date: {args.cutoff!r}")

    records, _input_hash = load_jsonl(args.input)
    schema, _schema_hash = load_schema(args.schema)
    validate_candidate_records(records, schema)
    screened = screen_records(records, cutoff)
    validate_candidate_records(screened, schema)
    summary = build_summary(screened, cutoff)

    _write_jsonl(args.output, screened)
    Path(args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
