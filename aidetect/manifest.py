"""Validation and deterministic, leakage-aware benchmark splitting."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


SPLITTER_VERSION = "0.1"


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def load_jsonl(path):
    """Load a UTF-8 JSONL manifest and return records plus the file hash."""
    path = Path(path)
    content = path.read_bytes()
    records = []
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        records.append(record)
    if not records:
        raise ValueError("Manifest contains no records")
    return records, sha256_bytes(content)


def load_schema(path):
    path = Path(path)
    content = path.read_bytes()
    return json.loads(content), sha256_bytes(content)


def validate_manifest_records(
    records,
    schema,
    text_root,
    verify_text_files=True,
):
    """Validate schema, lineage references, hashes, and exact duplicates."""
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    document_ids = {}
    text_hashes = {}
    text_root = Path(text_root)

    for record_number, record in enumerate(records, 1):
        errors = sorted(
            validator.iter_errors(record),
            key=lambda error: tuple(str(part) for part in error.path),
        )
        if errors:
            error = errors[0]
            field = ".".join(str(part) for part in error.path) or "record"
            raise ValueError(f"record {record_number} ({field}): {error.message}")

        document_id = record["document_id"]
        if document_id in document_ids:
            raise ValueError(f"duplicate document_id: {document_id!r}")
        document_ids[document_id] = record

        text_hash = record["text_sha256"].lower()
        if text_hash in text_hashes:
            raise ValueError(
                "exact duplicate text_sha256 for "
                f"{text_hashes[text_hash]!r} and {document_id!r}"
            )
        text_hashes[text_hash] = document_id

        if verify_text_files:
            text_path = Path(record["text_path"])
            if not text_path.is_absolute():
                text_path = text_root / text_path
            if not text_path.is_file():
                raise ValueError(f"{document_id!r}: text file not found: {text_path}")
            text_bytes = text_path.read_bytes()
            try:
                text_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(f"{document_id!r}: text file is not valid UTF-8") from exc
            actual_hash = sha256_bytes(text_bytes)
            if actual_hash != text_hash:
                raise ValueError(
                    f"{document_id!r}: text_sha256 mismatch "
                    f"(expected {text_hash}, got {actual_hash})"
                )

    for record in records:
        parent_id = record.get("parent_document_id")
        if parent_id is None:
            continue
        if parent_id not in document_ids:
            raise ValueError(
                f"{record['document_id']!r}: unknown parent_document_id {parent_id!r}"
            )
        if document_ids[parent_id]["source_id"] != record["source_id"]:
            raise ValueError(
                f"{record['document_id']!r}: parent_document_id must share source_id"
            )


def _source_components(records):
    sources = {record["source_id"] for record in records}
    parent = {source_id: source_id for source_id in sources}

    def find(source_id):
        while parent[source_id] != source_id:
            parent[source_id] = parent[parent[source_id]]
            source_id = parent[source_id]
        return source_id

    def union(left, right):
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    first_source_by_cluster = {}
    for record in records:
        cluster_id = record["near_duplicate_cluster_id"]
        if cluster_id is None:
            continue
        source_id = record["source_id"]
        if cluster_id in first_source_by_cluster:
            union(source_id, first_source_by_cluster[cluster_id])
        else:
            first_source_by_cluster[cluster_id] = source_id

    components = defaultdict(list)
    for index, record in enumerate(records):
        components[find(record["source_id"])].append(index)
    return list(components.values())


def _component_key(records, indices):
    source_ids = sorted({records[index]["source_id"] for index in indices})
    cluster_ids = sorted(
        {
            records[index]["near_duplicate_cluster_id"]
            for index in indices
            if records[index]["near_duplicate_cluster_id"] is not None
        }
    )
    return json.dumps(
        {"source_ids": source_ids, "near_duplicate_cluster_ids": cluster_ids},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _hash_fraction(seed, component_key):
    digest = hashlib.sha256(f"{seed}\0{component_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _component_split_counts(component_count, train_fraction, calibration_fraction):
    if component_count < 3:
        raise ValueError(
            "at least three formal non-held-out leakage components are required"
        )
    fractions = (
        train_fraction,
        calibration_fraction,
        1 - train_fraction - calibration_fraction,
    )
    raw_counts = [component_count * fraction for fraction in fractions]
    counts = [int(value) for value in raw_counts]
    remaining = component_count - sum(counts)
    remainder_order = sorted(
        range(3),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in remainder_order[:remaining]:
        counts[index] += 1

    for empty_index, count in enumerate(counts):
        if count != 0:
            continue
        donor_index = max(range(3), key=lambda index: (counts[index], -index))
        if counts[donor_index] <= 1:
            raise ValueError("not enough leakage components to populate every split")
        counts[donor_index] -= 1
        counts[empty_index] = 1
    return counts


def assign_splits(
    records,
    seed,
    train_fraction,
    calibration_fraction,
    held_out_generators,
):
    """Assign source-disjoint splits and isolate specified AI generators."""
    if train_fraction <= 0 or calibration_fraction <= 0:
        raise ValueError("train and calibration fractions must be greater than zero")
    if train_fraction + calibration_fraction >= 1:
        raise ValueError("train and calibration fractions must sum to less than one")

    held_out_generators = set(held_out_generators)
    if not held_out_generators:
        raise ValueError("at least one held-out generator is required")

    assigned = [dict(record) for record in records]
    found_held_out_generators = set()
    in_distribution_components = []

    for indices in _source_components(records):
        dry_run_flags = [records[index]["split"] == "dry_run" for index in indices]
        if any(dry_run_flags) and not all(dry_run_flags):
            raise ValueError(
                "one source/near-duplicate component mixes dry_run and formal records"
            )
        if all(dry_run_flags):
            for index in indices:
                assigned[index]["split"] = "dry_run"
                assigned[index]["evaluation_partition"] = "pipeline_dry_run"
            continue

        ai_generators = {
            records[index]["generator"]
            for index in indices
            if records[index]["human_or_ai"] == "ai"
        }
        matched_generators = ai_generators & held_out_generators
        if matched_generators:
            non_held_out_generators = ai_generators - held_out_generators
            if non_held_out_generators:
                raise ValueError(
                    "a source/near-duplicate component mixes held-out and "
                    "non-held-out AI generators"
                )
            found_held_out_generators.update(matched_generators)
            for index in indices:
                assigned[index]["split"] = "test"
                assigned[index]["evaluation_partition"] = "generator_held_out"
        else:
            in_distribution_components.append(indices)

    missing_generators = held_out_generators - found_held_out_generators
    if missing_generators:
        missing = ", ".join(sorted(missing_generators))
        raise ValueError(f"held-out generator not found in formal AI records: {missing}")

    component_counts = _component_split_counts(
        len(in_distribution_components), train_fraction, calibration_fraction
    )
    ordered_components = sorted(
        in_distribution_components,
        key=lambda indices: (
            _hash_fraction(seed, _component_key(records, indices)),
            _component_key(records, indices),
        ),
    )
    split_labels = (
        ["train"] * component_counts[0]
        + ["calibration"] * component_counts[1]
        + ["test"] * component_counts[2]
    )
    for indices, split in zip(ordered_components, split_labels):
        for index in indices:
            assigned[index]["split"] = split
            assigned[index]["evaluation_partition"] = "in_distribution"

    verify_split_integrity(assigned, held_out_generators)
    return assigned


def verify_split_integrity(records, held_out_generators):
    """Reject source, duplicate-cluster, or held-out-generator leakage."""
    source_splits = defaultdict(set)
    cluster_splits = defaultdict(set)
    for record in records:
        source_splits[record["source_id"]].add(record["split"])
        cluster_id = record["near_duplicate_cluster_id"]
        if cluster_id is not None:
            cluster_splits[cluster_id].add(record["split"])
        if record["human_or_ai"] == "ai" and record["generator"] in held_out_generators:
            if (
                record["split"] != "test"
                or record["evaluation_partition"] != "generator_held_out"
            ):
                raise ValueError(
                    f"held-out generator leakage in {record['document_id']!r}"
                )

    leaking_sources = sorted(key for key, values in source_splits.items() if len(values) > 1)
    if leaking_sources:
        raise ValueError(f"source_id crosses splits: {leaking_sources[0]!r}")
    leaking_clusters = sorted(
        key for key, values in cluster_splits.items() if len(values) > 1
    )
    if leaking_clusters:
        raise ValueError(
            f"near_duplicate_cluster_id crosses splits: {leaking_clusters[0]!r}"
        )


def write_jsonl(path, records):
    content = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    ).encode("utf-8")
    Path(path).write_bytes(content)
    return sha256_bytes(content)


def split_summary(records):
    component_splits = Counter()
    for indices in _source_components(records):
        component_splits[records[indices[0]]["split"]] += 1
    return {
        "record_count": len(records),
        "source_count": len({record["source_id"] for record in records}),
        "leakage_component_count": len(_source_components(records)),
        "records_by_split": dict(sorted(Counter(r["split"] for r in records).items())),
        "leakage_components_by_split": dict(sorted(component_splits.items())),
        "records_by_evaluation_partition": dict(
            sorted(Counter(r["evaluation_partition"] for r in records).items())
        ),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate and deterministically split an AIDetect manifest."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--text-root", type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-fraction", required=True, type=float)
    parser.add_argument("--calibration-fraction", required=True, type=float)
    parser.add_argument(
        "--held-out-generator",
        action="append",
        required=True,
        help="Generator reserved for test; repeat for multiple generators.",
    )
    parser.add_argument(
        "--skip-text-file-checks",
        action="store_true",
        help="Validate metadata only without opening text_path files.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    resolved_paths = {
        args.input.resolve(),
        args.output.resolve(),
        args.metadata_output.resolve(),
    }
    if len(resolved_paths) != 3:
        parser.error("input, output, and metadata-output must be different files")

    try:
        records, input_hash = load_jsonl(args.input)
        schema, schema_hash = load_schema(args.schema)
        text_root = args.text_root or args.input.parent
        validate_manifest_records(
            records,
            schema,
            text_root=text_root,
            verify_text_files=not args.skip_text_file_checks,
        )
        assigned = assign_splits(
            records,
            seed=args.seed,
            train_fraction=args.train_fraction,
            calibration_fraction=args.calibration_fraction,
            held_out_generators=args.held_out_generator,
        )
        validate_manifest_records(
            assigned,
            schema,
            text_root=text_root,
            verify_text_files=not args.skip_text_file_checks,
        )
        output_hash = write_jsonl(args.output, assigned)
        metadata = {
            "splitter_version": SPLITTER_VERSION,
            "assignment_method": "sha256_seeded_source_component_order_v1",
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "calibration_fraction": args.calibration_fraction,
            "test_fraction": round(
                1 - args.train_fraction - args.calibration_fraction, 12
            ),
            "held_out_generators": sorted(set(args.held_out_generator)),
            "input_manifest_sha256": input_hash,
            "schema_sha256": schema_hash,
            "output_manifest_sha256": output_hash,
            "text_files_verified": not args.skip_text_file_checks,
            **split_summary(assigned),
        }
        args.metadata_output.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    print(f"Wrote {len(assigned)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
