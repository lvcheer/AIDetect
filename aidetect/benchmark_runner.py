"""Auditable per-sample runner for a frozen AIDetect benchmark manifest."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .features import calculate_burstiness_feature, calculate_perplexity_feature
from .inference import infer_raw_score, inspect_tokenization
from .manifest import (
    load_jsonl,
    load_schema,
    sha256_bytes,
    validate_manifest_records,
    verify_split_integrity,
    write_jsonl,
)
from .models import (
    MODEL_REGISTRY,
    PERPLEXITY_MODEL_ID,
    load_classifier,
    load_perplexity_model,
)


RUNNER_VERSION = "0.1"
RESULT_SCHEMA_VERSION = "0.1"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
COMMIT_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def canonical_json_hash(value):
    content = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(content)


def load_json_file(path):
    content = Path(path).read_bytes()
    value = json.loads(content.decode("utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value, sha256_bytes(content)


def hash_directory(path):
    """Hash relative file names and contents for a local model directory."""
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative_path = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(file_path.stat().st_size.to_bytes(8, "big"))
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_manifest(manifest_hash, schema_hash, split_metadata, records):
    if split_metadata.get("output_manifest_sha256") != manifest_hash:
        raise ValueError("manifest hash does not match split metadata")
    if split_metadata.get("schema_sha256") != schema_hash:
        raise ValueError("schema hash does not match split metadata")
    held_out_generators = split_metadata.get("held_out_generators")
    if split_metadata.get("manifest_mode") == "dry_run_only":
        if held_out_generators != []:
            raise ValueError("dry-run-only split metadata must use no held-out generators")
        if any(
            record["split"] != "dry_run"
            or record["evaluation_partition"] != "pipeline_dry_run"
            for record in records
        ):
            raise ValueError("dry-run-only metadata requires only dry_run records")
    elif not isinstance(held_out_generators, list) or not held_out_generators:
        raise ValueError("split metadata must list held_out_generators")
    verify_split_integrity(records, set(held_out_generators))


def _error_text(exc):
    return f"{type(exc).__name__}: {exc}"


def _manifest_fields(record):
    return {
        "document_id": record["document_id"],
        "source_id": record["source_id"],
        "language": record["language"],
        "domain": record["domain"],
        "reference_label": record["human_or_ai"],
        "generator": record["generator"],
        "generator_revision": record["generator_revision"],
        "editing_condition": record["editing_condition"],
        "split": record["split"],
        "evaluation_partition": record["evaluation_partition"],
        "text_sha256": record["text_sha256"].lower(),
    }


def run_record(
    manifest_record,
    text,
    classifier,
    run_context,
    max_length,
    perplexity=None,
    clock=time.perf_counter,
):
    """Score one manifest record without converting failures into scores."""
    total_start = clock()
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        **_manifest_fields(manifest_record),
        **run_context,
        "character_count": len(text),
        "input_token_length": None,
        "effective_token_length": None,
        "max_token_length": max_length,
        "truncated": None,
        "truncation_side": None,
        "classifier_class_count": None,
        "classifier_class_scores": None,
        "raw_ai_score": None,
        "score_scale": "unit_interval_0_1",
        "score_direction": "higher_is_more_ai_like",
        "classifier_latency_ms": None,
        "perplexity_value": None,
        "perplexity_heuristic_score": None,
        "perplexity_latency_ms": None,
        "burstiness_cv": None,
        "burstiness_heuristic_score": None,
        "classifier_error": None,
        "perplexity_error": None,
        "burstiness_error": None,
        "status": "success",
        "total_latency_ms": None,
    }

    classifier_start = clock()
    try:
        tokenization = inspect_tokenization(
            text,
            tokenizer=classifier.tokenizer,
            max_length=max_length,
        )
        result.update(
            {
                "input_token_length": tokenization.input_token_length,
                "effective_token_length": tokenization.effective_token_length,
                "truncated": tokenization.truncated,
                "truncation_side": tokenization.truncation_side,
            }
        )
        raw_score = infer_raw_score(
            text,
            tokenizer=classifier.tokenizer,
            model=classifier.model,
            device=classifier.device,
            ai_label_index=classifier.ai_label_index,
            max_length=max_length,
        )
        result["classifier_class_count"] = len(raw_score.scores)
        result["classifier_class_scores"] = list(raw_score.scores)
        result["raw_ai_score"] = raw_score.ai_score
    except Exception as exc:
        result["classifier_error"] = _error_text(exc)
    result["classifier_latency_ms"] = round((clock() - classifier_start) * 1000, 3)

    try:
        burstiness = calculate_burstiness_feature(text)
        if burstiness is not None:
            result["burstiness_cv"] = burstiness.coefficient_of_variation
            result["burstiness_heuristic_score"] = burstiness.heuristic_score
    except Exception as exc:
        result["burstiness_error"] = _error_text(exc)

    if perplexity is not None:
        perplexity_start = clock()
        try:
            feature = calculate_perplexity_feature(
                text,
                tokenizer=perplexity.tokenizer,
                model=perplexity.model,
                max_length=max_length,
            )
            result["perplexity_value"] = feature.perplexity
            result["perplexity_heuristic_score"] = feature.heuristic_score
        except Exception as exc:
            result["perplexity_error"] = _error_text(exc)
        result["perplexity_latency_ms"] = round(
            (clock() - perplexity_start) * 1000, 3
        )

    if result["classifier_error"] is not None:
        result["status"] = "failed"
    elif result["perplexity_error"] or result["burstiness_error"]:
        result["status"] = "partial_failure"
    result["total_latency_ms"] = round((clock() - total_start) * 1000, 3)
    return result


def run_benchmark(
    records,
    text_root,
    classifier,
    run_context,
    max_length,
    included_splits=None,
    perplexity=None,
):
    """Run records sequentially so failures remain attributable per sample."""
    included_splits = set(included_splits or [])
    outputs = []
    for record in records:
        if included_splits and record["split"] not in included_splits:
            continue
        text_path = Path(record["text_path"])
        if not text_path.is_absolute():
            text_path = Path(text_root) / text_path
        text = text_path.read_text(encoding="utf-8")
        outputs.append(
            run_record(
                record,
                text,
                classifier=classifier,
                run_context=run_context,
                max_length=max_length,
                perplexity=perplexity,
            )
        )
    if not outputs:
        raise ValueError("No manifest records match the selected splits")
    return outputs


def _class_labels(model):
    labels = getattr(getattr(model, "config", None), "id2label", {})
    return {str(index): str(label) for index, label in labels.items()}


def _ai_label_name(classifier):
    labels = _class_labels(classifier.model)
    return labels.get(str(classifier.ai_label_index))


def collect_runtime_environment():
    packages = {}
    for package in ("jsonschema", "matplotlib", "pandas", "torch", "transformers"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run an auditable detector baseline on a frozen manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split-metadata", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--text-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-metadata-output", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--ai-label-index", required=True, type=int)
    parser.add_argument("--max-length", required=True, type=int)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument(
        "--include-split",
        action="append",
        required=True,
        choices=("train", "calibration", "test", "dry_run"),
    )
    parser.add_argument("--perplexity", action="store_true")
    parser.add_argument("--perplexity-revision")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if not COMMIT_PATTERN.fullmatch(args.code_commit):
        parser.error("--code-commit must be a full 40-character Git commit")
    if not COMMIT_PATTERN.fullmatch(args.model_revision):
        parser.error("--model-revision must be a full 40-character commit")
    if args.max_length <= 0:
        parser.error("--max-length must be greater than zero")
    if args.perplexity and not args.perplexity_revision:
        parser.error("--perplexity requires --perplexity-revision")
    if args.perplexity_revision and not args.perplexity:
        parser.error("--perplexity-revision requires --perplexity")
    if args.perplexity_revision and not COMMIT_PATTERN.fullmatch(
        args.perplexity_revision
    ):
        parser.error("--perplexity-revision must be a full 40-character commit")

    primary_paths = {
        args.manifest.resolve(),
        args.split_metadata.resolve(),
        args.schema.resolve(),
        args.output.resolve(),
        args.run_metadata_output.resolve(),
    }
    if len(primary_paths) != 5:
        parser.error("manifest, metadata, schema, and output paths must be distinct")

    started_at = datetime.now(timezone.utc).isoformat()
    try:
        records, manifest_hash = load_jsonl(args.manifest)
        schema, schema_hash = load_schema(args.schema)
        split_metadata, split_metadata_hash = load_json_file(args.split_metadata)
        text_root = args.text_root or args.manifest.parent
        validate_manifest_records(records, schema, text_root=text_root)
        verify_frozen_manifest(manifest_hash, schema_hash, split_metadata, records)

        model_id = MODEL_REGISTRY.get(args.model, args.model)
        classifier = load_classifier(
            model_id,
            args.models_dir,
            revision=args.model_revision,
            ai_label_index=args.ai_label_index,
        )
        local_model_hash = (
            hash_directory(classifier.source)
            if isinstance(classifier.source, Path)
            else None
        )
        immutable_model_revision = (
            f"local-sha256:{local_model_hash}"
            if local_model_hash
            else str(classifier.resolved_revision or args.model_revision)
        )
        ai_label_name = _ai_label_name(classifier)
        if ai_label_name is None:
            raise ValueError(
                f"AI label index {args.ai_label_index} is absent from model id2label"
            )

        perplexity = None
        if args.perplexity:
            perplexity = load_perplexity_model(
                device=classifier.device,
                revision=args.perplexity_revision,
            )
        immutable_perplexity_revision = (
            str(perplexity.resolved_revision or args.perplexity_revision)
            if perplexity
            else None
        )

        included_splits = sorted(set(args.include_split))
        configuration = {
            "runner_version": RUNNER_VERSION,
            "run_id": args.run_id,
            "code_commit": args.code_commit.lower(),
            "manifest_sha256": manifest_hash,
            "model_id": model_id,
            "requested_model_revision": args.model_revision.lower(),
            "immutable_model_revision": immutable_model_revision,
            "model_artifact_sha256": local_model_hash,
            "ai_label_index": args.ai_label_index,
            "ai_label_name": ai_label_name,
            "score_direction": "higher_is_more_ai_like",
            "max_length": args.max_length,
            "truncation_side": getattr(classifier.tokenizer, "truncation_side", None),
            "included_splits": included_splits,
            "perplexity_model_id": PERPLEXITY_MODEL_ID if perplexity else None,
            "requested_perplexity_revision": (
                args.perplexity_revision.lower() if args.perplexity_revision else None
            ),
            "immutable_perplexity_revision": immutable_perplexity_revision,
        }
        configuration_hash = canonical_json_hash(configuration)
        run_context = {
            "run_id": args.run_id,
            "configuration_sha256": configuration_hash,
            "code_commit": args.code_commit.lower(),
            "manifest_sha256": manifest_hash,
            "model_id": model_id,
            "model_revision": immutable_model_revision,
            "model_artifact_sha256": local_model_hash,
            "device": classifier.device,
            "ai_label_index": classifier.ai_label_index,
            "ai_label_name": ai_label_name,
            "perplexity_model_id": PERPLEXITY_MODEL_ID if perplexity else None,
            "perplexity_model_revision": immutable_perplexity_revision,
        }
        output_records = run_benchmark(
            records,
            text_root=text_root,
            classifier=classifier,
            run_context=run_context,
            max_length=args.max_length,
            included_splits=included_splits,
            perplexity=perplexity,
        )
        output_hash = write_jsonl(args.output, output_records)
        run_metadata = {
            "run_metadata_schema_version": "0.1",
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration,
            "configuration_sha256": configuration_hash,
            "manifest_sha256": manifest_hash,
            "split_metadata_sha256": split_metadata_hash,
            "schema_sha256": schema_hash,
            "results_sha256": output_hash,
            "record_count": len(output_records),
            "records_by_status": dict(
                sorted(Counter(record["status"] for record in output_records).items())
            ),
            "model": {
                "model_id": model_id,
                "source": str(classifier.source),
                "requested_revision": args.model_revision.lower(),
                "resolved_revision": (
                    str(classifier.resolved_revision)
                    if classifier.resolved_revision is not None
                    else None
                ),
                "immutable_revision": immutable_model_revision,
                "artifact_sha256": local_model_hash,
                "device": classifier.device,
                "ai_label_index": classifier.ai_label_index,
                "ai_label_name": ai_label_name,
                "class_labels": _class_labels(classifier.model),
                "tokenizer_class": type(classifier.tokenizer).__name__,
                "tokenizer_model_max_length": getattr(
                    classifier.tokenizer, "model_max_length", None
                ),
                "truncation_side": getattr(
                    classifier.tokenizer, "truncation_side", None
                ),
            },
            "perplexity_model": (
                {
                    "model_id": PERPLEXITY_MODEL_ID,
                    "requested_revision": args.perplexity_revision.lower(),
                    "resolved_revision": (
                        str(perplexity.resolved_revision)
                        if perplexity.resolved_revision is not None
                        else None
                    ),
                    "device": perplexity.device,
                }
                if perplexity
                else None
            ),
            "runtime_environment": collect_runtime_environment(),
        }
        args.run_metadata_output.write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        parser.error(str(exc))

    print(f"Wrote {len(output_records)} benchmark records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
