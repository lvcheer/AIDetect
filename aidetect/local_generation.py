"""Auditable preparation primitives for local-only text generation."""

import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


RECEIPT_SCHEMA_VERSION = "0.1"
LENGTH_BANDS = {
    "zh": {"short": (250, 399), "medium": (400, 699), "long": (700, 1100)},
    "en": {"short": (150, 239), "medium": (240, 419), "long": (420, 700)},
}
DOMAINS = {"general", "academic", "professional"}
REQUIRED_PROMPT_FIELDS = {
    "request_id",
    "prompt_family_id",
    "language",
    "domain",
    "length_band",
    "system_prompt",
    "user_prompt",
}


def canonical_sha256(value):
    """Hash JSON-compatible data using a stable UTF-8 representation."""
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_prompt_records(records):
    """Validate stage-zero prompts without reading or generating model output."""
    seen_ids = set()
    validated = []
    for index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"prompt {index}: expected an object")
        missing = sorted(REQUIRED_PROMPT_FIELDS - record.keys())
        if missing:
            raise ValueError(f"prompt {index}: missing fields {', '.join(missing)}")
        unexpected = sorted(record.keys() - REQUIRED_PROMPT_FIELDS)
        if unexpected:
            raise ValueError(f"prompt {index}: unexpected fields {', '.join(unexpected)}")
        for field in ("request_id", "prompt_family_id", "system_prompt", "user_prompt"):
            if not isinstance(record[field], str) or not record[field].strip():
                raise ValueError(f"prompt {index}: {field} must be a non-empty string")
        if record["request_id"] in seen_ids:
            raise ValueError(f"duplicate request_id: {record['request_id']!r}")
        if record["language"] not in LENGTH_BANDS:
            raise ValueError(f"prompt {index}: unsupported language")
        if record["domain"] not in DOMAINS:
            raise ValueError(f"prompt {index}: unsupported domain")
        if record["length_band"] not in LENGTH_BANDS[record["language"]]:
            raise ValueError(f"prompt {index}: unsupported length band")
        seen_ids.add(record["request_id"])
        validated.append(dict(record))
    if not validated:
        raise ValueError("no prompt records supplied")
    return validated


def measure_length(text, language):
    """Measure Chinese CJK characters or whitespace-delimited English words."""
    if language == "zh":
        return sum("\u3400" <= character <= "\u9fff" for character in text)
    return len(text.split())


def _empty_result_fields():
    return {
        "output_text": None,
        "output_sha256": None,
        "prompt_tokens": None,
        "generation_tokens": None,
        "finish_reason": None,
        "elapsed_seconds": None,
        "tokens_per_second": None,
        "peak_memory_gb": None,
        "measured_length": None,
        "length_in_band": None,
    }


def run_stage_zero(
    prompts,
    generator,
    *,
    model_id,
    model_revision,
    runtime,
    runtime_version,
    generation_config,
    now=None,
):
    """Call a local generator exactly once per prompt and retain every outcome."""
    prompts = validate_prompt_records(prompts)
    config_hash = canonical_sha256(generation_config)
    timestamp = (now or datetime.now(timezone.utc)).isoformat()
    receipts = []

    for prompt in prompts:
        prompt_payload = {
            "system_prompt": prompt["system_prompt"],
            "user_prompt": prompt["user_prompt"],
        }
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "request_id": prompt["request_id"],
            "prompt_family_id": prompt["prompt_family_id"],
            "language": prompt["language"],
            "domain": prompt["domain"],
            "length_band": prompt["length_band"],
            "prompt_sha256": canonical_sha256(prompt_payload),
            "model_id": model_id,
            "model_revision": model_revision,
            "runtime": runtime,
            "runtime_version": runtime_version,
            "generation_config_sha256": config_hash,
            "generated_at": timestamp,
            "status": "failed",
            **_empty_result_fields(),
            "error_type": None,
            "error_message": None,
        }
        try:
            result = generator(prompt_payload, generation_config)
            output_text = result["text"]
            if not isinstance(output_text, str) or not output_text.strip():
                raise ValueError("generator returned empty text")
            measured = measure_length(output_text, prompt["language"])
            minimum, maximum = LENGTH_BANDS[prompt["language"]][prompt["length_band"]]
            receipt.update(
                status="success",
                output_text=output_text,
                output_sha256=text_sha256(output_text),
                prompt_tokens=result.get("prompt_tokens"),
                generation_tokens=result.get("generation_tokens"),
                finish_reason=result.get("finish_reason"),
                elapsed_seconds=result.get("elapsed_seconds"),
                tokens_per_second=result.get("tokens_per_second"),
                peak_memory_gb=result.get("peak_memory_gb"),
                measured_length=measured,
                length_in_band=minimum <= measured <= maximum,
            )
        except Exception as exc:  # preserve local runtime failures as data
            receipt["error_type"] = type(exc).__name__
            receipt["error_message"] = str(exc) or type(exc).__name__
        receipts.append(receipt)
    return receipts


def validate_receipts(receipts, schema):
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    for index, receipt in enumerate(receipts, 1):
        errors = sorted(validator.iter_errors(receipt), key=lambda error: list(error.path))
        if errors:
            error = errors[0]
            field = ".".join(str(part) for part in error.path) or "record"
            raise ValueError(f"receipt {index} ({field}): {error.message}")


def build_preflight(target_dir, *, minimum_free_gib=20, runtime_module="mlx_lm"):
    """Return a read-only local runtime and disk readiness report."""
    target = Path(target_dir).expanduser().resolve()
    existing_target = target
    while not existing_target.exists() and existing_target != existing_target.parent:
        existing_target = existing_target.parent
    usage = shutil.disk_usage(existing_target)
    free_gib = usage.free / (1024**3)
    runtime_available = importlib.util.find_spec(runtime_module) is not None
    checks = {
        "disk_headroom": free_gib >= minimum_free_gib,
        "runtime_available": runtime_available,
    }
    return {
        "preflight_version": "0.1",
        "target_dir": str(target),
        "checked_filesystem_path": str(existing_target),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "runtime_module": runtime_module,
        "minimum_free_gib": minimum_free_gib,
        "free_gib": round(free_gib, 2),
        "checks": checks,
        "ready": all(checks.values()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Read-only preflight for the AIDetect local generator pilot."
    )
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=20)
    parser.add_argument("--runtime-module", default="mlx_lm")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.minimum_free_gib < 0:
        parser.error("--minimum-free-gib must be non-negative")

    report = build_preflight(
        args.target_dir,
        minimum_free_gib=args.minimum_free_gib,
        runtime_module=args.runtime_module,
    )
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(content, encoding="utf-8")
    else:
        sys.stdout.write(content)
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
