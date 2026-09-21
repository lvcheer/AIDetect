"""Batch JSONL/CSV command-line interface for AIDetect."""

import argparse
import csv
import json
from pathlib import Path

from .models import MODEL_REGISTRY, load_classifier, load_perplexity_model
from .pipeline import detect_text


SUPPORTED_FORMATS = {".csv", ".jsonl"}
DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def _file_format(path):
    suffix = Path(path).suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError("Input and output files must use .jsonl or .csv")
    return suffix


def _validate_input_record(record, location, seen_ids):
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected an object or CSV row")

    document_id = record.get("document_id")
    text = record.get("text")
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError(f"{location}: document_id must be a non-empty string")
    if document_id in seen_ids:
        raise ValueError(f"{location}: duplicate document_id {document_id!r}")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{location}: text must be a non-empty string")

    seen_ids.add(document_id)
    return {"document_id": document_id, "text": text}


def read_input(path):
    """Read and validate JSONL or CSV records with document_id and text."""
    path = Path(path)
    file_format = _file_format(path)
    records = []
    seen_ids = set()

    if file_format == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"line {line_number}: invalid JSON") from exc
                records.append(
                    _validate_input_record(record, f"line {line_number}", seen_ids)
                )
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV input must include a header row")
            for row_number, record in enumerate(reader, 2):
                records.append(
                    _validate_input_record(record, f"row {row_number}", seen_ids)
                )

    if not records:
        raise ValueError("Input contains no records")
    return records


def detect_batch(records, classifier, model_id, perplexity=None):
    """Detect an iterable of input records and return flat output records."""
    outputs = []
    for input_record in records:
        detection = detect_text(
            input_record["text"],
            tokenizer=classifier.tokenizer,
            model=classifier.model,
            device=classifier.device,
            ai_label_index=classifier.ai_label_index,
            perplexity_tokenizer=(perplexity.tokenizer if perplexity else None),
            perplexity_model=(perplexity.model if perplexity else None),
        )
        output = {
            "document_id": input_record["document_id"],
            "model_id": model_id,
            "model_source": str(classifier.source),
            "device": classifier.device,
            "perplexity_model_id": perplexity.source if perplexity else None,
        }
        output.update(detection.to_dict())
        outputs.append(output)
    return outputs


def write_output(path, records):
    """Write flat result records as JSONL or CSV."""
    path = Path(path)
    file_format = _file_format(path)
    records = list(records)
    if not records:
        raise ValueError("Cannot write an empty result set")

    if file_format == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def resolve_model_id(model_argument):
    """Accept either a registry display name or a Hugging Face model ID."""
    return MODEL_REGISTRY.get(model_argument, model_argument)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one AIDetect classifier over JSONL or CSV records."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--model",
        required=True,
        help="Registry display name or Hugging Face model ID.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Directory containing downloaded classifier models.",
    )
    parser.add_argument(
        "--perplexity",
        action="store_true",
        help="Enable the existing GPT-2 perplexity heuristic.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")

    try:
        input_records = read_input(args.input)
        model_id = resolve_model_id(args.model)
        classifier = load_classifier(model_id, args.models_dir)
        perplexity = (
            load_perplexity_model(device=classifier.device)
            if args.perplexity
            else None
        )
        output_records = detect_batch(
            input_records,
            classifier=classifier,
            model_id=model_id,
            perplexity=perplexity,
        )
        write_output(args.output, output_records)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Wrote {len(output_records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
