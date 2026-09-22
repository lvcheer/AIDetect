# Gate D-L: Local-Only Pilot Preparation

Preparation date: `2026-09-22`

Status: **prepared but blocked by disk headroom; no model downloaded or run**

## 1. Scope change

The user declined paid-model execution. OpenAI, Gemini, and Claude remain
documented Gate B candidates but are inactive and must not be called. The
three-provider confirmatory generator-generalisation study is suspended.

The next executable study is a smaller **exploratory local-model benchmark**.
It may measure detector behaviour on the selected open-weight checkpoint, but
it cannot support claims about commercial frontier models, provider-held-out
generalisation, or model-family generalisation. Restoring those claims requires
at least three eligible, family-disjoint generators and a newly approved plan.

## 2. Observed local environment

| Item | Observation |
|---|---|
| Computer | MacBook Pro `Mac14,9` |
| Processor | Apple M2 Pro, 10 CPU cores |
| Architecture | `arm64` |
| Unified memory | 16 GB |
| Python environment | project `.venv`, Python 3.11.15 |
| Candidate runtime | MLX / MLX-LM; not currently installed |
| Free disk at preflight | approximately 14 GiB; data volume 97% used |
| Existing Hugging Face cache | approximately 4.2 GB, containing detector models only |

This snapshot is informational and must be regenerated immediately before an
installation or model download.

## 3. Provisional checkpoint

The first candidate is `Qwen/Qwen3-1.7B` because its official model repository:

- identifies it as a multilingual text-generation model;
- declares Apache-2.0;
- is approximately 4.08 GB in its unquantized Hugging Face repository; and
- is supported by Transformers, while the official Qwen project documents
  Qwen3 support through MLX-LM on Apple Silicon.

The floating `main` branch is not a reproducible model revision. Before any
download, record and approve an exact Hugging Face commit; after download,
hash the licence, model card, configuration, tokenizer, generation
configuration, index, and every weight file. A community quantization is not
interchangeable with the official checkpoint: it requires its own provenance,
conversion settings, licence review, and quality pilot.

No second or third local family is named merely to preserve the earlier design.
Each additional family would need independent evidence for Chinese and English,
licensing, hardware feasibility, and a fixed checkpoint. Until then, Qwen3 is
one exploratory generator, not a held-out-generator design.

Official references:

- Qwen3 official repository and Apple-Silicon runtime guidance:
  <https://github.com/QwenLM/Qwen3>
- Qwen3-1.7B official checkpoint:
  <https://huggingface.co/Qwen/Qwen3-1.7B>
- Apple MLX-LM project:
  <https://github.com/ml-explore/mlx-lm>

## 4. Safety gates before installation

Installation or download is prohibited until all conditions are true:

1. The user separately approves the exact checkpoint and network download.
2. At least `20 GiB` is free on the target volume before download. The current
   approximately `14 GiB` fails this condition. No project or cache files are
   deleted automatically to create space.
3. The exact checkpoint revision, expected byte size, licence, and file list
   are recorded before transfer.
4. Model storage is an explicit directory, never an unresolved environment
   variable or the project root.
5. The runtime is installed in the project virtual environment with a pinned
   version and without changing the detector dependency range silently.
6. A one-prompt memory smoke test succeeds before any batch is attempted.
7. The process stops on memory pressure, thermal instability, low disk space,
   malformed metadata, or a checkpoint hash mismatch.

There is no API token fee, but local generation still consumes storage,
electricity, machine time, and researcher time. “Free” therefore means no paid
model service, not zero resource cost.

## 5. Stage-zero feasibility mini-pilot

Do not begin with the 288-document Gate C pilot. First run exactly one prompt
for every language × domain × length combination:

`2 languages × 3 domains × 3 length bands = 18 outputs`

The prompt packets must satisfy Gate C and be excluded from later formal tests.
Use one response per prompt, no best-of-N selection. The stage-zero run records:

- exact model and runtime revisions and hashes;
- full prompt packet and hash;
- native generation configuration and random seed where supported;
- start/end timestamps, prompt/output token counts, finish reason;
- elapsed time, tokens per second, and peak memory when exposed;
- language, character/word length, refusal and validation status;
- raw output hash and failure reason.

No detector is run during stage zero. Passing or failing is based only on
execution, language, length, encoding, and resource feasibility, so detector
performance cannot influence generator selection.

## 6. Stage-zero decision rule

Proceed to a larger exploratory pilot only if:

- all 18 prompts complete without process or memory failure;
- at least 17 outputs pass language and encoding validation;
- at least 16 land in their assigned length band without discretionary retry;
- no prompt or response is lost from the audit trail;
- measured runtime and storage permit the larger run under a written cap.

If the checkpoint fails, report the failure. Do not replace individual outputs
or switch quantization after viewing detector scores. Any new checkpoint or
quantization starts a new stage-zero run and receives a new generator ID.

## 7. Required implementation before execution

The next code phase, still runnable without model weights, should add:

- a schema for local generation receipts;
- validation of prompt/model/config hashes and required audit fields;
- a disk and runtime preflight command with explicit failure messages;
- an auditable stage-zero runner whose model-loading boundary can be tested
  with a fake generator;
- unit tests proving first-response retention, failure preservation, and that
  an output is never selected by detector score.

Model installation and actual generation remain separate, user-approved
actions after that code is reviewed.

