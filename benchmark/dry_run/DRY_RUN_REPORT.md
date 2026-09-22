# AIDetect bilingual dry-run report

Date: 2026-09-22

Status: pipeline validation only. This report does not contain or support model
accuracy, calibration, authorship, or comparative-performance claims.

## Purpose and outcome

The dry run tested whether the frozen manifest, provenance records, immutable
model revisions, explicit AI-label mappings, token-length inspection,
right-side truncation, per-record inference, failure handling, and artifact
hashing work together end to end.

Both detector runs completed successfully. Each produced one result for every
one of the eight frozen records, with no inference failures and with the
expected distinction between truncated and non-truncated inputs.

## Fixture coverage

The frozen manifest contains eight pipeline fixtures:

| Factor | Levels | Records |
|---|---|---:|
| Language | English, Chinese | 4 each |
| Reference provenance | human, AI | 4 each |
| Intended length condition | short, long | 4 each |
| Split | `dry_run` | 8 |
| Evaluation partition | `pipeline_dry_run` | 8 |

The records come from six source lineages. The two English human excerpts share
one Jane Austen lineage, and the two Chinese human excerpts share one Lu Xun
lineage. Each project-generated AI output has its own prompt lineage. Full
source, permission, prompt, and tokenizer details are recorded in
`PROVENANCE.md`.

## Frozen inputs

| Artifact | SHA-256 |
|---|---|
| Dataset schema | `575ea539b35e0ccf23893956efb4993cb29e78cd5dae25236f147beae1501a9a` |
| Candidate manifest | `24ae4dd0140ee8e97fbdcbc5cb7faae9fe6c08e80a99d7e73f895bf2a0b7377d` |
| Frozen manifest | `cd88bd9d547fba7c90ec7e5f62dc0cfdc4da422d530a1773f978bb7b84e84794` |
| Split metadata | `7790ddff21f6bb154fc104c2dea314897bd5c0df78cd643d41a8215b87a41397` |

The splitter recorded `manifest_mode = dry_run_only`, verified every text file,
and assigned all eight records to `dry_run/pipeline_dry_run`. No held-out
generator was required or represented for this pipeline-only fixture.

## Detector configurations

| Run | Model | Immutable revision | AI label | Code commit |
|---|---|---|---|---|
| `dry-run-tmr-20260922` | `Oxidane/tmr-ai-text-detector` | `0ceddea903015ef99cbaa040a4d8a216aed9c683` | index 1: `ai` | `517c1b7786033fbd5d6617e7e01a0aae843aee88` |
| `dry-run-chinese-roberta-20260922` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | `2f0f5f2af59af169a3f236c14ffad0fa7d6f072b` | index 1: `ChatGPT` | `08082f1dde8b9f235b157e22481947f596cb2113` |

Both runs used CPU inference, a maximum length of 512 tokens, right-side
truncation, `dry_run` as the only included split, and locally cached model files
in offline mode. Perplexity and score fusion were disabled.

## Execution coverage

| Run | Attempted | Successful | Failed | Truncated | Not truncated |
|---|---:|---:|---:|---:|---:|
| TMR detector | 8 | 8 | 0 | 6 | 2 |
| Chinese RoBERTa detector | 8 | 8 | 0 | 4 | 4 |
| Combined inference attempts | 16 | 16 | 0 | 10 | 6 |

Truncation differed because the two tokenizers encode the same text
differently:

| Run | English truncated | Chinese truncated |
|---|---:|---:|
| TMR detector | 2/4 | 4/4 |
| Chinese RoBERTa detector | 2/4 | 2/4 |

All truncated records had an effective length of exactly 512 tokens. All
non-truncated records retained their full inspected token length.

The median recorded total latency was approximately 112.7 ms per record for
the TMR run and 108.8 ms per record for the Chinese RoBERTa run. These values
are operational observations from one CPU run, not latency benchmarks. The
first inference in each run was substantially slower, consistent with a
warm-up effect that was not separately controlled.

## Result integrity

| Run | Results SHA-256 | Records by status |
|---|---|---|
| TMR detector | `4644124543815b8d6dedea39a96d96c7d64a319e7baafd13db9c75d0af9f7b72` | `success: 8` |
| Chinese RoBERTa detector | `ac3ad25fc71c39c16f57383ce33da16e08299ab71b9cad90208ac73bb4c5c80f` | `success: 8` |

Each result row records the full class-score vector, the explicitly selected
AI-class raw score, token lengths, truncation state, device, latency, source
revision, code commit, configuration hash, and manifest hash. Raw source text
is not copied into the result files. No score was rounded or converted into a
calibrated probability.

## What this dry run supports

The evidence supports the following narrow conclusions:

- the frozen manifest and its text hashes validate;
- both exact cached model revisions load without a network download;
- the configured label mappings are present and recorded explicitly;
- all eight records pass through both classifier pipelines;
- long-input truncation is detected and recorded rather than hidden;
- no classifier error was converted into a human prediction or zero score;
- result and configuration artifacts are cryptographically linked to their
  inputs.

## What this dry run does not support

The dry run cannot establish:

- accuracy, ROC-AUC, PR-AUC, false-positive rate, or any operating threshold;
- that raw softmax scores are calibrated probabilities;
- that one detector is better than the other;
- validity for Chinese, English, any writing domain, or any real population;
- robustness to unseen generators, editing, paraphrasing, or distribution
  shift;
- fairness, intent, misconduct, or true authorship of an arbitrary document.

## Important limitations

1. There are only eight deliberately constructed fixtures, two human works,
   and one task-session AI generator. The records are neither random nor
   representative.
2. The human and AI records differ in source, period, topic, and style. Those
   variables are confounded with the reference label.
3. Each detector was run on both languages. Cross-language completion confirms
   software behavior only; it does not demonstrate linguistic validity.
4. Long records are evaluated only through their first 512 tokens under the
   recorded right-truncation policy.
5. The AI fixture generator revision and sampling temperature were unavailable
   and therefore recorded as `null` rather than inferred.
6. Immutable Hugging Face revisions were recorded, but `model_artifact_sha256`
   is `null` because the models were loaded by repository ID from the local
   cache rather than from a project-local model directory.
7. Latency comes from one CPU execution without warm-up repetitions, randomized
   order, or hardware replication.
8. Configuration labels and controlled-provenance fixtures provide a sanity
   check, not independent evidence that a detector score measures AI
   authorship or is calibrated.

## Gate before a formal benchmark

A formal run remains blocked until the protocol version, lawful data sources,
sample counts, deduplication procedure, generator-held-out design, split
fractions, primary operating point, calibration method, uncertainty procedure,
and analysis plan are reviewed and frozen. Dry-run scores must not be used to
tune those choices.
