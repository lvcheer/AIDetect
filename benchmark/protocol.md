# AIDetect Benchmark Protocol

Protocol version: `0.1-draft`
Baseline code commit: `25521b6a35d77845fff7f7fb1de09004abd2dcde`

## Research question

How reliably can existing AI-text detectors distinguish human-authored and
AI-generated text across languages, domains, generators, and editing
conditions, and when should the system abstain and require human review?

This benchmark estimates detector performance under specified conditions. It
does not prove authorship, intent, misconduct, or the absence of AI assistance.

## Objectives

1. Compare individual detectors and non-neural baselines on the same frozen
   evaluation data.
2. Measure performance shifts across Chinese and English, writing domains,
   generators, text lengths, and editing conditions.
3. Evaluate generalisation to at least one held-out generator and, if feasible,
   a held-out domain or later collection period.
4. Separate uncalibrated model scores from calibrated probabilities.
5. Quantify uncertainty and identify settings where automatic decisions should
   be withheld.

## Scope

### Included

- Chinese (`zh`) and English (`en`) text.
- Human-authored and AI-generated source lineages with documented provenance.
- Academic, general, application, or professional writing, subject to data
  availability and licensing.
- Original AI output, human-edited AI output, and machine-paraphrased AI output.
- Binary AI-versus-human evaluation, calibration, selective prediction, and
  descriptive subgroup analysis.

### Excluded from primary claims

- Mixed-authorship texts whose reference label cannot be defined consistently.
- Text with unknown provenance or an incompatible licence.
- Authorship attribution to a particular person or model.
- Plagiarism, factuality, writing quality, or misconduct assessment.
- Causal claims about demographic fairness or discrimination.
- Performance claims based only on the small dry run.

Ambiguous and mixed-authorship examples may be retained in a separate
exploratory set, but they must not be silently assigned a binary ground-truth
label or included in primary metrics.

## Terminology and output semantics

- **Raw score:** an uncalibrated detector output. A larger value must mean
  "more AI-like" after model-specific label mapping is verified.
- **Calibrated probability:** an output produced by a calibration method fitted
  only on the calibration split. Raw softmax values and heuristic transforms
  are not called probabilities.
- **Decision state:** `AI-like`, `uncertain`, or `human-like`, produced from a
  prespecified calibrated decision policy.
- **Reference label:** the recorded provenance class used for evaluation, not a
  claim that the provenance can always be known in real-world use.

## Evaluation factors

The frozen manifest will identify every document by:

- language;
- domain;
- human or AI provenance;
- generator and generator revision, when applicable;
- prompt and generation settings, when available;
- editing condition;
- source lineage and near-duplicate group;
- text length;
- licence and provenance record;
- split and distribution-shift partition.

The final model list, data sources, sample counts, split proportions, and target
recall operating points remain open until the code and lawful data inventory is
complete. They must be frozen before the full benchmark begins.

## Dataset eligibility

### Inclusion criteria

- Text has a documented source and an operationally defensible reference label.
- Collection and redistribution or derived-result reporting are permitted.
- Language and domain can be assigned using a documented procedure.
- The full source lineage can be placed in one leakage group.
- Text passes encoding, minimum-length, and duplicate checks defined before the
  full benchmark.

### Exclusion criteria

- Unknown or conflicting provenance.
- Missing licence or use permission.
- Exact duplicate of another record.
- Content created by a process that crosses the binary label boundary without a
  defensible final label.
- Processing failure that prevents the original text from being recovered.

Exclusions and detector failures will be recorded with reason codes; they will
not be silently dropped.

## Manifest and lineage rules

Every row must validate against `dataset_manifest_schema.json`.

- `document_id` identifies one observed text.
- `source_id` is the leakage-group identifier. An original document, all of its
  edits, translations, paraphrases, continuations, and extracted segments must
  share one `source_id`.
- Exact and near-duplicate checks occur before splitting.
- `near_duplicate_cluster_id`, when assigned, cannot cross splits.
- AI records identify the generator and, when available, the prompt; unavailable
  prompt or generation settings are represented explicitly as `null`, not
  guessed.
- Human records must not be inferred to be human merely because a detector gives
  a low score.

## Split policy

1. Build and deduplicate the eligible manifest before assigning splits.
2. Split by `source_id`, never by individual row.
3. Keep train, calibration, and final test data disjoint.
4. Keep every version of a source document in the same split.
5. Reserve at least one generator exclusively for `generator_held_out` testing.
6. If feasible, reserve a domain or later collection period for an additional
   shift evaluation.
7. Do not use final test labels, metrics, or examples to select models,
   calibration methods, fusion weights, thresholds, or preprocessing rules.
8. Treat the dry-run partition as pipeline validation only; it is not part of
   the frozen test result.

A deterministic split script will record its random seed and input manifest
hash. Any post-freeze correction requires a new manifest version and a written
change note.

## Benchmark procedure

### Before the full run

1. Freeze the protocol, metric definitions, manifest, and analysis plan.
2. Record the Git commit, environment, dependency versions, model identifiers,
   immutable model revisions, tokenizer settings, and device.
3. Verify each model's label mapping and raw-score direction with controlled
   examples; do not infer label semantics only from class position.
4. Run a small end-to-end dry run covering both languages, long inputs, invalid
   inputs, and each error path.

### Per-sample execution

Record at minimum:

- document and source identifiers;
- detector and immutable model revision;
- raw score and confirmed score direction;
- input length, effective token length, maximum length, and truncation status;
- latency, device, and failure status;
- perplexity and burstiness as separate features when enabled;
- run ID, configuration hash, code commit, and manifest hash.

Raw detector scores, perplexity-derived scores, and burstiness-derived scores
remain separate in the baseline. No manually weighted composite is a primary
baseline unless it was prespecified independently of test results.

### Analysis sequence

1. Report coverage and failure rates.
2. Report individual detector and feature-baseline discrimination metrics.
3. Report prespecified language, domain, length, generator, and editing groups.
4. Report generator-held-out and other shift partitions separately.
5. Fit calibration and any learned fusion only with permitted training and
   calibration data.
6. Evaluate calibrated and selective-decision outputs once on the frozen test
   set.
7. Inspect a small, prespecified number of high-confidence false positives and
   false negatives as qualitative failure cases.

Metric definitions and uncertainty procedures are specified in `metrics.md`.

## Threats to validity and mitigations

| Threat | Required mitigation |
|---|---|
| Exact or near-duplicate leakage | Deduplicate before splitting; group by source and duplicate cluster. |
| Prompt or source leakage | Keep prompt families and source lineages within one split; report residual risk. |
| Generator memorisation | Use a generator-held-out test partition and immutable model revisions. |
| Domain, topic, or length confounding | Balance or stratify where feasible and report subgroup results. |
| Human/AI label ambiguity | Use provenance-based definitions; isolate ambiguous and mixed cases. |
| Incorrect label mapping | Verify model metadata and controlled examples; record mapping per model. |
| Truncation artefacts | Record effective length and truncation; analyse truncated inputs separately. |
| Post-test tuning | Freeze rules before test access; log protocol changes and rerun under a new version. |
| Selective failure handling | Report failures and denominator changes; never replace failures with zero scores. |
| Multiple subgroup comparisons | Mark subgroup analyses as prespecified or exploratory and report uncertainty. |
| Time and model drift | Record collection dates, model revisions, and benchmark version. |
| Licensing or privacy violations | Include only documented, permitted data and avoid sensitive text release. |

## Reproducibility requirements

Each formal run must produce:

- the validated manifest or a permitted metadata-only version;
- a configuration file and cryptographic hashes of material inputs;
- immutable model identifiers and revisions;
- per-sample result files;
- automatically generated tables and figures;
- environment and dependency information;
- an analysis log distinguishing prespecified from exploratory work.

## Change control

This draft becomes frozen only after the data and model inventory is reviewed.
After freezing, changes that affect eligibility, splits, preprocessing, primary
metrics, calibration, or thresholds require a new protocol version and an
explanation. Test results from an earlier version must not be used to optimise a
later version without being relabelled as development evidence.
