# AIDetect Benchmark Metrics

Metrics specification version: `0.1-draft`

## Conventions

- The positive class is AI-generated text (`human_or_ai = "ai"`).
- All detector outputs are transformed, if necessary, so that larger raw scores
  mean more AI-like text. The transformation and source label mapping are saved
  per detector.
- Raw scores are not interpreted as probabilities.
- Calibrated probabilities are reported only after a named calibration method
  has been fitted without final-test data.
- Metrics use documents as observations, while confidence intervals resample
  source lineages to preserve dependence among edited versions.

## Analysis populations

Every result table states its population and denominator:

1. **All attempted records:** used for coverage and failure reporting.
2. **Successfully scored records:** used for discrimination metrics, accompanied
   by the excluded failure count and rate.
3. **Calibration-eligible records:** calibration data not used to fit the model
   being evaluated in that calculation.
4. **Automatically decided records:** used for selective-prediction risk, always
   accompanied by coverage.

The in-distribution test, generator-held-out test, and any domain- or time-held-
out test are reported separately before any pooled summary.

## Primary discrimination metrics

### ROC-AUC

Probability that a randomly selected AI record receives a higher raw score than
a randomly selected human record. Report with a 95% bootstrap confidence
interval. ROC-AUC does not by itself establish acceptable false-positive risk.

### PR-AUC

Area under the precision-recall curve with AI as the positive class. Report the
AI prevalence alongside PR-AUC because the value depends on class prevalence.

### Human false-positive rate at fixed AI recall

Report the human false-positive rate at one or more AI-recall targets frozen
before the final test is evaluated. Candidate targets are 90% and 95%; the
primary target will be selected during protocol freeze based on the intended-use
cost model, not on final-test performance.

Thresholds for fixed-recall results are selected using permitted development or
calibration data and then applied unchanged to final-test partitions.

## Secondary thresholded metrics

At each prespecified operating point, report:

- balanced accuracy;
- sensitivity/recall for AI text;
- specificity for human text;
- precision/positive predictive value;
- F1 score;
- confusion-matrix counts;
- human false-positive rate and AI false-negative rate.

Accuracy alone is not a primary metric. Every thresholded metric must identify
where the threshold was selected.

## Calibration metrics

Calibration metrics apply only to calibrated probabilities.

### Brier score

Mean squared difference between calibrated AI probability and the binary
reference label. Lower is better. Report the uncalibrated score only if the raw
output is naturally bounded in `[0, 1]`, and label it clearly as an uncalibrated
comparison.

### Log loss

Binary cross-entropy of calibrated probabilities. Probabilities are clipped
only for numerical stability using a documented epsilon shared by all methods.

### Expected calibration error

Report ECE with the binning rule, number of bins, and sample counts shown. The
default candidate is 15 equal-frequency bins, to be frozen before evaluation.
Because ECE depends on binning and can hide local error, it is always accompanied
by a reliability diagram and Brier score.

### Reliability diagram

Plot mean predicted probability against observed AI frequency with bin counts
and bootstrap uncertainty where sample size permits. Provide overall, Chinese,
and English views; small groups are labelled as imprecise rather than treated as
evidence of no calibration error.

Calibration methods are compared using data not used to fit the candidate
calibrator. The final test is not used to choose Platt, temperature, isotonic, or
another method.

## Selective prediction and abstention

The final interface has three possible states:

- `human-like`;
- `uncertain — human review required`;
- `AI-like`.

For each prespecified pair of lower and upper thresholds, report:

- **coverage:** proportion receiving an automatic human-like or AI-like decision;
- **abstention rate:** proportion assigned to human review;
- **selective risk:** error rate among automatically decided records;
- human false-positive rate among covered human records;
- AI false-negative rate among covered AI records;
- counts for all denominators.

Plot coverage against selective risk. Select at least two operating scenarios
before final evaluation: one prioritising low human false positives and one
prioritising low AI false negatives. The cost assumptions must be written next
to the chosen thresholds.

## Subgroup and robustness reporting

Prespecified descriptive groups include:

- language (`zh`, `en`);
- domain;
- text-length band, defined before the full run;
- generator;
- editing condition;
- truncation status;
- generator-held-out and other shift partitions;
- author group only when lawfully and reliably recorded.

For each sufficiently populated group, report sample counts, class prevalence,
primary metrics, failure rate, and 95% confidence intervals. Report absolute
differences from the relevant reference or overall group when useful. Do not
describe observational differences as causal discrimination.

Groups below the frozen minimum effective sample size are shown as
`insufficient evidence`; their point estimates are not used for strong claims.
Exploratory groups are labelled as such, and multiple subgroup searches are not
presented as confirmatory findings.

## Uncertainty

Use a source-cluster bootstrap so that every record sharing a `source_id` is
resampled together. Use the same resamples when comparing detectors to preserve
pairing. The candidate default is 2,000 replicates with a fixed random seed;
the replicate count and interval method will be frozen before the full run.

Report 95% intervals for primary metrics and key differences. A wide interval
is interpreted as imprecision, not as evidence that systems are equivalent.
Failed or undefined bootstrap replicates are counted and disclosed.

## Detector comparisons and fusion

- Compare detectors on identical records using paired estimates.
- Report effect differences with confidence intervals, not rankings alone.
- Perplexity and burstiness are separate baselines before any fusion.
- Manually chosen fusion weights are not treated as validated probabilities.
- Any learned fusion is fitted only on training/calibration data and compared
  against the best single model on unchanged held-out partitions.
- Retain the best single model if fusion does not show stable benefit under
  generator or domain shift.

## Operational metrics

For each detector and partition, report:

- attempted, successful, and failed record counts;
- failure rate and failure reasons;
- median and tail latency per document;
- hardware/device and batch size;
- peak memory when measurement is available;
- input token counts and truncation rate.

Errors are not converted into human predictions or zero scores. Performance and
coverage must be read together.

## Minimum reporting table

Every formal result bundle must contain:

1. manifest composition by split, class, language, domain, generator, and
   editing condition;
2. coverage and failure table;
3. overall discrimination table;
4. held-out generator and distribution-shift table;
5. prespecified subgroup table;
6. calibration table and reliability diagrams;
7. coverage-risk results for abstention;
8. paired detector or fusion comparisons;
9. qualitative false-positive and false-negative cases with sensitive text
   removed or redacted as required.

All aggregate tables and figures must be generated from versioned per-sample
results. Manually transcribed headline numbers are not authoritative.
