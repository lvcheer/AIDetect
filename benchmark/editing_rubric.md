# Editing Conditions and Audit Rubric

Rubric version: `0.1-draft`

Edits test robustness; they do not create additional independent samples. The
editor or paraphraser must never see detector outputs.

## `original`

- Preserve the exact eligible source or first valid generated response.
- Apply no spelling, whitespace, punctuation, or formatting corrections.
- Storage normalization, if required, must be byte-reproducible and recorded as
  acquisition processing rather than prose editing.

## `human_light_edit`

Permitted operations:

- spelling, punctuation, capitalization, and obvious grammar correction;
- remove repetition or repair a locally unclear sentence;
- replace a word or short phrase for idiomatic clarity;
- split or combine adjacent sentences without changing their claims.

Forbidden operations:

- add or remove substantive claims, examples, evidence, or citations;
- change stance, conclusion, audience, genre, or language;
- reorder paragraphs or rewrite the document globally;
- use a generative writing, grammar, translation, or paraphrasing tool.

Time is capped at 10 minutes per 700 Chinese characters or 500 English words,
prorated with a five-minute minimum. At least 90% normalized character
similarity to the parent is required using the frozen similarity implementation.
Falling below the threshold is a protocol failure, not a candidate for manual
promotion to a different condition.

## `machine_paraphrase`

- Use one model family that is separate from the three evaluated generator
  families where feasible; otherwise declare the resulting confounding.
- Pin its checkpoint or dated revision, system/user prompt, runtime, native
  parameters, and maximum output.
- Request meaning preservation, the same language, genre, and length band;
  forbid new facts, citations, translation, summarisation, and expansion.
- Make one request and retain the first valid output. No best-of-N selection.
- Record semantic-similarity and length flags for review, but do not select a
  text based on its detector score.

## `human_deep_edit`

This condition permits paragraph restructuring, substantial rewriting, and
addition or deletion of supporting material within a separate task brief. It
is mixed-authorship evidence and is excluded from binary primary metrics. The
editor records time, operations, and a short non-sensitive rationale.

## Required audit fields

- derivative and parent document IDs;
- shared source and near-duplicate component IDs;
- editor pseudonymous ID or paraphraser revision;
- start/end timestamps or elapsed duration;
- permitted-operation checklist and violation flag;
- original and edited hashes;
- character/word counts and normalized similarity;
- prompt/configuration hash for machine paraphrase;
- failure or exclusion reason.

Inter-editor consistency is checked on a small duplicated training set that is
not part of the pilot or formal corpus. Disagreement is used to clarify the
rubric before freeze, never to choose which formal edit is easier to detect.

