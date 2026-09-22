# Gate C Pilot Design Protocol

Protocol date: `2026-09-22`

Status: **design specification only; no collection or generation authorised**

This protocol defines the pilot cells, prompt families, length bands, editing
conditions, quality controls, and decision rules needed before formal sample
counts can be frozen. It does not authorise API spending or data acquisition.

## 1. Confirmatory scope

The matched core contains two languages (`zh`, `en`), three domains
(`general`, `academic`, `professional`), three provisional generator families,
and three length bands. The primary comparison uses unedited human and unedited
AI text only. Editing conditions are prespecified secondary robustness tests.

Application writing is not part of the matched core. Real applications create
privacy and consent risks, while synthetic applications do not form a clean
human reference class. It may be added only as a separately approved
exploratory corpus.

### Domain definitions

| Domain | Included | Excluded | AI task family |
|---|---|---|---|
| `general` | self-contained explanatory or narrative prose for a non-specialist reader | news copying, lists, dialogue, poetry, personal correspondence, encyclopaedia passages copied into prompts | explain a neutral topic or narrate a non-personal scenario |
| `academic` | abstract-like or short scholarly prose stating a question, approach, findings or expected contribution, and limitations | equations-only text, references section, copied abstracts, fabricated citations, discipline-specific formats that cannot be matched across languages | write a citation-free abstract-like passage from an independently written research brief |
| `professional` | public-information, administrative, policy, or procedural prose for a defined audience | marketing copy, legal advice, medical advice, private workplace records, application materials | write a guidance note, briefing, or procedural explanation from an independently written scenario |

Domain assignment is based on communicative purpose, not source website. A
second reviewer resolves ambiguous assignments without seeing detector scores.
Unresolved material is excluded with a reason code.

## 2. Prompt-block design

A **topic packet** is independently written benchmark material containing only
neutral facts or a fictional scenario, the intended audience, genre, and target
length. It must not quote, translate, summarize, or closely paraphrase a human
benchmark document. It cannot mention AI detection, evasion, authorship, or ask
the model to “sound human.”

Each approved topic packet creates one `prompt_family_id`. Its Chinese and
English versions are authored for their own linguistic context; they are not
sentence-level translations. A factual-equivalence reviewer may verify matched
difficulty, but the two languages receive distinct source IDs.

Every generator receives the same frozen language-specific packet and output
contract. All outputs derived from one topic packet—including outputs from
different generators, retries, edits, and paraphrases—share a leakage component
and remain in one split. Only the first valid response to the first successful
request is used; invalid requests may be retried under a documented mechanical
retry rule, never because an output looks difficult for a detector.

The system and user templates are defined in `prompt_packet_template.md`.
Tools, browsing, grounding, retrieval, citations, conversational history, and
assistant prefilling are disabled. Provider-native default sampling is used as
specified by Gate B, and all native settings are recorded.

## 3. Length bands

Length is measured on normalized visible prose before detector-specific
tokenization. Chinese uses CJK character count excluding whitespace; English
uses Unicode word count. The provisional bands are:

| Band | Chinese | English | Pilot allocation per language × domain |
|---|---:|---:|---:|
| `short` | 250–399 characters | 150–239 words | 4 topic packets and 4 human lineages |
| `medium` | 400–699 characters | 240–419 words | 4 topic packets and 4 human lineages |
| `long` | 700–1,100 characters | 420–700 words | 4 topic packets and 4 human lineages |

These are feasibility bands, not claims that a Chinese character equals an
English word or a model token. The pilot records characters, words, and every
detector tokenizer's effective token count. Before formal collection, band
boundaries may be adjusted using only coverage, truncation, and distributional
fit—not detector accuracy—and the reason must be committed. After Gate D they
cannot be changed without a new benchmark version.

A response outside its assigned band is retained in the generation audit. One
mechanical retry may restate the unchanged target band if and only if this rule
was configured before the request. A second miss is a recorded failure and is
not replaced by manual selection.

## 4. Pilot size and factorial allocation

The complete pilot contains:

- `12` independent human lineages per language × domain, with four in each
  length band: `2 × 3 × 12 = 72` human originals;
- `12` independent AI topic-packet components per language × domain, with four
  in each band and one output from each of three generator families:
  `2 × 3 × 12 × 3 = 216` AI originals;
- `144` independent components and `288` original documents in total.

The three outputs sharing a topic packet are paired observations, not three
independent lineages. Derivatives produced under editing conditions also do not
increase the independent count.

For editing feasibility, deterministically select two topic-packet components
per language × domain—one medium and one long component—using a seed fixed
before text is viewed. Apply every eligible editing condition to the selected
development-generator outputs. Held-out-generator material receives no
developmental editing or qualitative inspection.

The pilot is excluded from all final test partitions. Human and development-
generator pilot samples may inform operational rules. The held-out generator
is run only after prompts and rules are frozen; its pilot is limited to API,
language, encoding, refusal, and length validation. It may not be used to tune
prompts, thresholds, detectors, or editing rules. A failure that would require
such tuning excludes or replaces the entire held-out family before formal
generation.

## 5. Editing conditions

Detailed operations and audit requirements are in `editing_rubric.md`.

| Condition | Analysis status | Core rule |
|---|---|---|
| `original` | confirmatory primary | exact eligible human source or first valid AI response; no prose edits |
| `human_light_edit` | prespecified secondary | local spelling, punctuation, grammar, and clarity only; no new claims or paragraph restructuring |
| `machine_paraphrase` | prespecified secondary | one separately pinned paraphraser and one frozen prompt; no candidate selection |
| `human_deep_edit` | exploratory only | may restructure or rewrite; treated as mixed authorship and excluded from binary primary metrics |

All derivatives preserve `parent_document_id`, `source_id`, prompt component,
and split. The original is never overwritten. Diff statistics, editor time,
operations, paraphraser revision, failures, and hashes are retained.

## 6. Blinding and quality control

1. Source eligibility reviewers do not see detector scores.
2. Domain and language adjudicators do not see generator identity where
   blinding is feasible and never see detector results.
3. Editors receive the editing instruction and text only; they do not receive
   detector scores or a target detector outcome.
4. Detector developers may inspect only training, calibration, dry-run, and
   permitted development-pilot material.
5. Held-out content, labels, and detector outputs remain access-controlled
   until the analysis configuration is signed and hashed.

Automated checks cover encoding, empty text, assigned language, exact length,
duplicate hashes, near duplication, template leakage, unexpected citations,
and forbidden prompt phrases. Automated language identification is a flag, not
the sole exclusion authority. All exclusions use frozen reason codes.

## 7. Gate D inputs and pass rules

Gate C is a design gate; Gate D decides feasibility. The pilot report must show
every language × domain × generator × length cell separately, and also pool the
four-document length cells into each 12-component language × domain × generator
cell, reporting:

- attempted, valid-first-response, retry, refusal, blocked, empty, wrong-
  language, and out-of-band counts;
- actual character, word, and detector-token distributions and truncation;
- duplicate-component structure and editing retention statistics;
- input/output tokens, latency, and realised cost;
- source-eligibility and acquisition failure rates for human material.

A pooled 12-component language × domain × generator cell passes operationally
only if at least `10` components are eligible without discretionary replacement,
its total generation failure rate is at most `10%`, every length band has at
least three valid components, and no unresolved systematic language or
truncation problem remains. These are feasibility thresholds, not evidence of
detector accuracy. Failed cells trigger a documented redesign before any formal
test is assembled.

Gate D then uses the source-cluster bootstrap and simulation already required
by `formal_data_source_plan.md` to choose formal counts. The reporting floors
remain floors; the pilot does not automatically justify them as statistically
adequate.

## 8. Frozen deliverables before generation

- approved source and generator reviews from Gates A and B;
- versioned topic-packet register and bilingual templates;
- deterministic packet, editing-subset, and split seeds;
- approved length bands and reason codes;
- editor instructions and adjudication form;
- generation configuration, model revisions, budget and stop limit;
- access-control plan for held-out data;
- hashes of every material document and the Git commit.

Changing a domain, prompt family, length band, editing rule, pilot allocation,
or held-out access rule after viewing relevant detector results creates a new
benchmark version.
