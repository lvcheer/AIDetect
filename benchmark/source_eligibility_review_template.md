# Source Eligibility Review Template

Template version: `0.1-draft`

Complete one review for every source collection and attach an item-level record
for every admitted document. A collection-level approval never overrides a
more restrictive document-level notice.

## A. Source identity

| Field | Required entry |
|---|---|
| Review ID | Stable project identifier |
| Source/collection title | Official title |
| Publisher or custodian | Legal or institutional entity |
| Canonical URI | Official landing page, not a search-result URL |
| Intended benchmark role | Language, domain, reference class, and partition |
| Reviewer and review date | Name/identifier and ISO date |
| Archived evidence | URI plus hash or snapshot reference, when permitted |

## B. Provenance and reference label

| Question | Evidence/answer |
|---|---|
| Who created the text? | Author, institution, generator, or documented category |
| How is human/AI provenance known? | Do not infer from style or detector output |
| Is AI assistance excluded or documented? | State the method and residual uncertainty |
| Is the exact version identifiable? | Revision, edition, DOI, publication date, or dated model revision |
| Are translations, edits, or extracts present? | Identify their creator and parent work |
| Can the complete lineage share one `source_id`? | Explain the leakage component |

## C. Rights and access review

| Question | Evidence/answer |
|---|---|
| Rights statement URI | Exact licence or permission page |
| Document-level licence | SPDX/CC identifier or verbatim permission category |
| Copyright holder | As stated by the source |
| Permitted acquisition method | API, bulk archive, manual retrieval, or supplied copy |
| Research processing permitted? | Include extraction, normalisation, tokenisation, hashing, and scoring |
| Modification permitted? | Required for excerpts, cleaning, or edited variants |
| Redistribution permitted? | Raw text, excerpts, metadata only, hashes only, or prohibited |
| Commercial restriction? | `yes`, `no`, or `unclear` |
| Attribution/share-alike duties | Exact text and downstream obligations |
| Third-party material excluded? | Images, quotations, tables, appendices, or syndicated text |
| Geographic/jurisdiction caveat? | Record any relevant limitation |
| Privacy or sensitive-data risk? | Personal, medical, educational, or confidential content |
| Terms stable and saved? | Retrieval date and evidence location |

## D. Scientific eligibility

| Question | Evidence/answer |
|---|---|
| Domain definition satisfied? | Apply the frozen operational definition |
| Language and authorship appropriate? | Distinguish original writing from translation |
| Detector-training overlap known? | Search model cards, papers, and named corpora |
| Generator-pretraining contamination risk | `low`, `medium`, `high`, or `unknown`, with reason |
| Topic/style confounding controlled? | State the proposed matching or stratification |
| Duplicate controls possible? | Exact hash, normalised hash, and near-duplicate method |
| Length targets feasible? | Based on pilot measurements, not manual impression |

## E. Decision

Choose exactly one status:

- `approved_for_pilot`: all planned uses are supported by written evidence;
- `conditional`: usable only under named restrictions or after item-level checks;
- `permission_required`: written permission is needed before acquisition;
- `metadata_only`: raw text cannot enter the corpus;
- `excluded`: provenance, rights, privacy, or scientific validity is inadequate.

Record:

| Field | Required entry |
|---|---|
| Decision | One status above |
| Permitted benchmark role | Exact language/domain/partition and release mode |
| Conditions | Every restriction that must be enforced |
| Unresolved questions | None may be silently interpreted as permission |
| Re-review trigger/date | Licence change, source update, release-policy change, or fixed date |
| Approver | Project decision owner |

## F. Item-level acquisition ledger fields

Every admitted item must record at least:

`document_id`, `source_id`, source title, creator, canonical URI, source
revision, publication date, retrieval date, rights statement URI, exact
licence, required attribution, redistribution category, language, original or
translated language status, domain, text hash, transformation history,
exclusion flags, and reviewer.

This template is a research governance check, not legal advice. Ambiguous
rights require permission or exclusion.

