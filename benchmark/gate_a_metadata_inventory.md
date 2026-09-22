# Gate A — Metadata-Only Feasibility Inventory

Inventory version: `0.1-draft`

Inventory date: `2026-09-22`

Scope: public collection-level metadata and counts only. No article full text,
PDF, supplementary file, or benchmark sample was downloaded or extracted.

## Question

Is the openly licensed, pre-generative-AI Chinese academic candidate pool large
enough to justify a pilot without using CC BY-NC text?

## Candidate-pool observations

### ChinaXiv

The public ChinaXiv browse interface displayed `46,388` records in total and
the following submission-year facets:

| Submission year | Public metadata count |
|---:|---:|
| 2017 | 5,995 |
| 2018 | 3,614 |
| 2019 | 1,289 |
| 2020 | 1,076 |
| 2021 | 1,205 |
| **2017–2021 subtotal** | **13,179** |

The subtotal is deliberately limited to complete years before 2022. It is an
unfiltered upper candidate pool, not an eligible benchmark count. The browse
facets do not establish:

- whether a record's full text is Chinese;
- whether the selected item/version licence is CC BY or CC0 rather than
  CC BY-SA, CC BY-NC-SA, or another licence;
- whether the deposited file is an author manuscript rather than a publisher
  PDF;
- whether a later version crossed the date boundary;
- whether the work is a preprint, accepted manuscript, or published article;
- whether multiple versions or related records share one source lineage.

The official platform documentation confirms that language is entered as
structured metadata and that the interface supports date, field, author, and
institution browsing. Its licence policy confirms that the licence is selected
per submission. Therefore, an exact eligible count requires item/version-level
metadata filtering; it cannot be inferred from the repository total.

Authoritative sources:

- ChinaXiv public browse interface: <https://www.chinaxiv.org/user/search.htm>.
- ChinaXiv platform help and metadata/search description:
  <https://www.chinaxiv.org/user/help.htm>.
- ChinaXiv item-level licence options:
  <https://www.chinaxiv.org/user/license.htm>.

### *Language and Linguistics / 語言暨語言學*

The institutional TOAJ record states that, beginning in 2017, the journal
publishes one Chinese issue per year with an average of six or seven articles
per issue. It also states that all articles use CC BY 4.0 and that the journal
does not publish translations.

For 2017–2021 this implies an expected collection-level supply of approximately
`30–35` Chinese articles. This is an expectation from the official publication
schedule, not an item count; article type, language, licence, and exact version
still require item-level confirmation.

Source:
<https://toaj.stpi.niar.org.tw/index/journal/4b1141f97ce46933017ce469b63f0099>.

## Feasibility decision

| Decision level | Outcome | Reason |
|---|---|---|
| Pilot (`10–20` lineages per planned cell) | **Feasible in principle** | One established peer-reviewed CC BY stream plus a large versioned repository candidate pool exists without relying on CC BY-NC material |
| Formal reporting floor | **Not yet demonstrated** | The 13,179 repository records have not been filtered jointly by language, item licence, version date, document type, and lineage |
| Broad academic-domain representation | **Not yet demonstrated** | Linguistics alone is field-specific; ChinaXiv subject balance among eligible Chinese items is unknown |
| Pure-human reference label | **Conditional** | Pre-`2022-11-30` first versions reduce, but do not eliminate, undisclosed machine-assistance risk |

Gate A therefore supports designing a metadata-screening pilot, but it does not
authorise full-text collection or prove that the final sample-size floor can be
met.

## Required screening before any text acquisition

Create a metadata-only candidate table with one row per immutable version and
the following minimum fields:

`candidate_id`, `source`, `record_uri`, `title`, `language`, `subject`,
`document_type`, `first_deposit_date`, `version_date`, `version_id`,
`is_first_version`, `licence`, `licence_evidence_uri`, `peer_review_status`,
`publisher_pdf_flag`, `related_record_id`, and `screening_decision`.

Apply these rules in order:

1. retain Chinese full-text metadata only;
2. retain first versions dated no later than `2022-11-30`, with no later
   version substituted;
3. retain only explicit item-level `CC BY 4.0` or `CC0 1.0`;
4. exclude publisher PDFs unless separate reuse rights are explicit;
5. separate peer-reviewed articles from preprints;
6. deduplicate DOIs, titles, author/title combinations, and related versions;
7. tabulate counts by subject before selecting any records;
8. stop if the available interface does not permit compliant metadata access;
   do not replace it with unrestricted scraping.

The screening output remains candidate metadata. Passing it does not admit the
associated text; every selected item still needs the source eligibility review
template.

## Metadata access audit

The official help pages document interactive browsing, faceted search, and
record detail pages. The reviewed help, licence, and legal-statement pages do
not document a public bulk metadata API, OAI-PMH endpoint, or authorised bulk
export. The legal statement requires lawful and reasonable platform use but
does not itself grant permission for automated harvesting.

Consequently, the project screening tool is deliberately source-agnostic: it
validates and screens metadata supplied through an authorised route, but it
does not scrape ChinaXiv. Before an automated inventory is attempted, the
project must obtain either official interface documentation or written
permission from the platform contact listed in its help pages. Manual browsing
may support small feasibility checks but is not a reproducible formal
acquisition method.

Sources:

- <https://www.chinaxiv.org/user/help.htm>
- <https://www.chinaxiv.org/user/license.htm>
- <https://www.chinaxiv.org/user/law.htm>

## Local screening implementation

The repository provides:

- `candidate_metadata_schema.json` for metadata-only records;
- `python -m aidetect.candidate_metadata` (or the installed
  `aidetect-screen-metadata` command) to validate, screen, preserve all
  decisions, and produce aggregate rejection/eligibility counts.

The tool has no network or text-acquisition code. Its current deterministic
eligibility rules implement the language, cutoff-date, licence, and publisher-
PDF gates above. Passing the automated screen remains necessary but not
sufficient for source admission.
