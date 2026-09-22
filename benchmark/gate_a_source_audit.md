# Gate A — Initial Lawful-Source Audit

Audit version: `0.1-draft`

Audit date: `2026-09-22`

Scope: official-source and licence review only. No corpus text was downloaded,
extracted, or admitted by this audit.

## Decision vocabulary

- **Conditional pilot approval:** the source type may enter a pilot only after
  every document passes the item-level template.
- **Pending:** evidence is promising but insufficient for acquisition.
- **Excluded:** the source may not be collected under the current plan.

No source in this document is approved for the frozen final test.

## Audit results

| Candidate | Intended cell | Evidence | Decision | Conditions and scientific caveats |
|---|---|---|---|---|
| UK government official Chinese guidance explicitly carrying OGL v3.0 | `zh/professional/human` | The official 2025 Chinese FIRS guidance states Crown copyright, OGL v3.0 reuse, and a third-party-rights caveat inside the document | **Conditional pilot approval** | Verify the same notice in every document; exclude third-party material; retain Crown attribution; record that the prose is an official translation; group Chinese and English counterparts under one lineage |
| Matching English version of the same UK guidance | `en/professional/human` | OGL v3.0 permits copying, distribution, adaptation, and commercial/non-commercial reuse with attribution | **Conditional pilot approval** | Use only content expressly covered by OGL; pair with translated versions without crossing splits; do not count translations as independent author lineages |
| HKBU *人文中國學報* | `zh/academic/human` | The publisher says all content is CC BY-NC 4.0 and provides Chinese peer-reviewed articles | **Conditional, not yet approved for acquisition** | The benchmark corpus and redistributed text would have to remain non-commercial with attribution; confirm the licence in each article; exclude third-party quotations/figures; release-policy decision is required first |
| HKBU *中外醫學哲學* | `zh/academic/human` | The publisher identifies it as a Chinese-language academic journal and licences content CC BY-NC 4.0 | **Conditional, not yet approved for acquisition** | Same non-commercial and item-level conditions; medical/bioethical subject matter may create domain and sensitivity effects, so it cannot be the sole Chinese academic source |
| *中國語文通訊* | `zh/academic/human` | DOAJ reports Chinese/English manuscripts under CC BY-NC-ND | **Pending** | No-derivatives terms may conflict with excerpting, cleaning, and edited conditions; obtain publisher/item-level confirmation before any use |
| *Writing Chinese* | possible academic source | Publisher/DOAJ indicate open access and CC BY, but the journal primarily publishes English research and English translations of Chinese scholarship | **Excluded from the Chinese core** | It may supply English academic material after item-level review, but subject matter about Chinese writing does not make the evaluated text Chinese |
| Mainland Chinese government and ministry web pages as a general collection | `zh/professional/human` | Official sites show inconsistent terms: the Ministry of Justice prohibits reuse of its copyrighted works without written permission, while other ministries impose source-specific rules | **Excluded as a blanket source** | “Government website” is not a licence. A page may be reconsidered only with explicit reuse permission or a narrow statutory exclusion reviewed separately |
| Mainland statutes and official legal instruments | possible `zh/professional` material | The Copyright Law and official sources distinguish legal instruments from ordinary authored web content | **Pending/exploratory only** | Legal instruments have atypical formulaic style and may confound detection; obtain a scoped legal review and keep them separate from ordinary professional prose |
| Chinese Wikisource works with verified work-level public-domain/free status | `zh/general/human` | Site contributions use CC BY-SA 4.0/GFDL and work pages carry source-specific rights information | **Conditional pilot approval** | Verify each work, edition, author death date/local status, transcription contribution terms, and attribution/share-alike duties; report historical text separately |

## Evidence register

### Professional Chinese

1. UK Home Office, *Guidance on the Foreign Influence Registration Scheme:
   enhanced tier (Chinese)*, 2025. The official PDF contains its OGL v3.0 and
   third-party-rights notice:
   <https://assets.publishing.service.gov.uk/media/685e8c04f85b4b993fd7536f/Guidance%2Bon%2Bthe%2Benhanced%2Btier_Chinese.pdf>.
2. The National Archives, Open Government Licence v3.0. It grants a worldwide,
   royalty-free, perpetual, non-exclusive right to copy, publish, distribute,
   adapt, and exploit covered information, subject to attribution and stated
   exclusions:
   <https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/>.
3. Ministry of Justice of the People's Republic of China, legal notice. It says
   works copyrighted by the site may not be reproduced without written
   authorisation:
   <https://www.moj.gov.cn/pub/sfbgw/flsm/201812/t20181231_164581.html>.
4. National Copyright Administration, notice on online republication. It states
   that online media generally require permission, remuneration, and
   attribution unless an exception applies:
   <https://www.ncac.gov.cn/xxfb/tzgg/201504/t20150422_50363.html>.

### Academic Chinese

1. Hong Kong Baptist University, *人文中國學報*. The journal page identifies
   the open-access terms as CC BY-NC 4.0 and lists Chinese research articles:
   <https://ejournals.lib.hkbu.edu.hk/index.php/sinohumanitas/index>.
2. Hong Kong Baptist University, *International Journal of Chinese &
   Comparative Philosophy of Medicine / 中外醫學哲學*. The publisher describes
   it as a Chinese-language academic journal and applies CC BY-NC 4.0:
   <https://ejournals.lib.hkbu.edu.hk/index.php/ijccpm/about>.
3. DOAJ, *中國語文通訊*. The directory reports Chinese/English submissions and
   CC BY-NC-ND; publisher and article-level verification remains necessary:
   <https://doaj.org/toc/1726-9245>.
4. White Rose University Press, *Writing Chinese*. Its stated objective includes
   English translations of Chinese-language scholarship, so the published
   language must be checked rather than inferred from subject matter:
   <https://writingchinesejournal.org/about>.

### General Chinese

1. Chinese Wikisource copyright policy. Contributions are provided under CC
   BY-SA 4.0 and GFDL, while underlying works require work-level review:
   <https://zh.wikisource.org/wiki/Wikisource:%E7%89%88%E6%9D%83%E4%BF%A1%E6%81%AF>.

## Gate A outcome

Gate A is **partially satisfied for pilot planning**, not closed:

- a defensible Chinese professional pilot source now exists through
  document-level OGL-licensed official translations;
- Chinese general historical material remains conditionally feasible;
- Chinese academic sources exist, but the strongest verified candidates are
  non-commercial. They cannot be admitted until the project chooses a corpus
  release policy compatible with CC BY-NC;
- ordinary mainland government webpages are not approved as a blanket source.

The next Gate A decision is therefore a release-policy choice: either permit a
separately licensed non-commercial text corpus, seek a sufficiently broad
CC BY/CC0 Chinese academic source, or obtain direct permission. Metadata and
aggregate detector results may remain under a different project licence, but
licence boundaries must be explicit.

