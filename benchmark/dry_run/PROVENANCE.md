# Dry-run fixture provenance

Status: pipeline validation only. These records must not be used for formal
performance claims.

Collection date: 2026-09-22

## Human-authored source records

### English

- Work: *Pride and Prejudice* by Jane Austen (1813).
- Source: Project Gutenberg eBook 1342,
  https://www.gutenberg.org/ebooks/1342
- Retrieved plain-text artifact:
  https://www.gutenberg.org/cache/epub/1342/pg1342.txt
- Retrieved artifact SHA-256:
  `3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75`
- Source metadata on retrieval: most recently updated 2026-09-01; marked
  public domain in the USA. Jane Austen died in 1817, so the underlying work
  is also beyond the ordinary UK life-plus-70 term.
- Extraction: `en_human_short.txt` is an excerpt from Chapter I;
  `en_human_long.txt` is the prose of Chapter II. Project Gutenberg headers,
  footers, illustrations, captions, emphasis markup, and line wrapping were
  removed. No modern synopsis was copied.
- Recorded permission category: `Public-Domain-Underlying-Work; Project-Gutenberg-Source`.

### Chinese

- Work: 《狂人日记》 by 鲁迅 (1918).
- Fixed source revision:
  https://zh.wikisource.org/w/index.php?title=狂人日記&oldid=2605391
- The fixed page marks the underlying work as public domain and records that
  Lu Xun died in 1936. Page transcription and contribution terms are provided
  under CC BY-SA 4.0; source attribution is retained here.
- Extraction: `zh_human_short.txt` uses the beginning of sections 一 and 二;
  `zh_human_long.txt` uses sections 三 and 四. Page navigation, annotations,
  and template text were removed; wording and punctuation were retained.
- Recorded permission category: `Public-Domain-Underlying-Work; CC-BY-SA-4.0-Transcription`.

## AI-generated source records

The four AI records were written directly for this fixture by the OpenAI
Codex assistant in the project task on 2026-09-22. The task interface does not
expose an immutable generator revision or sampling temperature, so both are
recorded as `null`; they are not guessed. The project owner explicitly
authorized creation of these dry-run fixtures in the task conversation.

Generator identifier: `OpenAI Codex task model`

### Prompt `dry-en-short-v1`

> Write an original 130–180 word English passage about a neighborhood library
> preparing for a power outage. Do not quote or imitate any named author.

Output: `texts/en_ai_short.txt`

### Prompt `dry-en-long-v1`

> Write an original 900–1100 word English expository article about how a
> coastal town prepares a community science fair. Avoid named authors,
> quotations, and copyrighted characters.

Output: `texts/en_ai_long.txt`

### Prompt `dry-zh-short-v1`

> 请写一段原创中文短文（约250–350字），主题为社区图书馆在暴雨前做准备，
> 不引用或模仿任何具体作者。

Output: `texts/zh_ai_short.txt`

### Prompt `dry-zh-long-v1`

> 请写一篇原创中文说明文（约1200–1500字），主题为海滨小镇筹备社区科学展，
> 不引用或模仿任何具体作者、作品或受版权保护的角色。

Output: `texts/zh_ai_long.txt`

Recorded permission category for all four outputs:
`Project-Generated-User-Authorized-Dry-Run-Fixture`.

## Length and truncation controls

Token counts were measured offline with the exact cached tokenizer revisions
selected for the first dry run. They are descriptive fixture checks, not model
results.

| Record | Tokenizer | Revision | Tokens |
|---|---|---|---:|
| `en_human_short.txt` | `Oxidane/tmr-ai-text-detector` | `0ceddea903015ef99cbaa040a4d8a216aed9c683` | 219 |
| `en_human_long.txt` | `Oxidane/tmr-ai-text-detector` | `0ceddea903015ef99cbaa040a4d8a216aed9c683` | 1,212 |
| `en_ai_short.txt` | `Oxidane/tmr-ai-text-detector` | `0ceddea903015ef99cbaa040a4d8a216aed9c683` | 201 |
| `en_ai_long.txt` | `Oxidane/tmr-ai-text-detector` | `0ceddea903015ef99cbaa040a4d8a216aed9c683` | 1,085 |
| `zh_human_short.txt` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | `2f0f5f2af59af169a3f236c14ffad0fa7d6f072b` | 283 |
| `zh_human_long.txt` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | `2f0f5f2af59af169a3f236c14ffad0fa7d6f072b` | 1,275 |
| `zh_ai_short.txt` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | `2f0f5f2af59af169a3f236c14ffad0fa7d6f072b` | 329 |
| `zh_ai_long.txt` | `Hello-SimpleAI/chatgpt-detector-roberta-chinese` | `2f0f5f2af59af169a3f236c14ffad0fa7d6f072b` | 1,554 |

The short records remain below 512 tokens and every long record exceeds 512,
so both non-truncated and truncated paths can be checked explicitly.
