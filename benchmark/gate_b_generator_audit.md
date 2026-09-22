# Gate B Generator Audit and Provisional Selection

Audit date: `2026-09-22`

Status: **Gate B planning evidence; no API calls or generated samples**

## Decision

Use three provider-disjoint commercial API families for the formal design,
subject to a final preflight review:

| Role | Provisional family | Why | Condition before use |
|---|---|---|---|
| development A | OpenAI GPT-5.6 Luna | economical high-volume text generation; official API documentation lists snapshot locking | an actual immutable snapshot, not a moving alias, must be returned and recorded at freeze |
| development B | Google Gemini 3.8 Flash | stable endpoint, low current price, and explicit stable/preview/latest lifecycle documentation | use paid service; confirm the stable endpoint and no announced shutdown immediately before collection |
| generator-held-out | Anthropic Claude Sonnet 5 | provider- and family-disjoint multilingual generator with an announced support horizon | held-out status is lost if any output, label, or detector score is inspected during development; freeze the dated snapshot behind the alias |
| reproducibility contingency | Qwen3 open-weight checkpoint | fixed weights and Apache-2.0 model licence can provide a locally reproducible sensitivity analysis | not in the primary three unless exact checkpoint, commit, hashes, runtime, quantisation, hardware, and adequate bilingual pilot quality are approved |

This allocation deliberately holds out a **generator family**, not merely a
model name. Provider, model family, prompt packet, and all derivatives stay in
the same leakage component. The held-out family cannot be used for prompt
tuning, threshold selection, preprocessing choices, or qualitative error
analysis before the analysis freeze.

The recommendation is provisional because present OpenAI and Anthropic family
names are aliases in their public catalogues. If an immutable/durable revision
cannot be established, that candidate is ineligible rather than silently used
under an alias. Roles may be reassigned before any sample is generated, but the
reason and date must be committed before the pilot.

## Evidence review

| Candidate | Version evidence | Language evidence | Current standard text price (USD / 1M tokens) | Terms/data finding | Decision |
|---|---|---|---|---|---|
| GPT-5.6 Luna | its model page says snapshots lock behaviour, but the page currently exposes only the family ID; this is not yet sufficient for the formal freeze | text input/output; bilingual ability still requires a balanced pilot | `$0.20` input, `$1.20` output | business/API agreement assigns output to the customer to the extent permitted by law; API content is not used to improve models unless the customer opts in; restrictions still apply | conditional development A |
| Gemini 3.8 Flash | official endpoint is marked stable; Google says specific stable models usually do not change, while `latest` hot-swaps and preview may receive only two weeks' notice | official Gemini materials describe multilingual models; Chinese and English quality still require the same pilot | `$0.75` input, `$3.75` output through 2026-12-31; scheduled `$1.50`/`$7.50` thereafter | Google does not claim ownership of generated content; paid-service prompts/responses are not used to improve products, but limited abuse logging remains | conditional development B |
| Claude Sonnet 5 | `claude-sonnet-5` is an alias/current ID; Anthropic documents snapshot aliases and lists no retirement earlier than 2027-06-30, but the dated snapshot must be captured | official model overview states all current models are multilingual; pilot remains required | `$2.00` input, `$10.00` output | commercial API materials state customers retain output ownership; commercial customer data is not used for model training by default, subject to current account terms | conditional held-out |
| Qwen3 open weights | a Hugging Face commit and every weight/tokenizer/config hash can be pinned | official project describes multilingual support; a Chinese/English pilot and capacity match are required | no token fee; compute, storage, electricity, and engineering cost remain | official Qwen3 repository says its open-weight models use Apache-2.0; the selected checkpoint's own licence file controls | contingency only |

Prices are observations on the audit date, not a budget authorisation. They
must be refreshed and archived on the collection date. Batch pricing may be
used only if the same service tier is applied within a generator stratum and
the provider returns enough audit metadata.

## Why equal temperature is not the control

Claude Sonnet 5 rejects non-default `temperature`, `top_p`, and `top_k` values.
Gemini 3.8 Flash documentation likewise instructs clients to remove those
sampling parameters. Consequently, forcing a numerical temperature across
providers would exclude current models or create provider-specific behaviour
that is falsely described as equivalent.

The controlled policy is instead:

1. use exactly the same versioned topic packet, genre instruction, target
   language, and target-length band within a matched prompt block;
2. disable tools, web search, grounding, retrieval, and conversational history;
3. use each provider's documented native default sampling and explicitly
   disable optional reasoning where supported and scientifically appropriate;
4. retain the complete request configuration and raw response metadata;
5. treat generator/provider as a design factor rather than nuisance noise.

If a provider cannot disable hidden or adaptive reasoning, this remains a
documented model characteristic. Reasoning tokens are counted in cost and the
provider is not post-processed to mimic another model.

## Freeze and drift controls

Immediately before each pilot or formal run:

1. Complete `generator_eligibility_review_template.md` for every candidate.
2. Query the official model metadata endpoint, retain the raw response, and
   hash it. Resolve aliases to a dated/immutable revision where supported.
3. Recheck deprecation pages, governing terms, data retention, regional access,
   redistribution, and prices. Require at least 120 days of expected access.
4. Freeze SDK/runtime versions, prompt packet hash, model ID, native parameters,
   maximum output, and account/service tier.
5. Run a small bilingual pilot excluded from all final partitions. Accept only
   candidates meeting the same refusal, length, encoding, and quality rules.
6. Collect each generator stratum in a compact window. A revision change stops
   collection; it never silently continues under the same generator label.
7. Archive request IDs, timestamps, returned model identifiers, usage, finish
   reasons, safety blocks, raw response hashes, and all failures. Do not replace
   refusals after inspecting detector results.

Provider retirement may make an exact rerun impossible. Therefore the release
must contain permitted outputs or their controlled-access package, prompt and
response hashes, model metadata, terms/pricing snapshots, and complete
provenance—not just a family name.

## Budget rule

For generator `g`, calculate before approval:

`cost_g = input_tokens_g × input_rate_g + billed_output_tokens_g × output_rate_g`

Add provider-specific cache, reasoning, batch, or service-tier charges, then a
20% reserve for auditable retries and refusals. The estimate must show low,
expected, and hard-cap scenarios using pilot token counts. Generation software
must stop before the approved hard cap; a reviewer must explicitly authorise
any increase. Free tiers are not used merely to avoid approval because their
data handling, capacity, and service guarantees may differ.

## Redistribution decision

Gate B finds no automatic provider-ownership bar to research release for the
three API candidates, but this is **not** blanket legal clearance. Before
release, confirm the account's then-current governing agreement, input-source
rights, required attribution, similarity risk, and any prohibition on using
outputs to develop competing models. Because this project evaluates detectors,
the release purpose and whether any trained artefact is distributed must be
checked explicitly. When uncertain, release hashes, metadata, prompts where
permitted, and evaluation results rather than full generated text.

## Official evidence consulted

- OpenAI GPT-5.6 Luna model page and pricing:
  <https://developers.openai.com/api/docs/models/gpt-5.6-luna>
- OpenAI Services Agreement (effective 2026-01-01):
  <https://openai.com/policies/services-agreement/>
- OpenAI API data controls:
  <https://platform.openai.com/docs/models/default-usage-policies-by-endpoint>
- Gemini model lifecycle and stable endpoints:
  <https://ai.google.dev/gemini-api/docs/models>
- Gemini pricing:
  <https://ai.google.dev/gemini-api/docs/pricing>
- Gemini API Additional Terms (effective 2026-03-23):
  <https://ai.google.dev/gemini-api/terms>
- Anthropic model overview and versioning:
  <https://platform.claude.com/docs/en/models/overview> and
  <https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions>
- Anthropic pricing and model deprecations:
  <https://platform.claude.com/docs/en/about-claude/pricing> and
  <https://platform.claude.com/docs/en/about-claude/model-deprecations>
- Anthropic commercial API data handling and output-rights announcement:
  <https://support.anthropic.com/en/articles/9267385-does-anthropic-act-as-a-data-processor-or-controller>
  and <https://www.anthropic.com/news/expanded-legal-protections-api-improvements>
- Qwen3 official repository and checkpoint licences:
  <https://github.com/QwenLM/Qwen3>

