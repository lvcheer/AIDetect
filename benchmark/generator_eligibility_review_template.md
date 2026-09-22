# Generator Eligibility Review

Review date: `YYYY-MM-DD`

Reviewer: `NAME`

Decision: `eligible | conditional | excluded`

This review is a reproducibility and research-use screen, not legal advice.
Complete it again immediately before a pilot or formal generation run.

## 1. Identity and access

- Provider and model family:
- Requested model ID:
- Resolved immutable or dated revision:
- Official model-metadata response and SHA-256:
- API, SDK, or local runtime version:
- Region and account tier:
- Stable/GA rather than preview or experimental: `yes | no`
- Earliest announced retirement date:
- At least 120 days of expected availability from collection start: `yes | no`

## 2. Research suitability

- Chinese text output supported by official documentation and pilot: `yes | no`
- English text output supported by official documentation and pilot: `yes | no`
- Tools, search, retrieval, grounding, and browsing disabled: `yes | no`
- A single response can satisfy the frozen length range: `yes | no`
- Refusals and safety blocks can be retained rather than silently replaced: `yes | no`
- Prompt and output token usage is returned or independently countable: `yes | no`

## 3. Rights and data handling

- Governing terms and effective date:
- Provider claim concerning output ownership or non-ownership:
- Restrictions relevant to detector research, publication, or redistribution:
- Input rights required:
- Training use of API inputs/outputs:
- Retention and abuse-monitoring policy:
- Redistribution decision: `full text | controlled access | hashes/metadata only | excluded`
- Evidence URLs and archived copies/hashes:

The reviewer must not treat provider ownership language as a guarantee that an
output is copyrightable, unique, accurate, or free of third-party rights.

## 4. Version and decoding controls

- Alias-to-revision resolution method:
- System and user prompt packet version/hash:
- Native generation parameters actually supported:
- Thinking/reasoning setting and whether its tokens are billed:
- Seed support and value, if any:
- Maximum output tokens:
- Response fields retained: request ID, resolved model, usage, finish reason,
  safety status, timestamp, raw response hash.

Numerically equal sampling parameters across providers are not assumed to be
equivalent. Unsupported parameters must be omitted, not emulated after the
fact. Any model-revision change creates a new generator stratum.

## 5. Budget

- Standard input price per million tokens:
- Standard output price per million tokens:
- Batch discount and completion window, if used:
- Pricing page retrieval date and archived hash:
- Pilot token estimate:
- Formal upper-bound estimate, including 20% retry/refusal reserve:
- Approved spending cap and approver:

No paid request is authorised by this review alone.

## 6. Decision

- Intended role: `development | held-out | contingency | excluded`
- Unresolved conditions:
- Rationale:
- Signature/date:

