# Bilingual Prompt Packet Template

Packet version: `0.1-draft`

This template defines structure, not prompt content. Each completed packet is
reviewed and hashed before generation.

## Packet metadata

- `prompt_family_id`:
- `language`: `zh | en`
- `domain`: `general | academic | professional`
- `length_band`: `short | medium | long`
- `topic_family`:
- `audience`:
- `genre`:
- `facts_or_scenario_source`: `independently authored | reusable source + licence`
- `author` and reviewer:
- `packet_version` and SHA-256:
- related-language packet ID, if difficulty-matched:

The facts/scenario must not reproduce any human benchmark text. It must avoid
sensitive personal data, current high-stakes advice, partisan persuasion,
copyrighted fictional characters, and facts likely to require web retrieval.

## Shared system instruction

### Chinese

> 你将根据用户提供的主题资料撰写一篇独立、连贯的中文文章。只输出正文，不要说明写作过程，不要使用网络、工具或外部资料。不要添加引文、参考文献或无法由主题资料支持的具体事实。遵守指定的文体、读者和长度范围。

### English

> Write a self-contained, coherent English passage from the topic packet supplied by the user. Output only the passage. Do not use the web, tools, or external sources. Do not add citations, references, or specific facts unsupported by the packet. Follow the specified genre, audience, and length range.

## User template

### Chinese

```text
领域：{domain}
文体：{genre}
目标读者：{audience}
主题与可用信息：
{independently_authored_topic_packet}

长度：{minimum}–{maximum} 个汉字（不计空格）。
要求：正文应自成一体；使用自然段；不要列出参考文献；不要提及这些写作指令。
```

### English

```text
Domain: {domain}
Genre: {genre}
Audience: {audience}
Topic and permitted information:
{independently_authored_topic_packet}

Length: {minimum}–{maximum} words.
Requirements: Make the passage self-contained; use prose paragraphs; do not provide references; do not mention these instructions.
```

## Domain-specific contract

- `general`: explain or narrate for a non-specialist; no invented quotations or
  personal experience presented as real.
- `academic`: state a research question and a plausible approach, result or
  expected contribution, and limitation exactly as supplied; no invented
  authors, institutions, statistics, citations, or completed empirical result.
- `professional`: identify the audience and purpose, then provide neutral,
  actionable guidance within the supplied scenario; no legal or medical advice.

## Mechanical validation

- output contains prose only and no instruction preamble;
- assigned language and length band are satisfied;
- no URL, citation marker, bibliography heading, or tool trace appears;
- packet phrases are checked for excessive verbatim copying;
- exact response, request, metadata, and hashes are retained even on failure.

No reviewer may select the “best” among multiple valid generations. The retry
and failure policy in `gate_c_pilot_protocol.md` controls acceptance.

