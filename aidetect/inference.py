"""Raw-score inference independent of the desktop GUI."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RawScore:
    """Uncalibrated class scores returned by a sequence classifier."""

    text: str
    scores: tuple
    ai_label_index: int

    @property
    def ai_score(self):
        return self.scores[self.ai_label_index]

    @property
    def is_ai(self):
        return self.ai_score > 0.5


@dataclass(frozen=True)
class TokenizationInfo:
    """Input-length and truncation metadata for one tokenizer call policy."""

    input_token_length: int
    effective_token_length: int
    max_length: int
    truncated: bool
    truncation_side: str


def inspect_tokenization(text, tokenizer, max_length=512):
    """Measure token length before applying the inference truncation limit."""
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids = encoded["input_ids"]
    if hasattr(input_ids, "shape"):
        input_token_length = int(input_ids.shape[-1])
    elif input_ids and isinstance(input_ids[0], (list, tuple)):
        input_token_length = len(input_ids[0])
    else:
        input_token_length = len(input_ids)
    return TokenizationInfo(
        input_token_length=input_token_length,
        effective_token_length=min(input_token_length, max_length),
        max_length=max_length,
        truncated=input_token_length > max_length,
        truncation_side=str(getattr(tokenizer, "truncation_side", "right")),
    )


def infer_raw_score(text, tokenizer, model, device, ai_label_index, max_length=512):
    """Run one text through a classifier and return uncalibrated scores."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    normalized_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    scores = tuple(float(value) for value in normalized_scores.detach().cpu().tolist())
    if not 0 <= ai_label_index < len(scores):
        raise ValueError(
            f"AI label index {ai_label_index} is outside {len(scores)} model classes"
        )

    return RawScore(text=text, scores=scores, ai_label_index=ai_label_index)
