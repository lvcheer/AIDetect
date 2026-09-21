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
