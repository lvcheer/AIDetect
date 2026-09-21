"""Stable, JSON-serialisable result records for AIDetect."""

from dataclasses import dataclass
from typing import Optional


RESULT_SCHEMA_VERSION = "0.1"
SCORE_SCALE = "percent_0_100"


@dataclass(frozen=True)
class DetectionRecord:
    """One text result with raw, heuristic-feature, and fused scores."""

    text: str
    raw_classifier_score: float
    raw_classifier_is_ai: bool
    perplexity_value: Optional[float]
    perplexity_heuristic_score: Optional[float]
    burstiness_cv: Optional[float]
    burstiness_heuristic_score: Optional[float]
    fused_score: float
    explanation: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self):
        """Return the stable record intended for JSON/JSONL output."""
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "score_scale": SCORE_SCALE,
            "text": self.text,
            "raw_classifier_score": self.raw_classifier_score,
            "raw_classifier_is_ai": self.raw_classifier_is_ai,
            "perplexity_value": self.perplexity_value,
            "perplexity_heuristic_score": self.perplexity_heuristic_score,
            "burstiness_cv": self.burstiness_cv,
            "burstiness_heuristic_score": self.burstiness_heuristic_score,
            "fused_score": self.fused_score,
            "explanation": self.explanation,
            "error": self.error,
        }

    def to_display_dict(self):
        """Return the GUI/CSV shape without implying score calibration."""
        result = {
            "sentence": self.text,
            "ai_score": self.fused_score,
            "complement_score": round(100 - self.fused_score, 2),
            "is_ai": self.raw_classifier_is_ai,
        }
        if self.perplexity_value is not None:
            result["perplexity"] = self.perplexity_value
        if self.burstiness_cv is not None:
            result["burstiness_cv"] = self.burstiness_cv
        if self.explanation is not None:
            result["explanation"] = self.explanation
        if self.error is not None:
            result["error"] = self.error
        return result
