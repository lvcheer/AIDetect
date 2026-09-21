"""Reusable inference components for AIDetect."""

from .features import (
    BurstinessFeature,
    PerplexityFeature,
    calculate_burstiness_feature,
    calculate_perplexity_feature,
)
from .fusion import fuse_heuristic_scores
from .inference import RawScore, infer_raw_score
from .models import MODEL_REGISTRY, find_ai_label_index, local_model_path
from .pipeline import detect_text
from .schema import DetectionRecord, RESULT_SCHEMA_VERSION

__all__ = [
    "BurstinessFeature",
    "DetectionRecord",
    "MODEL_REGISTRY",
    "PerplexityFeature",
    "RESULT_SCHEMA_VERSION",
    "RawScore",
    "calculate_burstiness_feature",
    "calculate_perplexity_feature",
    "detect_text",
    "find_ai_label_index",
    "fuse_heuristic_scores",
    "infer_raw_score",
    "local_model_path",
]
