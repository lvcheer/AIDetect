"""Model registry and model-label helpers."""

from pathlib import Path
from typing import Mapping


MODEL_REGISTRY = {
    "中文优先（RoBERTa）": "Hello-SimpleAI/chatgpt-detector-roberta-chinese",
    "中文新版（AIGC v2）": "yuchuantian/AIGC_detector_zhv2",
    "英文通用（OpenAI Detector）": "roberta-base-openai-detector",
    "英文新版（TMR Detector）": "Oxidane/tmr-ai-text-detector",
    "多语言（ChatGPT Detector）": "Hello-SimpleAI/chatgpt-detector-roberta",
}

AI_LABEL_KEYWORDS = ("fake", "chatgpt", "ai", "machine", "generated", "aigc")


def local_model_path(models_dir, model_id):
    """Return the on-disk directory used by ``download_models.py``."""
    return Path(models_dir) / model_id.replace("/", "__")


def find_ai_label_index(id2label: Mapping, default=1):
    """Find the class index whose label describes AI-generated text.

    Hugging Face configurations may expose integer or string keys. The fallback
    preserves the existing application's convention for generic ``LABEL_0`` /
    ``LABEL_1`` classifiers.
    """
    for raw_index, label in id2label.items():
        label_text = str(label).lower()
        if any(keyword in label_text for keyword in AI_LABEL_KEYWORDS):
            try:
                return int(raw_index)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid label index: {raw_index!r}") from exc
    return default
