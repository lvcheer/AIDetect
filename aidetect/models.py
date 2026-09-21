"""Model registry and model-label helpers."""

from dataclasses import dataclass
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
PERPLEXITY_MODEL_ID = "uer/gpt2-chinese-cluecorpussmall"


@dataclass(frozen=True)
class LoadedClassifier:
    tokenizer: object
    model: object
    device: str
    ai_label_index: int
    source: object


@dataclass(frozen=True)
class LoadedPerplexityModel:
    tokenizer: object
    model: object
    device: str
    source: str


def local_model_path(models_dir, model_id):
    """Return the on-disk directory used by ``download_models.py``."""
    return Path(models_dir) / model_id.replace("/", "__")


def resolve_model_source(models_dir, model_id):
    """Prefer a downloaded model directory and otherwise use the remote ID."""
    local_path = local_model_path(models_dir, model_id)
    return local_path if local_path.exists() else model_id


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


def load_classifier(
    model_id,
    models_dir,
    tokenizer_loader=None,
    model_loader=None,
    torch_module=None,
):
    """Load and prepare one sequence classifier using the existing policy."""
    if tokenizer_loader is None or model_loader is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        model_loader = model_loader or AutoModelForSequenceClassification.from_pretrained
    if torch_module is None:
        import torch as torch_module

    source = resolve_model_source(models_dir, model_id)
    tokenizer = tokenizer_loader(source)
    model = model_loader(source)
    device = "cuda" if torch_module.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    ai_label_index = find_ai_label_index(model.config.id2label)
    return LoadedClassifier(
        tokenizer=tokenizer,
        model=model,
        device=device,
        ai_label_index=ai_label_index,
        source=source,
    )


def load_perplexity_model(
    device="cpu",
    model_id=PERPLEXITY_MODEL_ID,
    tokenizer_loader=None,
    model_loader=None,
):
    """Load and prepare the existing causal language model feature."""
    if tokenizer_loader is None or model_loader is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer_loader = tokenizer_loader or AutoTokenizer.from_pretrained
        model_loader = model_loader or AutoModelForCausalLM.from_pretrained

    tokenizer = tokenizer_loader(model_id)
    model = model_loader(model_id)
    model.to(device)
    model.eval()
    return LoadedPerplexityModel(
        tokenizer=tokenizer,
        model=model,
        device=device,
        source=model_id,
    )
