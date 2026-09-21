"""Standalone perplexity and sentence-length features."""

import math
import re
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerplexityFeature:
    perplexity: float
    heuristic_score: float


@dataclass(frozen=True)
class BurstinessFeature:
    coefficient_of_variation: float
    heuristic_score: float


def calculate_perplexity_feature(text, tokenizer, model, max_length=512):
    """Calculate language-model perplexity and the existing heuristic score."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    ).to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss

    perplexity = torch.exp(loss).item()
    heuristic_score = 1 / (1 + math.exp((perplexity - 40) / 12)) * 100
    return PerplexityFeature(
        perplexity=perplexity,
        heuristic_score=heuristic_score,
    )


def calculate_burstiness_feature(text):
    """Calculate sentence-length variation and the existing heuristic score."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[。！？；.!?;]", text)
        if len(sentence.strip()) > 5
    ]
    if len(sentences) < 3:
        return None

    lengths = [len(sentence) for sentence in sentences]
    mean_length = sum(lengths) / len(lengths)
    if mean_length == 0:
        return None

    variance = sum((length - mean_length) ** 2 for length in lengths) / len(lengths)
    coefficient_of_variation = variance**0.5 / mean_length
    heuristic_score = (
        1 / (1 + math.exp((coefficient_of_variation - 0.4) / 0.15)) * 100
    )
    return BurstinessFeature(
        coefficient_of_variation=coefficient_of_variation,
        heuristic_score=heuristic_score,
    )
