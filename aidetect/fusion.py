"""Existing heuristic score fusion, isolated from the GUI."""


FULL_FEATURE_WEIGHTS = {
    "classifier": 0.2,
    "perplexity": 0.6,
    "burstiness": 0.2,
}

PERPLEXITY_ONLY_WEIGHTS = {
    "classifier": 0.25,
    "perplexity": 0.75,
}

BURSTINESS_ONLY_WEIGHTS = {
    "classifier": 0.7,
    "burstiness": 0.3,
}


def fuse_heuristic_scores(
    classifier_score,
    perplexity_score=None,
    burstiness_score=None,
):
    """Combine available scores with the application's existing weights."""
    if perplexity_score is not None and burstiness_score is not None:
        fused_score = (
            classifier_score * FULL_FEATURE_WEIGHTS["classifier"]
            + perplexity_score * FULL_FEATURE_WEIGHTS["perplexity"]
            + burstiness_score * FULL_FEATURE_WEIGHTS["burstiness"]
        )
    elif perplexity_score is not None:
        fused_score = (
            classifier_score * PERPLEXITY_ONLY_WEIGHTS["classifier"]
            + perplexity_score * PERPLEXITY_ONLY_WEIGHTS["perplexity"]
        )
    elif burstiness_score is not None:
        fused_score = (
            classifier_score * BURSTINESS_ONLY_WEIGHTS["classifier"]
            + burstiness_score * BURSTINESS_ONLY_WEIGHTS["burstiness"]
        )
    else:
        fused_score = classifier_score

    return round(fused_score, 2)
