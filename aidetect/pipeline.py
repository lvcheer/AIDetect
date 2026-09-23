"""Non-GUI orchestration for one AIDetect text result."""

from .features import calculate_burstiness_feature, calculate_perplexity_feature
from .fusion import fuse_heuristic_scores
from .inference import infer_raw_score
from .schema import DetectionRecord


def generate_explanation(fused_score, perplexity_value=None, burstiness_cv=None):
    """Generate the application's existing score-based explanation."""
    if fused_score < 30:
        base = "文本符合人类写作特征，语言自然，逻辑有正常波动，无明显AI痕迹。"
    elif fused_score < 70:
        base = "文本疑似混合生成，部分语句结构较规整，存在AI特征但仍有人类表达痕迹。"
    else:
        base = "文本高度疑似AI生成，语言过于规范，句式模板化，缺少人类写作的情感起伏。"

    parts = [base]
    if perplexity_value is not None:
        if perplexity_value < 25:
            parts.append(
                f"困惑度 {perplexity_value}（极低）：文本对语言模型几乎没有意外，"
                "流畅度异常高，强烈暗示AI生成。"
            )
        elif perplexity_value < 45:
            parts.append(
                f"困惑度 {perplexity_value}（偏低）：文本较为流畅规律，有一定AI生成可能。"
            )
        else:
            parts.append(
                f"困惑度 {perplexity_value}（正常）：文本流畅度在人类写作正常范围内。"
            )

    if burstiness_cv is not None:
        if burstiness_cv < 0.2:
            parts.append(
                f"突发性 CV={burstiness_cv}（极低）：句子长度高度均匀，"
                "缺乏人类写作的节奏变化，强烈暗示AI生成。"
            )
        elif burstiness_cv < 0.4:
            parts.append(
                f"突发性 CV={burstiness_cv}（偏低）：句子长度较均匀，AI风格明显。"
            )
        else:
            parts.append(
                f"突发性 CV={burstiness_cv}（正常）：句子长度有自然波动，"
                "符合人类写作节奏。"
            )

    return " | ".join(parts)


def detect_text(
    text,
    tokenizer,
    model,
    device,
    ai_label_index,
    perplexity_tokenizer=None,
    perplexity_model=None,
):
    """Run the existing classifier, optional features, and fusion for one text."""
    error = None
    try:
        raw_score = infer_raw_score(
            text,
            tokenizer=tokenizer,
            model=model,
            device=device,
            ai_label_index=ai_label_index,
        )
        classifier_score = round(raw_score.ai_score * 100, 2)
        classifier_is_ai = raw_score.is_ai
    except Exception as exc:
        classifier_score = 0.0
        classifier_is_ai = False
        error = str(exc)

    perplexity_value = None
    perplexity_score = None
    if perplexity_tokenizer is not None and perplexity_model is not None:
        try:
            feature = calculate_perplexity_feature(
                text,
                tokenizer=perplexity_tokenizer,
                model=perplexity_model,
            )
            perplexity_value = round(feature.perplexity, 2)
            perplexity_score = round(feature.heuristic_score, 2)
        except Exception:
            pass

    burstiness_feature = calculate_burstiness_feature(text)
    if burstiness_feature is None:
        burstiness_cv = None
        burstiness_score = None
    else:
        burstiness_cv = round(burstiness_feature.coefficient_of_variation, 3)
        burstiness_score = round(burstiness_feature.heuristic_score, 2)

    fused_score = fuse_heuristic_scores(
        classifier_score,
        perplexity_score=perplexity_score,
        burstiness_score=burstiness_score,
    )
    explanation = generate_explanation(
        fused_score,
        perplexity_value=perplexity_value,
        burstiness_cv=burstiness_cv,
    )
    return DetectionRecord(
        text=text,
        raw_classifier_score=classifier_score,
        raw_classifier_is_ai=classifier_is_ai,
        perplexity_value=perplexity_value,
        perplexity_heuristic_score=perplexity_score,
        burstiness_cv=burstiness_cv,
        burstiness_heuristic_score=burstiness_score,
        fused_score=fused_score,
        explanation=explanation,
        error=error,
    )
