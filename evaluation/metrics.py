"""Text-generation evaluation metrics.

We implement the metric computations behind a thin, dependency-tolerant API so
that:
  * the evaluation pipeline reads cleanly (``compute_all_metrics(...)``),
  * each metric degrades gracefully if its optional library is missing,
  * the functions are individually unit-testable on toy inputs.

Metrics
-------
ROUGE     : n-gram + longest-common-subsequence overlap (recall-oriented).
BLEU      : n-gram precision with brevity penalty (via sacrebleu).
BERTScore : semantic similarity using contextual embeddings.
Latency   : measured in the evaluation loop, summarised here.
"""

from __future__ import annotations

import statistics
from typing import Any

from src.utils import get_logger

logger = get_logger(__name__)


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1/2/L F-measure (mean over examples)."""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        logger.warning("rouge-score not installed -> skipping ROUGE")
        return {}

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )
    agg = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references, strict=False):
        scores = scorer.score(ref, pred)
        for key in agg:
            agg[key].append(scores[key].fmeasure)
    return {k: round(statistics.mean(v), 4) if v else 0.0 for k, v in agg.items()}


def compute_bleu(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute corpus BLEU using sacreBLEU (0-100 scale)."""
    try:
        import sacrebleu
    except ImportError:
        logger.warning("sacrebleu not installed -> skipping BLEU")
        return {}

    # sacreBLEU expects references as a list of reference-streams.
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {"bleu": round(bleu.score, 4)}


def compute_bertscore(
    predictions: list[str], references: list[str], lang: str = "en"
) -> dict[str, float]:
    """Compute mean BERTScore precision/recall/F1."""
    try:
        from bert_score import score as bert_score
    except ImportError:
        logger.warning("bert-score not installed -> skipping BERTScore")
        return {}

    try:
        precision, recall, f1 = bert_score(
            predictions, references, lang=lang, verbose=False, rescale_with_baseline=False
        )
        return {
            "bertscore_precision": round(precision.mean().item(), 4),
            "bertscore_recall": round(recall.mean().item(), 4),
            "bertscore_f1": round(f1.mean().item(), 4),
        }
    except Exception as exc:  # noqa: BLE001 - downloads a model; can fail offline
        logger.warning("BERTScore failed (%s) -> skipping", exc)
        return {}


def summarise_latency(latencies_ms: list[float]) -> dict[str, float]:
    """Summarise per-example latency measurements."""
    if not latencies_ms:
        return {}
    ordered = sorted(latencies_ms)
    p95_idx = max(0, int(len(ordered) * 0.95) - 1)
    return {
        "latency_ms_mean": round(statistics.mean(latencies_ms), 1),
        "latency_ms_median": round(statistics.median(latencies_ms), 1),
        "latency_ms_p95": round(ordered[p95_idx], 1),
    }


def compute_all_metrics(
    predictions: list[str],
    references: list[str],
    latencies_ms: list[float] | None = None,
) -> dict[str, Any]:
    """Compute the full metric suite for one model's predictions."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have equal length")

    metrics: dict[str, Any] = {}
    metrics.update(compute_rouge(predictions, references))
    metrics.update(compute_bleu(predictions, references))
    metrics.update(compute_bertscore(predictions, references))
    if latencies_ms:
        metrics.update(summarise_latency(latencies_ms))
    metrics["num_examples"] = len(predictions)
    return metrics
