"""Evaluation pipeline: base model vs QLoRA fine-tuned model.

Generates predictions on the held-out test set with BOTH the base model and the
fine-tuned (base + LoRA) model, scores them with ROUGE / BLEU / BERTScore +
latency, and writes a human-readable ``evaluation_report.md`` (plus a JSON dump
of raw numbers and qualitative side-by-side examples).

Usage
-----
    python -m evaluation.evaluate --num-samples 100
    python -m evaluation.evaluate --num-samples 50 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import EVAL_DIR, PROCESSED_DATA_DIR, get_config, get_settings
from src.utils import get_logger

logger = get_logger(__name__)


def _load_test_examples(num_samples: int | None) -> list[dict[str, str]]:
    """Load the processed test split."""
    from data.preprocessing import read_jsonl

    cfg = get_config().data
    path = PROCESSED_DATA_DIR / cfg.test_file
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m data.preprocessing` first."
        )
    examples = read_jsonl(path)
    if num_samples:
        examples = examples[:num_samples]
    logger.info("Loaded %d test examples", len(examples))
    return examples


def _generate_predictions(
    assistant,
    examples: list[dict[str, str]],
    max_new_tokens: int | None,
) -> tuple[list[str], list[float]]:
    """Run generation over the test set; return (predictions, latencies_ms)."""
    preds: list[str] = []
    latencies: list[float] = []
    for i, ex in enumerate(examples):
        result = assistant.generate(
            ex["input"],
            instruction=ex.get("instruction"),
            max_new_tokens=max_new_tokens,
        )
        preds.append(result.answer)
        latencies.append(result.latency_ms)
        if (i + 1) % 10 == 0:
            logger.info("  generated %d/%d", i + 1, len(examples))
    return preds, latencies


def evaluate(
    num_samples: int = 100,
    max_new_tokens: int | None = 256,
) -> dict[str, Any]:
    """Run the full base-vs-finetuned evaluation and write the report."""
    from src.inference import HealthcareAssistant

    from evaluation.metrics import compute_all_metrics

    settings = get_settings()
    examples = _load_test_examples(num_samples)
    references = [ex["output"] for ex in examples]

    # --- Base model (no adapter) ------------------------------------------- #
    logger.info("Evaluating BASE model: %s", settings.base_model)
    base_assistant = HealthcareAssistant(adapter_path=None)
    base_preds, base_lat = _generate_predictions(base_assistant, examples, max_new_tokens)
    base_metrics = compute_all_metrics(base_preds, references, base_lat)

    # --- Fine-tuned model (base + LoRA) ------------------------------------ #
    logger.info("Evaluating FINE-TUNED model (adapter=%s)", settings.adapter_path)
    ft_assistant = HealthcareAssistant(adapter_path=settings.adapter_path)
    ft_preds, ft_lat = _generate_predictions(ft_assistant, examples, max_new_tokens)
    ft_metrics = compute_all_metrics(ft_preds, references, ft_lat)

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "base_model": settings.base_model,
        "adapter_path": settings.adapter_path,
        "num_samples": len(examples),
        "base": base_metrics,
        "finetuned": ft_metrics,
    }

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "results.json").write_text(json.dumps(report, indent=2))

    qualitative = _build_qualitative(examples, base_preds, ft_preds, limit=5)
    _write_markdown_report(report, qualitative)
    logger.info("Evaluation complete. Report -> %s", get_config())
    return report


def _build_qualitative(
    examples: list[dict[str, str]],
    base_preds: list[str],
    ft_preds: list[str],
    limit: int = 5,
) -> list[dict[str, str]]:
    """Collect a few side-by-side examples for human evaluation."""
    rows = []
    for ex, b, f in zip(examples[:limit], base_preds[:limit], ft_preds[:limit], strict=False):
        rows.append(
            {
                "question": ex["input"],
                "reference": ex["output"],
                "base": b,
                "finetuned": f,
            }
        )
    return rows


def _fmt_delta(base: float | None, ft: float | None) -> str:
    """Format a metric improvement (fine-tuned minus base)."""
    if base is None or ft is None:
        return "n/a"
    delta = ft - base
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _write_markdown_report(report: dict[str, Any], qualitative: list[dict[str, str]]) -> None:
    """Render ``evaluation_report.md`` from the results dict."""
    base = report["base"]
    ft = report["finetuned"]
    metric_keys = sorted(set(base) | set(ft) - {"num_examples"})

    lines: list[str] = []
    lines.append("# Evaluation Report — Healthcare QLoRA")
    lines.append("")
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Base model:** `{report['base_model']}`")
    lines.append(f"- **Adapter:** `{report['adapter_path']}`")
    lines.append(f"- **Test samples:** {report['num_samples']}")
    lines.append("")
    lines.append("## Quantitative Results (Base vs Fine-tuned)")
    lines.append("")
    lines.append("| Metric | Base | Fine-tuned | Δ (FT − Base) |")
    lines.append("| --- | --- | --- | --- |")
    for key in metric_keys:
        if key == "num_examples":
            continue
        b = base.get(key)
        f = ft.get(key)
        b_s = f"{b:.4f}" if isinstance(b, (int, float)) else "n/a"
        f_s = f"{f:.4f}" if isinstance(f, (int, float)) else "n/a"
        delta = _fmt_delta(b if isinstance(b, (int, float)) else None,
                           f if isinstance(f, (int, float)) else None)
        lines.append(f"| {key} | {b_s} | {f_s} | {delta} |")
    lines.append("")
    lines.append(
        "> Higher is better for ROUGE/BLEU/BERTScore. For latency, lower is "
        "better; both models run on the same hardware/quantization so latency "
        "differences reflect adapter overhead only."
    )
    lines.append("")
    lines.append("## Qualitative / Human Evaluation Examples")
    lines.append("")
    for i, row in enumerate(qualitative, start=1):
        lines.append(f"### Example {i}")
        lines.append("")
        lines.append(f"**Question:** {row['question']}")
        lines.append("")
        lines.append(f"**Reference answer:** {row['reference']}")
        lines.append("")
        lines.append(f"**Base model:** {row['base']}")
        lines.append("")
        lines.append(f"**Fine-tuned model:** {row['finetuned']}")
        lines.append("")
        lines.append("---")
        lines.append("")

    out_path = Path("evaluation_report.md")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path.resolve())


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Base vs fine-tuned evaluation")
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    start = time.time()
    evaluate(num_samples=args.num_samples, max_new_tokens=args.max_new_tokens)
    logger.info("Done in %.1fs", time.time() - start)


if __name__ == "__main__":
    main()
