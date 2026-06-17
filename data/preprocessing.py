"""End-to-end dataset pipeline for the Healthcare QLoRA project.

Pipeline stages
---------------
1. **Load**     - download MedQuAD (HF), with a PubMedQA fallback.
2. **EDA**      - exploratory data analysis: shapes, lengths, duplicates, nulls.
3. **Clean**    - drop nulls/dupes/too-short/too-long, normalise whitespace,
                  strip HTML/boilerplate.
4. **Format**   - convert to the instruction-tuning schema
                  {"instruction", "input", "output"}.
5. **Split**    - reproducible train/val/test split.
6. **Persist**  - write JSONL files to ``data/processed/``.

Run as a script:
    python -m data.preprocessing --max-samples 5000
    python -m data.preprocessing --dataset pubmedqa

Design note: we deliberately keep the heavy ``datasets`` import inside the load
function so that importing this module for its pure-python helpers (used by the
unit tests) does not require the full ML stack.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from src.config import DataConfig, get_config
from src.utils import build_prompt, get_logger, set_seed

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Text cleaning helpers (pure functions -> trivially unit-testable).           #
# --------------------------------------------------------------------------- #
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
_MULTINEWLINE_RE = re.compile(r"\n{3,}")
# Common scraped boilerplate seen in medical web corpora.
_BOILERPLATE_RE = re.compile(
    r"(key points|summary)\s*[:\-]?\s*$", flags=re.IGNORECASE
)


def clean_text(text: str | None) -> str:
    """Normalise a raw text field.

    Steps: unescape HTML entities, strip HTML tags, collapse whitespace and
    excessive newlines, and trim. Returns "" for null-ish inputs so downstream
    length filters can drop them.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _MULTISPACE_RE.sub(" ", text)
    text = _MULTINEWLINE_RE.sub("\n\n", text)
    # Normalise spaces around newlines.
    text = "\n".join(line.strip() for line in text.split("\n"))
    return text.strip()


def is_valid_pair(
    question: str,
    answer: str,
    cfg: DataConfig,
) -> bool:
    """Return True if a (question, answer) pair passes quality thresholds."""
    if not question or not answer:
        return False
    if len(question) < cfg.min_question_chars:
        return False
    if len(answer) < cfg.min_answer_chars:
        return False
    if len(answer) > cfg.max_answer_chars:
        return False
    # Reject answers that are essentially boilerplate or echo the question.
    if answer.strip().lower() == question.strip().lower():
        return False
    return True


def to_instruction_example(
    question: str,
    answer: str,
    cfg: DataConfig,
) -> dict[str, str]:
    """Map a QA pair into the canonical instruction-tuning schema."""
    return {
        "instruction": cfg.instruction,
        "input": question.strip(),
        "output": answer.strip(),
    }


# --------------------------------------------------------------------------- #
# Stage 1: Load                                                                #
# --------------------------------------------------------------------------- #
def load_raw_dataframe(cfg: DataConfig, use_fallback: bool = False) -> pd.DataFrame:
    """Download the source dataset and return a tidy (question, answer) frame.

    We normalise every supported source to two columns: ``question`` and
    ``answer``. This decouples the rest of the pipeline from dataset-specific
    schemas.
    """
    from datasets import load_dataset  # heavy import kept local

    if not use_fallback:
        logger.info("Loading primary dataset: %s", cfg.dataset_name)
        try:
            ds = load_dataset(cfg.dataset_name, split="train")
            df = ds.to_pandas()
            df = _normalise_medquad(df, cfg)
            logger.info("Loaded %d rows from %s", len(df), cfg.dataset_name)
            return df
        except Exception as exc:  # noqa: BLE001 - we want a graceful fallback
            logger.warning(
                "Failed to load primary dataset (%s). Falling back to %s.",
                exc,
                cfg.dataset_fallback,
            )

    logger.info("Loading fallback dataset: %s (%s)", cfg.dataset_fallback, cfg.pubmedqa_config)
    ds = load_dataset(cfg.dataset_fallback, cfg.pubmedqa_config, split="train")
    df = ds.to_pandas()
    df = _normalise_pubmedqa(df)
    logger.info("Loaded %d rows from %s", len(df), cfg.dataset_fallback)
    return df


def _normalise_medquad(df: pd.DataFrame, cfg: DataConfig) -> pd.DataFrame:
    """Map MedQuAD columns to (question, answer)."""
    cols = {c.lower(): c for c in df.columns}
    q = cols.get(cfg.question_col, cols.get("question"))
    a = cols.get(cfg.answer_col, cols.get("answer"))
    if q is None or a is None:
        raise ValueError(
            f"Could not find question/answer columns in {list(df.columns)}"
        )
    return df.rename(columns={q: "question", a: "answer"})[["question", "answer"]]


def _normalise_pubmedqa(df: pd.DataFrame) -> pd.DataFrame:
    """Map PubMedQA columns to (question, answer).

    PubMedQA's ``long_answer`` is the free-text explanation we want the model to
    learn to generate (``final_decision`` is just yes/no/maybe).
    """
    answer_col = "long_answer" if "long_answer" in df.columns else "final_decision"
    out = df.rename(columns={"question": "question", answer_col: "answer"})
    return out[["question", "answer"]]


# --------------------------------------------------------------------------- #
# Stage 2: EDA                                                                 #
# --------------------------------------------------------------------------- #
def exploratory_data_analysis(df: pd.DataFrame) -> dict[str, Any]:
    """Compute and log summary statistics about the raw dataset."""
    q_len = df["question"].astype(str).str.len()
    a_len = df["answer"].astype(str).str.len()
    stats: dict[str, Any] = {
        "num_rows": int(len(df)),
        "num_duplicate_questions": int(df["question"].duplicated().sum()),
        "num_exact_duplicates": int(df.duplicated().sum()),
        "num_null_questions": int(df["question"].isna().sum()),
        "num_null_answers": int(df["answer"].isna().sum()),
        "question_len": {
            "mean": round(float(q_len.mean()), 1),
            "median": int(q_len.median()),
            "p95": int(q_len.quantile(0.95)),
            "max": int(q_len.max()),
        },
        "answer_len": {
            "mean": round(float(a_len.mean()), 1),
            "median": int(a_len.median()),
            "p95": int(a_len.quantile(0.95)),
            "max": int(a_len.max()),
        },
    }
    logger.info("EDA summary:\n%s", json.dumps(stats, indent=2))
    return stats


def save_eda_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Save question/answer length distribution plots (best-effort)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend for servers/CI
        import matplotlib.pyplot as plt

        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        df["question"].astype(str).str.len().plot.hist(
            bins=50, ax=axes[0], title="Question length (chars)"
        )
        df["answer"].astype(str).str.len().clip(upper=4000).plot.hist(
            bins=50, ax=axes[1], title="Answer length (chars, clipped@4000)"
        )
        fig.tight_layout()
        path = out_dir / "eda_length_distributions.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Saved EDA plot -> %s", path)
    except Exception as exc:  # noqa: BLE001 - plotting must never break the run
        logger.warning("Skipping EDA plots (%s)", exc)


# --------------------------------------------------------------------------- #
# Stage 3 + 4: Clean + format                                                  #
# --------------------------------------------------------------------------- #
def clean_and_format(df: pd.DataFrame, cfg: DataConfig) -> list[dict[str, str]]:
    """Apply cleaning + validation + instruction formatting; return examples."""
    initial = len(df)

    # Stage 3a: handle missing values.
    df = df.dropna(subset=["question", "answer"]).copy()
    logger.info("Dropped %d rows with null question/answer", initial - len(df))

    # Stage 3b: clean text.
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)

    # Stage 3c: remove duplicates (exact + question-level after cleaning).
    before = len(df)
    df = df.drop_duplicates(subset=["question", "answer"])
    df = df.drop_duplicates(subset=["question"], keep="first")
    logger.info("Removed %d duplicate rows", before - len(df))

    # Stage 3d + 4: validate quality and convert to instruction schema.
    examples: list[dict[str, str]] = []
    for q, a in zip(df["question"], df["answer"], strict=False):
        if is_valid_pair(q, a, cfg):
            examples.append(to_instruction_example(q, a, cfg))

    logger.info(
        "Kept %d / %d examples after quality filtering", len(examples), initial
    )
    return examples


# --------------------------------------------------------------------------- #
# Stage 5 + 6: Split + persist                                                 #
# --------------------------------------------------------------------------- #
def split_examples(
    examples: list[dict[str, str]], cfg: DataConfig
) -> dict[str, list[dict[str, str]]]:
    """Reproducibly split examples into train/val/test."""
    import random

    rng = random.Random(cfg.seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_test = int(n * cfg.test_size)
    n_val = int(n * cfg.val_size)
    test = shuffled[:n_test]
    val = shuffled[n_test : n_test + n_val]
    train = shuffled[n_test + n_val :]
    logger.info("Split -> train=%d val=%d test=%d", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a list of dicts as JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("Wrote %d rows -> %s", len(rows), path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON Lines file into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_pipeline(
    max_samples: int | None = None,
    use_fallback: bool = False,
    raw_dir: Path | None = None,
    processed_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full preprocessing pipeline and write processed splits.

    Returns a small manifest dict (counts + EDA stats) useful for logging/tests.
    """
    from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    cfg = get_config().data
    set_seed(cfg.seed)
    raw_dir = raw_dir or RAW_DATA_DIR
    processed_dir = processed_dir or PROCESSED_DATA_DIR

    df = load_raw_dataframe(cfg, use_fallback=use_fallback)
    if max_samples is not None:
        df = df.head(max_samples)

    # Persist a raw snapshot for provenance / debugging.
    raw_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(raw_dir / "medqa_raw.parquet", index=False)

    eda = exploratory_data_analysis(df)
    save_eda_plots(df, raw_dir)

    examples = clean_and_format(df, cfg)
    splits = split_examples(examples, cfg)

    write_jsonl(splits["train"], processed_dir / cfg.train_file)
    write_jsonl(splits["val"], processed_dir / cfg.val_file)
    write_jsonl(splits["test"], processed_dir / cfg.test_file)

    manifest = {
        "eda": eda,
        "counts": {k: len(v) for k, v in splits.items()},
        "total_examples": len(examples),
    }
    write_jsonl([manifest], processed_dir / "manifest.jsonl")
    return manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Healthcare QA preprocessing pipeline")
    parser.add_argument(
        "--max-samples", type=int, default=None, help="Cap raw rows (for quick runs)"
    )
    parser.add_argument(
        "--dataset",
        choices=["medquad", "pubmedqa"],
        default="medquad",
        help="Which source dataset to use",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    manifest = run_pipeline(
        max_samples=args.max_samples,
        use_fallback=(args.dataset == "pubmedqa"),
    )
    logger.info("Pipeline complete:\n%s", json.dumps(manifest["counts"], indent=2))

    # Show a formatted example so the user can sanity-check the schema.
    from src.config import PROCESSED_DATA_DIR

    sample = read_jsonl(PROCESSED_DATA_DIR / get_config().data.train_file)[:1]
    if sample:
        ex = sample[0]
        logger.info("Example instruction record:\n%s", json.dumps(ex, indent=2))
        logger.info("Rendered prompt preview:\n%s", build_prompt(ex["instruction"], ex["input"], ex["output"]))


if __name__ == "__main__":
    main()
