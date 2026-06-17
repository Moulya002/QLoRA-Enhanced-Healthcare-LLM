"""Unit tests for the data preprocessing pipeline (pure functions only)."""

from __future__ import annotations

import pandas as pd
from data.preprocessing import (
    clean_and_format,
    clean_text,
    is_valid_pair,
    read_jsonl,
    split_examples,
    to_instruction_example,
    write_jsonl,
)
from src.config import DataConfig


def test_clean_text_strips_html_and_whitespace():
    raw = "<p>Hello&amp;   world</p>\n\n\n\nDone"
    assert clean_text(raw) == "Hello& world\n\nDone"


def test_clean_text_handles_none_and_nonstring():
    assert clean_text(None) == ""
    assert clean_text(123) == "123"


def test_is_valid_pair_rejects_short_and_empty():
    cfg = DataConfig()
    assert not is_valid_pair("", "some long answer here", cfg)
    assert not is_valid_pair("short", "x", cfg)  # answer too short
    assert is_valid_pair(
        "What are the symptoms of diabetes?",
        "Increased thirst, frequent urination, and fatigue are common symptoms.",
        cfg,
    )


def test_is_valid_pair_rejects_echo_answer():
    cfg = DataConfig()
    q = "What is hypertension exactly?"
    assert not is_valid_pair(q, q, cfg)


def test_to_instruction_example_schema():
    cfg = DataConfig()
    ex = to_instruction_example("Q?", "A.", cfg)
    assert set(ex.keys()) == {"instruction", "input", "output"}
    assert ex["input"] == "Q?"
    assert ex["output"] == "A."
    assert ex["instruction"] == cfg.instruction


def test_clean_and_format_dedupes_and_filters(sample_qa_pairs):
    cfg = DataConfig()
    df = pd.DataFrame(sample_qa_pairs)
    examples = clean_and_format(df, cfg)
    # Only the first valid pair survives (dup question + invalid rows removed).
    assert len(examples) == 1
    assert examples[0]["input"] == "What are the symptoms of diabetes?"


def test_split_examples_is_reproducible_and_partitions():
    cfg = DataConfig()
    examples = [{"instruction": "i", "input": f"q{i}", "output": f"a{i}"} for i in range(100)]
    s1 = split_examples(examples, cfg)
    s2 = split_examples(examples, cfg)
    # Reproducible given the fixed seed.
    assert [e["input"] for e in s1["train"]] == [e["input"] for e in s2["train"]]
    # Partition covers all examples with no overlap.
    total = len(s1["train"]) + len(s1["val"]) + len(s1["test"])
    assert total == 100
    inputs = {e["input"] for e in s1["train"] + s1["val"] + s1["test"]}
    assert len(inputs) == 100


def test_write_and_read_jsonl_roundtrip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "out.jsonl"
    write_jsonl(rows, path)
    assert read_jsonl(path) == rows
