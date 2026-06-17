"""Unit tests for evaluation metrics.

These tests exercise the aggregation logic and graceful-degradation behaviour.
Metric libraries (rouge/sacrebleu/bert-score) are optional; when present we
assert sane ranges, otherwise the functions must return ``{}`` not crash.
"""

from __future__ import annotations

import pytest
from evaluation.metrics import (
    compute_all_metrics,
    compute_rouge,
    summarise_latency,
)


def test_summarise_latency_basic():
    out = summarise_latency([100.0, 200.0, 300.0])
    assert out["latency_ms_mean"] == 200.0
    assert out["latency_ms_median"] == 200.0
    assert "latency_ms_p95" in out


def test_summarise_latency_empty():
    assert summarise_latency([]) == {}


def test_compute_all_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_all_metrics(["a"], ["a", "b"])


def test_compute_all_metrics_includes_count():
    preds = ["increased thirst and frequent urination"]
    refs = ["increased thirst, frequent urination, and fatigue"]
    out = compute_all_metrics(preds, refs, latencies_ms=[123.0])
    assert out["num_examples"] == 1
    assert out["latency_ms_mean"] == 123.0


def test_compute_rouge_perfect_match_when_available():
    rouge = compute_rouge(["the cat sat"], ["the cat sat"])
    if rouge:  # only assert if rouge-score is installed
        assert rouge["rouge1"] == pytest.approx(1.0, abs=1e-6)
