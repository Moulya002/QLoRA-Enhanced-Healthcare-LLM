"""Pytest fixtures + path setup.

Adds the repo root to ``sys.path`` so ``import src...`` works without installing
the package, and provides small reusable fixtures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_qa_pairs() -> list[dict[str, str]]:
    """A few representative (question, answer) pairs for pipeline tests."""
    return [
        {
            "question": "What are the symptoms of diabetes?",
            "answer": "Common symptoms include increased thirst, frequent urination, "
            "fatigue, and blurred vision.",
        },
        {
            "question": "What are the symptoms of diabetes?",  # duplicate question
            "answer": "Increased thirst and frequent urination are common signs.",
        },
        {"question": "  ", "answer": "too short question"},  # invalid
        {"question": "What is hypertension?", "answer": "x"},  # answer too short
    ]
