"""Unit tests for utility helpers."""

from __future__ import annotations

import logging

from src.utils import build_prompt, get_logger, set_seed


def test_get_logger_no_duplicate_handlers():
    log = get_logger("test.logger")
    log2 = get_logger("test.logger")
    assert log is log2
    assert len(log.handlers) == 1
    assert isinstance(log, logging.Logger)


def test_set_seed_is_deterministic():
    import random

    set_seed(123)
    a = [random.random() for _ in range(3)]
    set_seed(123)
    b = [random.random() for _ in range(3)]
    assert a == b


def test_build_prompt_plaintext_training_includes_output():
    p = build_prompt("Answer the question.", "What is X?", output="X is a thing.")
    assert "### Instruction:" in p
    assert "What is X?" in p
    assert p.endswith("X is a thing.")


def test_build_prompt_plaintext_inference_omits_output():
    p = build_prompt("Answer the question.", "What is X?", output=None)
    assert p.strip().endswith("### Response:")


def test_build_prompt_uses_chat_template_when_available():
    class FakeTokenizer:
        chat_template = "exists"

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            roles = "|".join(m["role"] for m in messages)
            return f"TEMPLATED[{roles}|gen={add_generation_prompt}]"

    tok = FakeTokenizer()
    inf = build_prompt("ins", "q", output=None, tokenizer=tok)
    assert inf == "TEMPLATED[user|gen=True]"
    train = build_prompt("ins", "q", output="a", tokenizer=tok)
    assert train == "TEMPLATED[user|assistant|gen=False]"
