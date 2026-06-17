"""Cross-cutting utilities: logging, seeding, device selection, prompts.

Kept dependency-light so it can be imported by data scripts, the API and the UI
without pulling in heavy training-only packages at import time.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Any

# --------------------------------------------------------------------------- #
# Logging                                                                      #
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured module logger.

    We configure a single stream handler and disable propagation to avoid the
    classic "every log line printed twice" problem when libraries also touch the
    root logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # already configured
        return logger

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    logger.addHandler(handler)
    logger.setLevel(log_level)
    logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Reproducibility                                                              #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Seed all RNGs (python, numpy, torch) for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy optional in minimal environments
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Device / hardware                                                            #
# --------------------------------------------------------------------------- #
def get_device(force_cpu: bool = False) -> str:
    """Return the best available torch device string: cuda | mps | cpu."""
    try:
        import torch
    except ImportError:
        return "cpu"

    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU. Note: bitsandbytes 4-bit is NOT supported on MPS, so
    # training still requires CUDA; MPS is only useful for fp16/fp32 inference.
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def supports_bf16() -> bool:
    """True if the current CUDA device supports bfloat16 (Ampere+ / SM80+)."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except ImportError:
        return False


def gpu_memory_stats() -> dict[str, float]:
    """Return current GPU memory usage in GB (zeros if no CUDA)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    except ImportError:
        return {"allocated_gb": 0.0, "reserved_gb": 0.0, "max_allocated_gb": 0.0}


def count_trainable_parameters(model: Any) -> dict[str, Any]:
    """Return trainable vs total parameter counts and the trainable percentage."""
    trainable, total = 0, 0
    for param in model.parameters():
        n = param.numel()
        total += n
        if param.requires_grad:
            trainable += n
    pct = 100.0 * trainable / total if total else 0.0
    return {"trainable": trainable, "total": total, "trainable_pct": round(pct, 4)}


# --------------------------------------------------------------------------- #
# Timing                                                                       #
# --------------------------------------------------------------------------- #
@contextmanager
def timer(label: str = "block", logger: logging.Logger | None = None):
    """Context manager that measures wall-clock time of a code block."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    msg = f"[timer] {label}: {elapsed:.3f}s"
    (logger.info(msg) if logger else print(msg))


# --------------------------------------------------------------------------- #
# Prompt formatting                                                            #
# --------------------------------------------------------------------------- #
def build_prompt(
    instruction: str,
    user_input: str,
    output: str | None = None,
    tokenizer: Any | None = None,
) -> str:
    """Render an (instruction, input, output) triple into a chat-formatted string.

    Why this matters: instruct models are extremely sensitive to their chat
    template (special tokens like ``<|user|>``). When a tokenizer is provided we
    delegate to its official ``apply_chat_template`` so train-time and
    inference-time formatting are identical. A plain-text fallback keeps the data
    pipeline usable without a tokenizer (e.g. in unit tests / EDA).

    During training we append ``output`` so the model learns to produce it.
    During inference we pass ``output=None`` to get the generation prompt.
    """
    system_and_user = f"{instruction}\n\n{user_input}".strip()

    if tokenizer is not None and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": system_and_user}]
        if output is None:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        messages.append({"role": "assistant", "content": output})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    # Plain-text fallback (Alpaca-style).
    prompt = (
        f"### Instruction:\n{instruction}\n\n"
        f"### Input:\n{user_input}\n\n"
        f"### Response:\n"
    )
    if output is not None:
        prompt += output
    return prompt
