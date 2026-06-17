"""Inference pipeline: load base model + LoRA adapter and generate answers.

Exposes a reusable ``HealthcareAssistant`` class (used by the FastAPI service,
the Streamlit UI and the evaluation pipeline) plus a small CLI for quick manual
testing:

    python -m src.inference --question "What are the symptoms of diabetes?"
    python -m src.inference --interactive
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config import GenerationConfig, get_config, get_settings
from src.utils import build_prompt, get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = get_logger(__name__)


@dataclass
class GenerationResult:
    """Structured result returned by the assistant."""

    answer: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    model_name: str
    used_adapter: bool


class HealthcareAssistant:
    """Lazy-loading wrapper around a (base + LoRA) causal LM for QA.

    The model is loaded once on first use and cached, so the FastAPI worker pays
    the load cost at startup and serves subsequent requests cheaply.
    """

    def __init__(
        self,
        base_model: str | None = None,
        adapter_path: str | None = None,
        force_cpu: bool | None = None,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.base_model = base_model or settings.base_model
        self.adapter_path = adapter_path if adapter_path is not None else settings.adapter_path
        self.force_cpu = settings.force_cpu if force_cpu is None else force_cpu
        self.gen_config = gen_config or get_config().generation

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._used_adapter = False

    # -- lifecycle ----------------------------------------------------------- #
    def load(self) -> None:
        """Load model + tokenizer into memory (idempotent)."""
        if self._model is not None:
            return
        from src.model import load_model_for_inference

        logger.info("Loading assistant (base=%s, adapter=%s)", self.base_model, self.adapter_path)
        self._model, self._tokenizer = load_model_for_inference(
            base_model=self.base_model,
            adapter_path=self.adapter_path,
            force_cpu=self.force_cpu,
        )
        # Track whether the adapter actually loaded (vs base-only fallback).
        import os

        self._used_adapter = bool(self.adapter_path and os.path.isdir(self.adapter_path))

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # -- generation ---------------------------------------------------------- #
    def generate(
        self,
        question: str,
        instruction: str | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> GenerationResult:
        """Generate an answer for a single medical question."""
        import torch

        self.load()
        assert self._model is not None and self._tokenizer is not None

        instruction = instruction or get_config().data.instruction
        gc = self.gen_config
        prompt = build_prompt(instruction, question, output=None, tokenizer=self._tokenizer)

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_tokens = int(inputs["input_ids"].shape[-1])

        start = time.perf_counter()
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or gc.max_new_tokens,
                temperature=temperature if temperature is not None else gc.temperature,
                top_p=top_p if top_p is not None else gc.top_p,
                top_k=top_k if top_k is not None else gc.top_k,
                repetition_penalty=gc.repetition_penalty,
                do_sample=gc.do_sample,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Decode only the newly generated tokens (strip the prompt).
        new_tokens = output_ids[0][input_tokens:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        return GenerationResult(
            answer=answer,
            latency_ms=round(latency_ms, 1),
            input_tokens=input_tokens,
            output_tokens=int(new_tokens.shape[-1]),
            model_name=self.base_model,
            used_adapter=self._used_adapter,
        )


# Module-level singleton so the API/UI reuse one loaded model.
_assistant: HealthcareAssistant | None = None


def get_assistant() -> HealthcareAssistant:
    """Return a process-wide cached assistant instance."""
    global _assistant
    if _assistant is None:
        _assistant = HealthcareAssistant()
    return _assistant


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Healthcare LLM inference")
    p.add_argument("--question", type=str, help="A single question to answer")
    p.add_argument("--interactive", action="store_true", help="Interactive REPL")
    p.add_argument("--adapter", type=str, default=None, help="Override adapter path")
    p.add_argument("--base-only", action="store_true", help="Ignore adapter (base model)")
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    adapter = None if args.base_only else args.adapter
    assistant = HealthcareAssistant(adapter_path=adapter) if (adapter or args.base_only) else get_assistant()

    def _ask(q: str) -> None:
        res = assistant.generate(
            q,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nAnswer ({res.latency_ms:.0f} ms, {res.output_tokens} tok, "
              f"adapter={res.used_adapter}):\n{res.answer}\n")

    if args.interactive:
        print("Healthcare assistant (type 'exit' to quit)")
        while True:
            try:
                q = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                _ask(q)
    elif args.question:
        _ask(args.question)
    else:
        print("Provide --question '...' or --interactive")


if __name__ == "__main__":
    main()
