# Architecture & Design Decisions

This document records the *why* behind the major engineering choices. The
top-level `README.md` covers the *what* and *how to run*.

## 1. Configuration strategy: env settings + frozen dataclasses

Two kinds of configuration with different lifecycles:

- **Runtime/secret config** (`Settings`, `pydantic-settings`): model id, tokens,
  serving knobs. These vary per environment (laptop, CI, prod) and must never be
  committed → loaded from `.env`.
- **Experiment hyper-parameters** (`@dataclass(frozen=True)`): LoRA rank, LR,
  batch size, etc. These define *what a run is* and belong in version control so
  every result is reproducible and diff-able.

Both are cached singletons (`get_settings`, `get_config`) so the environment is
read once and configs are immutable.

## 2. One prompt builder, used everywhere

`src/utils.build_prompt` is the single source of truth for prompt formatting and
is called by training, inference, and evaluation. When a tokenizer with a
`chat_template` is available it delegates to `apply_chat_template`, guaranteeing
train/serve formatting parity (the #1 cause of silent quality regressions in LLM
apps). A plain-text Alpaca-style fallback keeps the pipeline usable in tests.

## 3. Quantization + LoRA isolation in `src/model.py`

All decisions about *how the model is loaded* (4-bit NF4, double quant, compute
dtype, LoRA target modules, k-bit prep) live in one module. Training, eval, and
serving call the same constructors, so the served model is byte-for-byte the one
that was trained/evaluated.

Graceful degradation: if CUDA is absent, `build_bnb_config` returns `None` and the
model loads in fp16/fp32, so the API and CLI work on a CPU-only laptop (just
slower) without code changes.

## 4. Thin-client UI, single model service

The model is expensive to load and hold in memory. We load it **once** behind a
FastAPI service (during lifespan startup) and make the Streamlit UI a stateless
HTTP client. This lets us scale UI and model independently and keeps the frontend
image tiny (no torch).

## 5. Tracking behind a façade

All Weights & Biases calls go through `experiments/tracking.py`. Training code
never imports `wandb` directly, so tracking can be disabled (`WANDB_MODE=disabled`)
or swapped for another backend without touching the training loop, and a missing
W&B install can never crash a long job.

## 6. Failure modes are anticipated

- Dataset download fails → fall back to PubMedQA.
- No GPU → skip 4-bit, run fp16/fp32 inference.
- No `bf16` support → fall back to fp16.
- Adapter missing → serve base model and report `used_adapter=false`.
- Metric library missing → that metric returns `{}` instead of crashing.
- Model fails to load at API startup → `/health` reports it; `/generate` → 503.

These make the system robust in heterogeneous environments (CI, laptop, GPU box,
container) — a hallmark of production code.
