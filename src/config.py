"""Centralised configuration for the Healthcare QLoRA project.

Design decisions
----------------
* We use ``pydantic-settings`` for the *environment-driven* runtime config
  (secrets, model id, serving knobs). This gives us validation + ``.env``
  loading for free and a single source of truth shared by training, the API
  and the UI.
* We use frozen ``dataclasses`` for the *experiment hyper-parameters*
  (LoRA / training / generation). Hyper-parameters belong in code and version
  control so every run is reproducible and diff-able, rather than hidden in
  environment variables.
* Every field has an inline comment explaining *why* the default was chosen,
  because the most valuable thing in an ML repo is knowing the reasoning behind
  each number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Project paths (resolved relative to the repo root, not the CWD).            #
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
LOGS_DIR: Path = PROJECT_ROOT / "logs"
EVAL_DIR: Path = PROJECT_ROOT / "evaluation"


# --------------------------------------------------------------------------- #
# Runtime settings (from environment / .env).                                 #
# --------------------------------------------------------------------------- #
class Settings(BaseSettings):
    """Environment-driven runtime settings (secrets + serving knobs)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets / accounts
    hf_token: str | None = None
    wandb_api_key: str | None = None
    wandb_project: str = "healthcare-qlora"
    wandb_entity: str | None = None
    # online | offline | disabled  -> controls whether W&B talks to the cloud.
    wandb_mode: str = "online"

    # Model selection. Phi-3-mini is the default because it is *not gated* and
    # therefore works out of the box without HF license acceptance, while still
    # being a strong 3.8B instruct model that fits in <8GB VRAM at 4-bit.
    base_model: str = "microsoft/Phi-3-mini-4k-instruct"
    adapter_path: str = "outputs/healthcare-qlora-adapter"

    # Default generation knobs for serving (overridable per request).
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9

    # API / UI wiring
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://localhost:8000"

    # Misc
    log_level: str = "INFO"
    force_cpu: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (read the env only once)."""
    return Settings()


# --------------------------------------------------------------------------- #
# Dataset configuration.                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DataConfig:
    """Dataset download + preprocessing configuration."""

    # MedQuAD-style medical QA. We use a public HF mirror of MedQuAD that ships
    # ready as a `datasets` load, with a documented fallback to PubMedQA.
    dataset_name: str = "lavita/MedQuAD"
    dataset_fallback: str = "pubmed_qa"  # config: "pqa_labeled"
    pubmedqa_config: str = "pqa_labeled"

    # Column names in the source dataset.
    question_col: str = "question"
    answer_col: str = "answer"

    # Fixed instruction prepended to every example (instruction-tuning format).
    instruction: str = (
        "You are a knowledgeable medical assistant. Answer the patient's health "
        "question accurately, clearly, and safely. If the question requires "
        "professional diagnosis, advise consulting a healthcare provider."
    )

    # Cleaning thresholds.
    min_question_chars: int = 12
    min_answer_chars: int = 20
    max_answer_chars: int = 4000  # drop pathological/scraped mega-answers

    # Reproducible train/val/test split.
    test_size: float = 0.10
    val_size: float = 0.10  # taken from the remaining train pool
    seed: int = 42

    # Output filenames (written to data/processed/).
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"


# --------------------------------------------------------------------------- #
# LoRA / QLoRA configuration.                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter hyper-parameters.

    LoRA freezes the base weights and injects small trainable low-rank matrices
    into chosen linear layers, training <1% of parameters. Combined with a
    4-bit frozen base (QLoRA) this lets us fine-tune a 3B model on a single
    consumer GPU.
    """

    # Rank of the low-rank update. 16 is a strong default: higher r = more
    # capacity but more memory; 8-32 is the sweet spot for instruction tuning.
    r: int = 16
    # Scaling factor. Convention alpha = 2*r keeps the effective LR stable.
    lora_alpha: int = 32
    # Regularisation on the adapter to reduce overfitting on a small dataset.
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # Target the attention projections (and MLP) where adaptation matters most.
    # These names cover both Llama-3.2 and Phi-3 architectures.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Phi-3 fuses qkv/gate-up; include their names so PEFT matches either arch.
        "qkv_proj",
        "gate_up_proj",
    )


@dataclass(frozen=True)
class QuantConfig:
    """BitsAndBytes 4-bit (NF4) quantization config for the frozen base model."""

    load_in_4bit: bool = True
    # NF4 = 4-bit NormalFloat, information-theoretically optimal for normally
    # distributed weights (the QLoRA paper's key contribution).
    bnb_4bit_quant_type: str = "nf4"
    # Double quantization quantizes the quantization constants too -> ~0.4
    # bits/param extra saving at no measurable quality cost.
    bnb_4bit_use_double_quant: bool = True
    # Compute in bfloat16: matmuls dequantize to bf16 for stable training.
    bnb_4bit_compute_dtype: str = "bfloat16"


# --------------------------------------------------------------------------- #
# Training configuration.                                                      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingConfig:
    """SFT training hyper-parameters (consumed by TRL SFTTrainer)."""

    output_dir: str = "outputs/healthcare-qlora-adapter"
    seed: int = 42

    # Effective batch size = per_device_batch * grad_accum = 16.
    # We keep per-device small to fit in VRAM and recover batch size via
    # gradient accumulation.
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    num_train_epochs: int = 3
    # Slightly higher LR is fine for LoRA since only adapters update.
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3  # gradient clipping for stability (QLoRA recipe)
    optim: str = "paged_adamw_8bit"  # paged optimizer avoids OOM spikes

    # Mixed precision. bf16 preferred on Ampere+; falls back to fp16 (handled
    # at runtime in train.py based on hardware support).
    bf16: bool = True
    fp16: bool = False

    # Memory: recompute activations in the backward pass to trade compute for
    # VRAM (essential for fitting longer sequences).
    gradient_checkpointing: bool = True

    # Sequence packing + length. Packing concatenates short QA pairs to fill the
    # context window -> far higher token throughput.
    max_seq_length: int = 1024
    packing: bool = False  # disabled so each example keeps its own loss masking

    # Logging / eval / checkpoint cadence (in optimizer steps).
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 50
    save_total_limit: int = 2
    eval_strategy: str = "steps"

    # Cap dataset size for quick smoke runs (None = use everything).
    max_train_samples: int | None = None
    max_eval_samples: int | None = 500

    report_to: str = "wandb"  # set to "none" to disable tracking


# --------------------------------------------------------------------------- #
# Generation / inference configuration.                                        #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenerationConfig:
    """Default decoding parameters for inference + evaluation."""

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True


# --------------------------------------------------------------------------- #
# Bundled config object passed around the codebase.                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProjectConfig:
    """Aggregate of all sub-configs for convenient single-import access."""

    data: DataConfig = field(default_factory=DataConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


@lru_cache
def get_config() -> ProjectConfig:
    """Return a cached, fully-populated ``ProjectConfig``."""
    return ProjectConfig()


def ensure_dirs() -> None:
    """Create the standard output directories if they don't exist."""
    for path in (RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)
