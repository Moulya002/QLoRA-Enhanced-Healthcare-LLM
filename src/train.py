"""QLoRA supervised fine-tuning entry point (TRL ``SFTTrainer``).

Usage
-----
    # full run (reads .env for BASE_MODEL / W&B)
    python -m src.train

    # quick smoke run on a tiny subset
    python -m src.train --max-train-samples 200 --epochs 1 --no-wandb

What this script does
---------------------
1. Loads the processed instruction dataset (data/processed/*.jsonl).
2. Loads the base model 4-bit quantized + attaches LoRA adapters (QLoRA).
3. Configures ``SFTTrainer`` with mixed precision, gradient accumulation,
   gradient checkpointing and a paged 8-bit optimizer.
4. Trains, evaluates, and saves the LoRA adapter + tokenizer.

Every hyper-parameter comes from ``src.config`` so runs are reproducible.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    TrainingConfig,
    ensure_dirs,
    get_config,
    get_settings,
)
from src.utils import get_logger, gpu_memory_stats, set_seed

logger = get_logger(__name__)


def _format_dataset(dataset, tokenizer, instruction_field: str = "instruction"):
    """Render each example into a single ``text`` field via the chat template.

    SFTTrainer trains on a plain text column; we precompute it so train-time and
    inference-time prompt formatting are byte-for-byte identical.
    """
    from src.utils import build_prompt

    def _render(example):
        text = build_prompt(
            example["instruction"],
            example["input"],
            example["output"],
            tokenizer=tokenizer,
        )
        return {"text": text}

    return dataset.map(_render, remove_columns=dataset.column_names)


def _load_splits(max_train: int | None, max_eval: int | None):
    """Load train/val JSONL splits as a ``DatasetDict``."""
    from datasets import load_dataset

    cfg = get_config().data
    train_path = PROCESSED_DATA_DIR / cfg.train_file
    val_path = PROCESSED_DATA_DIR / cfg.val_file
    if not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run `python -m data.preprocessing` first."
        )

    data_files = {"train": str(train_path), "validation": str(val_path)}
    ds = load_dataset("json", data_files=data_files)
    if max_train:
        ds["train"] = ds["train"].select(range(min(max_train, len(ds["train"]))))
    if max_eval:
        ds["validation"] = ds["validation"].select(
            range(min(max_eval, len(ds["validation"])))
        )
    logger.info(
        "Dataset loaded: train=%d val=%d", len(ds["train"]), len(ds["validation"])
    )
    return ds


def _build_sft_config(tcfg: TrainingConfig, use_wandb: bool):
    """Build a TRL ``SFTConfig`` (extends HF ``TrainingArguments``)."""
    from trl import SFTConfig

    from src.utils import supports_bf16

    # Choose mixed precision based on hardware: bf16 on Ampere+, else fp16.
    use_bf16 = tcfg.bf16 and supports_bf16()
    use_fp16 = tcfg.fp16 or (not use_bf16)

    return SFTConfig(
        output_dir=tcfg.output_dir,
        seed=tcfg.seed,
        per_device_train_batch_size=tcfg.per_device_train_batch_size,
        per_device_eval_batch_size=tcfg.per_device_eval_batch_size,
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        num_train_epochs=tcfg.num_train_epochs,
        learning_rate=tcfg.learning_rate,
        lr_scheduler_type=tcfg.lr_scheduler_type,
        warmup_ratio=tcfg.warmup_ratio,
        weight_decay=tcfg.weight_decay,
        max_grad_norm=tcfg.max_grad_norm,
        optim=tcfg.optim,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=tcfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=tcfg.max_seq_length,
        packing=tcfg.packing,
        logging_steps=tcfg.logging_steps,
        eval_strategy=tcfg.eval_strategy,
        eval_steps=tcfg.eval_steps,
        save_steps=tcfg.save_steps,
        save_total_limit=tcfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=("wandb" if use_wandb else "none"),
        dataset_text_field="text",
    )


def train(
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    epochs: int | None = None,
    use_wandb: bool = True,
) -> str:
    """Run QLoRA fine-tuning. Returns the path to the saved adapter."""
    from experiments.tracking import finish_run, init_wandb
    from trl import SFTTrainer

    from src.model import attach_lora_adapters, load_base_model, load_tokenizer

    cfg = get_config()
    settings = get_settings()
    ensure_dirs()
    set_seed(cfg.training.seed)

    # Allow CLI overrides without mutating the frozen config.
    tcfg = cfg.training
    if epochs is not None:
        tcfg = replace(tcfg, num_train_epochs=epochs)
    max_train = max_train_samples if max_train_samples is not None else tcfg.max_train_samples
    max_eval = max_eval_samples if max_eval_samples is not None else tcfg.max_eval_samples

    # --- Experiment tracking ------------------------------------------------ #
    if use_wandb:
        init_wandb(config_obj=cfg, settings=settings)

    # --- Data --------------------------------------------------------------- #
    tokenizer = load_tokenizer(settings.base_model)
    raw = _load_splits(max_train, max_eval)
    train_ds = _format_dataset(raw["train"], tokenizer)
    eval_ds = _format_dataset(raw["validation"], tokenizer)

    # --- Model (QLoRA) ------------------------------------------------------ #
    model = load_base_model(settings.base_model, quant=cfg.quant, for_training=True)
    model = attach_lora_adapters(
        model, lora=cfg.lora, use_gradient_checkpointing=tcfg.gradient_checkpointing
    )

    # --- Trainer ------------------------------------------------------------ #
    sft_config = _build_sft_config(tcfg, use_wandb)
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    logger.info("Starting training. GPU mem before: %s", gpu_memory_stats())
    start = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start

    # --- Persist ------------------------------------------------------------ #
    out_dir = Path(tcfg.output_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = train_result.metrics
    metrics["training_time_sec"] = round(elapsed, 2)
    metrics.update({f"gpu_{k}": v for k, v in gpu_memory_stats().items()})
    trainer.save_metrics("train", metrics)
    logger.info("Training complete in %.1fs. Adapter saved -> %s", elapsed, out_dir)
    logger.info("Final metrics: %s", metrics)

    if use_wandb:
        finish_run(summary=metrics)

    return str(out_dir)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="QLoRA SFT training for Healthcare LLM")
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-eval-samples", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    train(
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        epochs=args.epochs,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
