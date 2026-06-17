"""Healthcare QLoRA fine-tuning package.

Modules
-------
config     : Centralised, validated configuration (env + dataclasses).
utils      : Cross-cutting helpers (logging, seeding, device, prompts).
model      : Model + tokenizer + 4-bit quantization + LoRA construction.
train      : QLoRA SFT training entry point (TRL SFTTrainer).
inference  : Load base model + LoRA adapter and generate answers.
"""

__version__ = "0.1.0"
