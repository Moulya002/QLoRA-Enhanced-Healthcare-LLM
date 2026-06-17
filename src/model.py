"""Model + tokenizer construction for QLoRA fine-tuning and inference.

This module owns every decision about *how the model is loaded*:
  * 4-bit NF4 quantization (BitsAndBytes) for the frozen base.
  * LoRA adapter injection on attention + MLP projections (PEFT).
  * Tokenizer setup (pad token, padding side, chat template).

Keeping all of this in one place means training, evaluation and serving load
the model identically -- the #1 source of "works in training, broken in
production" bugs in LLM projects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import LoRAConfig, QuantConfig, get_config
from src.utils import get_logger, supports_bf16

if TYPE_CHECKING:  # avoid importing torch/transformers at module import time
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizer

logger = get_logger(__name__)


def _resolve_compute_dtype(name: str) -> torch.dtype:
    """Map a config string to a torch dtype, with a safe bf16->fp16 fallback."""
    import torch

    if name == "bfloat16":
        return torch.bfloat16 if supports_bf16() else torch.float16
    if name == "float16":
        return torch.float16
    return torch.float32


def build_bnb_config(quant: QuantConfig | None = None):
    """Build a ``BitsAndBytesConfig`` for 4-bit NF4 + double quantization.

    Returns ``None`` when CUDA is unavailable so callers can transparently fall
    back to full-precision loading (e.g. CPU inference on a laptop).
    """
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        logger.warning(
            "CUDA not available -> skipping 4-bit quantization (CPU/MPS fallback). "
            "QLoRA *training* requires a CUDA GPU; inference will run in fp32/fp16."
        )
        return None

    quant = quant or get_config().quant
    return BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=quant.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=_resolve_compute_dtype(quant.bnb_4bit_compute_dtype),
    )


def load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """Load a tokenizer configured for causal-LM SFT.

    * ``pad_token`` defaults to ``eos_token`` for models that lack one (Llama).
    * ``padding_side="right"`` for training (so loss aligns with labels);
      inference code switches to left padding where needed for batched generate.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer had no pad_token; set pad_token = eos_token")
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(
    model_name: str,
    quant: QuantConfig | None = None,
    for_training: bool = True,
) -> PreTrainedModel:
    """Load the base causal LM, quantized to 4-bit when a GPU is present."""
    import torch
    from transformers import AutoModelForCausalLM

    bnb_config = build_bnb_config(quant)
    compute_dtype = _resolve_compute_dtype(
        (quant or get_config().quant).bnb_4bit_compute_dtype
    )

    # On CPU we must load in float32: float16 matmuls are largely unimplemented
    # on CPU and bfloat16 generation is extremely slow. float32 is the only
    # reliable dtype for CPU inference (used by the local/Docker demo).
    if not torch.cuda.is_available():
        compute_dtype = torch.float32

    logger.info("Loading base model: %s (4-bit=%s)", model_name, bnb_config is not None)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        attn_implementation="eager",  # most portable; FA2 optional if installed
    )

    # Disable KV cache during training (incompatible with gradient checkpointing);
    # re-enable for inference to speed up generation.
    model.config.use_cache = not for_training
    return model


def attach_lora_adapters(
    model: PreTrainedModel,
    lora: LoRAConfig | None = None,
    use_gradient_checkpointing: bool = True,
) -> PreTrainedModel:
    """Prepare a 4-bit model for training and wrap it with LoRA adapters."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora = lora or get_config().lora

    # k-bit prep: casts layernorms to fp32, enables input grads, etc. -- the
    # standard QLoRA recipe for numerically stable 4-bit training.
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=use_gradient_checkpointing
    )

    peft_config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias=lora.bias,
        task_type=lora.task_type,
        target_modules=list(lora.target_modules),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_model_for_inference(
    base_model: str,
    adapter_path: str | None = None,
    force_cpu: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model (+ optional LoRA adapter) ready for generation.

    Used by both the inference script and the FastAPI service. When
    ``adapter_path`` is provided and exists, the LoRA weights are merged-in via
    PEFT for the fine-tuned behaviour; otherwise the raw base model is returned
    (handy for base-vs-finetuned evaluation).
    """
    import os

    from peft import PeftModel

    tokenizer = load_tokenizer(base_model)
    model = load_base_model(base_model, for_training=False)

    if adapter_path and os.path.isdir(adapter_path):
        logger.info("Loading LoRA adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
    elif adapter_path:
        logger.warning(
            "Adapter path %s not found -> serving the BASE model only.", adapter_path
        )

    if force_cpu:
        model = model.to("cpu")
    return model, tokenizer
