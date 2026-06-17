"""Unit tests for configuration objects."""

from __future__ import annotations

from src.config import (
    GenerationConfig,
    LoRAConfig,
    ProjectConfig,
    QuantConfig,
    TrainingConfig,
    get_config,
)


def test_get_config_returns_project_config():
    cfg = get_config()
    assert isinstance(cfg, ProjectConfig)
    assert isinstance(cfg.lora, LoRAConfig)
    assert isinstance(cfg.quant, QuantConfig)
    assert isinstance(cfg.training, TrainingConfig)
    assert isinstance(cfg.generation, GenerationConfig)


def test_get_config_is_cached_singleton():
    assert get_config() is get_config()


def test_lora_alpha_is_twice_rank_convention():
    lora = LoRAConfig()
    assert lora.lora_alpha == 2 * lora.r


def test_quant_uses_nf4_double_quant():
    q = QuantConfig()
    assert q.load_in_4bit is True
    assert q.bnb_4bit_quant_type == "nf4"
    assert q.bnb_4bit_use_double_quant is True


def test_target_modules_cover_both_architectures():
    # Must include both split (Llama) and fused (Phi-3) projection names.
    tm = set(LoRAConfig().target_modules)
    assert {"q_proj", "v_proj"} <= tm  # Llama-style
    assert {"qkv_proj", "gate_up_proj"} <= tm  # Phi-3-style


def test_effective_batch_size():
    t = TrainingConfig()
    effective = t.per_device_train_batch_size * t.gradient_accumulation_steps
    assert effective == 16
