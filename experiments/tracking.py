"""Weights & Biases experiment tracking.

We centralise W&B here (rather than scattering ``wandb.*`` calls across the
training code) so that:
  * the whole project can be made tracking-agnostic / disabled with one switch,
  * the exact set of hyper-parameters logged is defined in one place,
  * the API surface is small and easy to stub in tests.

TRL's ``SFTTrainer`` already streams training/validation loss, learning rate and
gradient norms to W&B automatically when ``report_to="wandb"``. This module
handles run *initialisation* (so the project/entity/config are correct) and
logging of things the trainer does not track for us: hyper-parameters, GPU
memory snapshots, total training time and final summary metrics.
"""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

from src.utils import get_logger, gpu_memory_stats

if TYPE_CHECKING:
    from src.config import ProjectConfig, Settings

logger = get_logger(__name__)

# Track whether a run is active so log/finish are safe no-ops when disabled.
_run_active = False


def _flatten_config(config_obj: ProjectConfig) -> dict[str, Any]:
    """Flatten the nested dataclass config into a single dict for W&B."""
    flat: dict[str, Any] = {}
    for section in dataclasses.fields(config_obj):
        sub = getattr(config_obj, section.name)
        if dataclasses.is_dataclass(sub):
            for f in dataclasses.fields(sub):
                flat[f"{section.name}.{f.name}"] = getattr(sub, f.name)
        else:
            flat[section.name] = sub
    return flat


def init_wandb(
    config_obj: ProjectConfig,
    settings: Settings,
    run_name: str | None = None,
    tags: list[str] | None = None,
) -> bool:
    """Initialise a W&B run. Returns True if tracking is active.

    Respects ``WANDB_MODE`` (online/offline/disabled). If the ``wandb`` package
    or API key is missing we degrade gracefully to a no-op rather than crashing
    a long training job.
    """
    global _run_active

    if settings.wandb_mode == "disabled":
        logger.info("W&B disabled via WANDB_MODE=disabled")
        return False

    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed -> experiment tracking disabled")
        return False

    # Make env-based config available to the underlying library + TRL callback.
    os.environ.setdefault("WANDB_PROJECT", settings.wandb_project)
    if settings.wandb_api_key:
        os.environ.setdefault("WANDB_API_KEY", settings.wandb_api_key)

    try:
        wandb.init(
            project=settings.wandb_project,
            entity=settings.wandb_entity,
            name=run_name,
            mode=settings.wandb_mode,
            tags=tags or ["qlora", "healthcare", settings.base_model.split("/")[-1]],
            config=_flatten_config(config_obj),
        )
        _run_active = True
        logger.info("W&B run initialised (project=%s)", settings.wandb_project)
    except Exception as exc:  # noqa: BLE001 - never let tracking kill training
        logger.warning("W&B init failed (%s) -> continuing without tracking", exc)
        _run_active = False
    return _run_active


def log_metrics(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a dict of metrics to the active run (no-op if tracking is off)."""
    if not _run_active:
        return
    try:
        import wandb

        wandb.log(metrics, step=step)
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B log failed: %s", exc)


def log_gpu_memory(step: int | None = None) -> None:
    """Snapshot and log current GPU memory usage."""
    stats = gpu_memory_stats()
    log_metrics({f"gpu/{k}": v for k, v in stats.items()}, step=step)


def finish_run(summary: dict[str, Any] | None = None) -> None:
    """Write summary metrics and close the active W&B run."""
    global _run_active
    if not _run_active:
        return
    try:
        import wandb

        if summary:
            for key, value in summary.items():
                wandb.summary[key] = value
        wandb.finish()
        logger.info("W&B run finished")
    except Exception as exc:  # noqa: BLE001
        logger.warning("W&B finish failed: %s", exc)
    finally:
        _run_active = False
