"""Pre-download (cache) the base model + tokenizer.

Useful for warming the Hugging Face cache before training or before building a
Docker image, so the (slow) download isn't on the critical path of a training run
or a container's first request.

    python scripts/download_base_model.py
    python scripts/download_base_model.py --model microsoft/Phi-3-mini-4k-instruct
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a standalone script (without installing the package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_settings  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download base model + tokenizer")
    parser.add_argument("--model", default=None, help="HF model id (default: BASE_MODEL)")
    args = parser.parse_args()

    model_id = args.model or get_settings().base_model
    logger.info("Caching tokenizer + model weights for: %s", model_id)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Download weights only (don't materialize on GPU); low_cpu_mem_usage avoids
    # allocating the full model in RAM just to populate the cache.
    AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, low_cpu_mem_usage=True
    )
    logger.info("Done. Model is cached under the Hugging Face cache directory.")


if __name__ == "__main__":
    main()
