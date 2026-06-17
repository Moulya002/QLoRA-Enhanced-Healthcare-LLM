"""Pydantic request/response schemas for the FastAPI service.

Keeping schemas in a dedicated module (separate from the route handlers) keeps
the API contract explicit and makes it trivial to share these models with
clients or generate OpenAPI docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request body for ``POST /generate``."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The medical question to answer.",
        examples=["What are the symptoms of diabetes?"],
    )
    # Optional decoding overrides. None -> fall back to server defaults.
    max_new_tokens: int | None = Field(
        default=None, ge=16, le=2048, description="Max tokens to generate."
    )
    temperature: float | None = Field(
        default=None, ge=0.0, le=2.0, description="Sampling temperature."
    )
    top_p: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Nucleus sampling probability."
    )


class GenerateResponse(BaseModel):
    """Response body for ``POST /generate``."""

    answer: str = Field(..., description="The model's generated answer.")
    latency_ms: float = Field(..., description="Server-side generation latency.")
    input_tokens: int
    output_tokens: int
    model_name: str
    used_adapter: bool = Field(
        ..., description="True if the fine-tuned LoRA adapter was used."
    )


class HealthResponse(BaseModel):
    """Response body for ``GET /health``."""

    status: str
    model_loaded: bool
    base_model: str
    adapter_path: str
    device: str


class ModelInfoResponse(BaseModel):
    """Response body for ``GET /model-info`` (consumed by the Streamlit UI)."""

    base_model: str
    adapter_path: str
    used_adapter: bool
    device: str
    max_new_tokens: int
    temperature: float
    top_p: float


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
