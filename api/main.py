"""FastAPI service exposing the Healthcare QLoRA assistant.

Endpoints
---------
GET  /health      - liveness/readiness probe (used by Docker healthcheck).
GET  /model-info  - model + decoding configuration (used by the Streamlit UI).
POST /generate    - answer a single medical question.

Design notes
------------
* The model is loaded once during the FastAPI *lifespan* startup, not per
  request, so the (slow) load cost is paid once and every request is cheap.
* Heavy generation runs in a threadpool (``run_in_threadpool``) so a single slow
  request doesn't block the event loop / other health checks.
* A safety disclaimer is appended to every answer — non-negotiable for a
  healthcare assistant.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from src.config import get_settings
from src.utils import get_device, get_logger

from api.schemas import (
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfoResponse,
)

logger = get_logger(__name__)

_DISCLAIMER = (
    "\n\n---\n*Disclaimer: This is an AI-generated response for informational "
    "purposes only and is not a substitute for professional medical advice. "
    "Always consult a qualified healthcare provider.*"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup, release it on shutdown."""
    from src.inference import get_assistant

    settings = get_settings()
    logger.info("API starting up — loading model %s", settings.base_model)
    assistant = get_assistant()
    try:
        assistant.load()
        logger.info("Model loaded; API ready.")
    except Exception as exc:  # noqa: BLE001
        # Don't crash the whole service if weights are missing in a demo env;
        # /health will report model_loaded=False and /generate returns 503.
        logger.error("Model failed to load at startup: %s", exc)
    app.state.assistant = assistant
    yield
    logger.info("API shutting down.")


app = FastAPI(
    title="Healthcare QLoRA Assistant API",
    description="Domain-specific medical QA powered by a QLoRA fine-tuned LLM.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS so the Streamlit frontend (different origin) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """Liveness + readiness probe."""
    settings = get_settings()
    assistant = getattr(app.state, "assistant", None)
    return HealthResponse(
        status="ok",
        model_loaded=bool(assistant and assistant.is_loaded),
        base_model=settings.base_model,
        adapter_path=settings.adapter_path,
        device=get_device(settings.force_cpu),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    """Return model + decoding configuration for the UI."""
    settings = get_settings()
    assistant = getattr(app.state, "assistant", None)
    used_adapter = bool(assistant and getattr(assistant, "_used_adapter", False))
    return ModelInfoResponse(
        base_model=settings.base_model,
        adapter_path=settings.adapter_path,
        used_adapter=used_adapter,
        device=get_device(settings.force_cpu),
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest) -> GenerateResponse:
    """Answer a single medical question."""
    assistant = getattr(app.state, "assistant", None)
    if assistant is None or not assistant.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    try:
        result = await run_in_threadpool(
            assistant.generate,
            req.question,
            None,  # instruction (use default)
            req.max_new_tokens,
            req.temperature,
            req.top_p,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation error: {exc}") from exc

    return GenerateResponse(
        answer=result.answer + _DISCLAIMER,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        model_name=result.model_name,
        used_adapter=result.used_adapter,
    )


def run() -> None:
    """Entrypoint for ``python -m api.main`` (uses uvicorn programmatically)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
