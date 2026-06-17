"""Unit tests for the FastAPI request/response schemas (validation rules)."""

from __future__ import annotations

import pytest
from api.schemas import GenerateRequest, GenerateResponse
from pydantic import ValidationError


def test_generate_request_minimal_valid():
    req = GenerateRequest(question="What are symptoms of diabetes?")
    assert req.max_new_tokens is None  # falls back to server default
    assert req.temperature is None


def test_generate_request_rejects_too_short_question():
    with pytest.raises(ValidationError):
        GenerateRequest(question="hi")


def test_generate_request_rejects_out_of_range_temperature():
    with pytest.raises(ValidationError):
        GenerateRequest(question="A valid question?", temperature=5.0)


def test_generate_request_accepts_decoding_overrides():
    req = GenerateRequest(
        question="A valid question?", max_new_tokens=128, temperature=0.5, top_p=0.8
    )
    assert req.max_new_tokens == 128
    assert req.temperature == 0.5


def test_generate_response_roundtrip():
    resp = GenerateResponse(
        answer="ans",
        latency_ms=12.3,
        input_tokens=5,
        output_tokens=10,
        model_name="m",
        used_adapter=True,
    )
    assert resp.model_dump()["used_adapter"] is True
