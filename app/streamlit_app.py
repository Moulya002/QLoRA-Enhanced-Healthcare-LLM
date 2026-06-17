"""Streamlit chat UI for the Healthcare QLoRA assistant.

This is a thin client: it talks to the FastAPI backend over HTTP rather than
loading the model itself. That separation means the (heavy) model lives in one
place and the UI stays lightweight and horizontally scalable.

Run:
    streamlit run app/streamlit_app.py

Configuration:
    API_URL env var points at the backend (default http://localhost:8000).
"""

from __future__ import annotations

import os
import time

import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = float(os.getenv("UI_REQUEST_TIMEOUT", "120"))

st.set_page_config(
    page_title="Healthcare QLoRA Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Backend helpers                                                              #
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=30)
def fetch_model_info() -> dict | None:
    """Fetch model metadata from the backend (cached briefly)."""
    try:
        resp = httpx.get(f"{API_URL}/model-info", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:  # noqa: BLE001 - UI must stay responsive if API is down
        return None


def call_generate(question: str, max_new_tokens: int, temperature: float, top_p: float) -> dict:
    """Call POST /generate and return the parsed JSON response."""
    payload = {
        "question": question,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    resp = httpx.post(f"{API_URL}/generate", json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------------- #
# Sidebar — model info + generation controls                                   #
# --------------------------------------------------------------------------- #
def render_sidebar() -> dict:
    st.sidebar.header("⚙️ Settings")

    info = fetch_model_info()
    st.sidebar.subheader("Model")
    if info:
        st.sidebar.success("Backend connected")
        st.sidebar.markdown(
            f"- **Base model:** `{info['base_model']}`\n"
            f"- **Adapter active:** `{info['used_adapter']}`\n"
            f"- **Device:** `{info['device']}`"
        )
    else:
        st.sidebar.error(f"Cannot reach API at {API_URL}")

    st.sidebar.subheader("Decoding")
    max_new_tokens = st.sidebar.slider("Max new tokens", 64, 1024, 512, 64)
    temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, 0.05)

    if st.sidebar.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.caption(
        "This assistant is for informational purposes only and is not a "
        "substitute for professional medical advice."
    )
    return {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}


# --------------------------------------------------------------------------- #
# Main chat interface                                                          #
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🩺 Healthcare QLoRA Assistant")
    st.caption("A domain-specific medical QA assistant fine-tuned with QLoRA.")

    params = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay conversation history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    prompt = st.chat_input("Ask a medical question…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            start = time.time()
            try:
                data = call_generate(
                    prompt,
                    params["max_new_tokens"],
                    params["temperature"],
                    params["top_p"],
                )
                elapsed = time.time() - start
                answer = data["answer"]
                meta = (
                    f"⏱️ {data.get('latency_ms', elapsed * 1000):.0f} ms · "
                    f"{data.get('output_tokens', '?')} tokens · "
                    f"adapter={data.get('used_adapter')}"
                )
            except httpx.HTTPStatusError as exc:
                answer = f"⚠️ Backend error ({exc.response.status_code}): {exc.response.text}"
                meta = ""
            except Exception as exc:  # noqa: BLE001
                answer = f"⚠️ Could not reach the backend at {API_URL}: {exc}"
                meta = ""

            st.markdown(answer)
            if meta:
                st.caption(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )


if __name__ == "__main__":
    main()
