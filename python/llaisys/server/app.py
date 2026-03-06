"""LLAISYS Chat Server — OpenAI-compatible FastAPI application.

Endpoints
---------
POST /v1/chat/completions   Chat completion (streaming & non-streaming)
GET  /v1/models              List available models
GET  /                       Serve Chat UI

Usage
-----
    python -m llaisys.server --model ./models/DeepSeek-R1-Distill-Qwen-1.5B
"""

import asyncio
import threading
import time
import uuid
from pathlib import Path
from typing import Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    DeltaMessage,
    Usage,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_max_seq_len = 512
_model_lock = threading.Lock()  # serialise access (single-GPU model)

app = FastAPI(title="LLAISYS Chat API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str = "nvidia", max_seq_len: int = 512,
               quantized: bool = False):
    """Load Qwen2 model + HuggingFace tokenizer into global state."""
    global _model, _tokenizer, _max_seq_len

    from llaisys import DeviceType
    from llaisys.models.qwen2 import Qwen2
    from transformers import AutoTokenizer

    device_map = {
        "cpu": DeviceType.CPU,
        "nvidia": DeviceType.NVIDIA,
        "metax": DeviceType.METAX,
        "tianshu": DeviceType.TIANSHU,
    }
    dev = device_map.get(device)
    if dev is None:
        raise ValueError(f"Unknown device '{device}'. Choose from: {list(device_map)}")

    mode = "INT8 quantized" if quantized else "FP32"
    print(f"[Server] Loading model from {model_path} on {device} ({mode}) ...")
    _model = Qwen2(model_path, device=dev, max_seq_len=max_seq_len, quantized=quantized)
    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _max_seq_len = max_seq_len
    print(f"[Server] Model ready. (max_seq_len={max_seq_len})")


def _require_model():
    if _model is None or _tokenizer is None:
        raise HTTPException(503, "Model not loaded. Start server with --model.")
    return _model, _tokenizer


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "deepseek-r1-distill-qwen-1.5b",
                "object": "model",
                "owned_by": "llaisys",
            }
        ],
    }


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model, tokenizer = _require_model()

    # Encode messages → token IDs via HuggingFace chat template
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True
    )
    seed = req.seed if req.seed is not None else 0

    # --- Guard against KV cache overflow ---
    # Reserve at least 32 tokens for generation; truncate input if needed.
    min_gen = 32
    max_input = _max_seq_len - min_gen
    if len(input_ids) > max_input:
        # Truncate from the LEFT (keep the most recent context)
        truncated = len(input_ids) - max_input
        input_ids = input_ids[-max_input:]
        print(f"[Server] Warning: Input truncated by {truncated} tokens "
              f"({len(input_ids)+truncated} → {len(input_ids)}) to fit max_seq_len={_max_seq_len}")

    # Clamp max_tokens so total doesn't exceed max_seq_len
    effective_max_tokens = min(req.max_tokens, _max_seq_len - len(input_ids))
    if effective_max_tokens < 1:
        raise HTTPException(400, f"Input too long ({len(input_ids)} tokens) for "
                            f"max_seq_len={_max_seq_len}. Try a shorter conversation.")

    if req.stream:
        # Sync generator — Starlette auto-wraps with iterate_in_threadpool
        return StreamingResponse(
            _stream_sync(model, tokenizer, input_ids, req, seed, effective_max_tokens),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ---- non-streaming ----
    result = await asyncio.to_thread(
        _generate_sync, model, tokenizer, input_ids, req, seed, effective_max_tokens
    )
    return result


# ---------------------------------------------------------------------------
# Non-streaming helper (runs in thread)
# ---------------------------------------------------------------------------

def _generate_sync(model, tokenizer, input_ids, req, seed, max_tokens):
    try:
        with _model_lock:
            model.reset()
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                seed=seed,
            )
    except Exception as e:
        print(f"[Server] Inference error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

    new_ids = output_ids[len(input_ids):]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)

    return ChatCompletionResponse(
        model=req.model,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=len(input_ids),
            completion_tokens=len(new_ids),
            total_tokens=len(input_ids) + len(new_ids),
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper (sync generator — Starlette wraps in threadpool)
# ---------------------------------------------------------------------------

def _stream_sync(
    model, tokenizer, input_ids, req, seed, max_tokens
) -> Generator[str, None, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    ts = int(time.time())
    name = req.model

    # First chunk: role announcement
    yield _sse(ChatCompletionChunk(
        id=cid, created=ts, model=name,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
    ))

    all_ids: list[int] = []
    prev_text = ""
    error_occurred = False

    try:
        with _model_lock:
            model.reset()
            for token_id in model.generate_stream(
                input_ids,
                max_new_tokens=max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature,
                seed=seed,
            ):
                if token_id == model.meta.end_token:
                    break

                all_ids.append(token_id)
                # Incremental decode — avoids BPE boundary artefacts
                full_text = tokenizer.decode(all_ids, skip_special_tokens=True)
                delta = full_text[len(prev_text):]
                if delta:
                    prev_text = full_text
                    yield _sse(ChatCompletionChunk(
                        id=cid, created=ts, model=name,
                        choices=[ChunkChoice(delta=DeltaMessage(content=delta))],
                    ))
    except Exception as e:
        error_occurred = True
        print(f"[Server] Streaming inference error: {e}")
        yield _sse(ChatCompletionChunk(
            id=cid, created=ts, model=name,
            choices=[ChunkChoice(delta=DeltaMessage(
                content=f"\n\n[错误: 推理失败 — {e}]"
            ))],
        ))

    # Finish chunk
    yield _sse(ChatCompletionChunk(
        id=cid, created=ts, model=name,
        choices=[ChunkChoice(delta=DeltaMessage(),
                             finish_reason="error" if error_occurred else "stop")],
    ))
    yield "data: [DONE]\n\n"


def _sse(chunk: ChatCompletionChunk) -> str:
    """Format a chunk as a Server-Sent Events data line."""
    return f"data: {chunk.model_dump_json()}\n\n"


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"


@app.get("/")
async def serve_ui():
    index = _static_dir / "index.html"
    if index.exists():
        return FileResponse(str(index), media_type="text/html")
    return {"message": "LLAISYS Chat API running. POST /v1/chat/completions"}


if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
