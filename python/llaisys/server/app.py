"""LLAISYS Chat Server — OpenAI-compatible FastAPI application."""

import asyncio
import ctypes
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ..libllaisys import LIB_LLAISYS
from .schemas import (
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChunkChoice,
    CreateSessionRequest,
    DeltaMessage,
    SessionDetail,
    SessionSummary,
    UpdateSessionRequest,
    Usage,
)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

_model = None
_tokenizer = None
_max_seq_len = 8192
_model_lock = threading.Lock()
MAX_INPUT_TOKENS = 4096
GENERATION_TIMEOUT_S = 300

# 当前真实驻留在模型 KV-cache 中的前缀状态（单模型实例）
_kv_active_session_id: Optional[str] = None
_kv_active_prefix: List[int] = []


@dataclass
class SessionState:
    id: str
    title: str = "新对话"
    messages: List[dict] = field(default_factory=list)
    updated_at: int = field(default_factory=lambda: int(time.time()))


class SessionManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: Dict[str, SessionState] = {}

    def list(self) -> List[SessionSummary]:
        with self._lock:
            sessions = sorted(
                self._sessions.values(),
                key=lambda s: s.updated_at,
                reverse=True,
            )
            return [
                SessionSummary(
                    id=s.id,
                    title=s.title,
                    updated_at=s.updated_at,
                    message_count=len(s.messages),
                )
                for s in sessions
            ]

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            return SessionState(
                id=session.id,
                title=session.title,
                messages=[dict(m) for m in session.messages],
                updated_at=session.updated_at,
            )

    def upsert(self, session_id: str, title: Optional[str], messages: List[dict]) -> SessionState:
        with self._lock:
            now = int(time.time())
            session = self._sessions.get(session_id)
            if session is None:
                session = SessionState(id=session_id)
                self._sessions[session_id] = session
            if title is not None:
                session.title = title
            session.messages = [dict(m) for m in messages]
            session.updated_at = now
            return SessionState(
                id=session.id,
                title=session.title,
                messages=[dict(m) for m in session.messages],
                updated_at=session.updated_at,
            )

    def create(self, title: str) -> SessionState:
        session_id = f"sess-{uuid.uuid4().hex[:12]}"
        return self.upsert(session_id, title, [])

    def patch(self, session_id: str, title: Optional[str], messages: Optional[List[dict]]) -> SessionState:
        with self._lock:
            now = int(time.time())
            session = self._sessions.get(session_id)
            if session is None:
                raise KeyError(session_id)
            if title is not None:
                session.title = title
            if messages is not None:
                session.messages = [dict(m) for m in messages]
            session.updated_at = now
            return SessionState(
                id=session.id,
                title=session.title,
                messages=[dict(m) for m in session.messages],
                updated_at=session.updated_at,
            )

    def delete(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]


_session_mgr = SessionManager()

app = FastAPI(title="LLAISYS Chat API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

def load_model(model_path: str, device: str = "nvidia", max_seq_len: int = 8192,
               quantized: bool = False):
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
# Helpers
# ---------------------------------------------------------------------------

def _infer_one(model, token: int, req: ChatCompletionRequest, seed: int) -> int:
    in_ptr = (ctypes.c_int64 * 1)(int(token))
    use_sampling = not (req.top_k == 1)
    if use_sampling:
        return int(LIB_LLAISYS.llaisysQwen2ModelInferEx(
            model.handle,
            in_ptr,
            ctypes.c_size_t(1),
            ctypes.c_float(req.temperature),
            ctypes.c_int(req.top_k),
            ctypes.c_float(req.top_p),
            ctypes.c_uint64(seed),
        ))
    return int(LIB_LLAISYS.llaisysQwen2ModelInfer(
        model.handle,
        in_ptr,
        ctypes.c_size_t(1),
    ))


def _ensure_prefilled_prefix(model, session_id: str, prefix_ids: List[int]):
    """KV-cache 前缀复用：相同会话且新前缀是旧前缀扩展时，仅补算delta。"""
    global _kv_active_session_id, _kv_active_prefix

    if (
        _kv_active_session_id == session_id
        and len(prefix_ids) >= len(_kv_active_prefix)
        and prefix_ids[:len(_kv_active_prefix)] == _kv_active_prefix
    ):
        start = len(_kv_active_prefix)
    else:
        model.reset()
        _kv_active_session_id = session_id
        _kv_active_prefix = []
        start = 0

    for token in prefix_ids[start:]:
        in_ptr = (ctypes.c_int64 * 1)(int(token))
        LIB_LLAISYS.llaisysQwen2ModelInfer(model.handle, in_ptr, ctypes.c_size_t(1))

    _kv_active_prefix = list(prefix_ids)


def _sanitize_input_ids(input_ids: List[int]) -> List[int]:
    if len(input_ids) > MAX_INPUT_TOKENS:
        raise HTTPException(400, f"Input too long: {len(input_ids)} tokens > limit {MAX_INPUT_TOKENS}")

    min_gen = 32
    max_input = _max_seq_len - min_gen
    if len(input_ids) > max_input:
        truncated = len(input_ids) - max_input
        input_ids = input_ids[-max_input:]
        print(
            f"[Server] Warning: Input truncated by {truncated} tokens "
            f"to fit max_seq_len={_max_seq_len}"
        )
    return input_ids


def _build_session_title(messages: List[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            text = (msg.get("content") or "").strip()
            if text:
                return (text[:24] + "…") if len(text) > 24 else text
    return "新对话"


# ---------------------------------------------------------------------------
# Session APIs
# ---------------------------------------------------------------------------

@app.get("/v1/sessions")
async def list_sessions():
    return {"data": [s.model_dump() for s in _session_mgr.list()]}


@app.post("/v1/sessions")
async def create_session(req: CreateSessionRequest):
    s = _session_mgr.create(req.title)
    return SessionDetail(
        id=s.id,
        title=s.title,
        messages=[ChatMessage(**m) for m in s.messages],
        updated_at=s.updated_at,
    )


@app.get("/v1/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        s = _session_mgr.get(session_id)
    except KeyError:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return SessionDetail(
        id=s.id,
        title=s.title,
        messages=[ChatMessage(**m) for m in s.messages],
        updated_at=s.updated_at,
    )


@app.patch("/v1/sessions/{session_id}")
async def patch_session(session_id: str, req: UpdateSessionRequest):
    payload = req.model_dump(exclude_none=True)
    try:
        s = _session_mgr.patch(
            session_id,
            payload.get("title"),
            payload.get("messages"),
        )
    except KeyError:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return SessionDetail(
        id=s.id,
        title=s.title,
        messages=[ChatMessage(**m) for m in s.messages],
        updated_at=s.updated_at,
    )


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str):
    _session_mgr.delete(session_id)
    global _kv_active_session_id, _kv_active_prefix
    if _kv_active_session_id == session_id:
        _kv_active_session_id = None
        _kv_active_prefix = []
    return {"ok": True}


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

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    input_ids = _sanitize_input_ids(input_ids)

    effective_max_tokens = min(req.max_tokens, _max_seq_len - len(input_ids))
    if effective_max_tokens < 1:
        raise HTTPException(400, f"Input too long ({len(input_ids)} tokens) for max_seq_len={_max_seq_len}")

    session_id = req.session_id or f"adhoc-{uuid.uuid4().hex[:12]}"
    seed = req.seed if req.seed is not None else 0

    # upsert用户侧会话快照（assistant结果稍后补）
    _session_mgr.upsert(session_id, _build_session_title(messages), messages)

    if req.stream:
        return StreamingResponse(
            _stream_sync(model, tokenizer, input_ids, req, seed, effective_max_tokens, session_id, messages),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await asyncio.to_thread(
        _generate_sync, model, tokenizer, input_ids, req, seed, effective_max_tokens, session_id, messages
    )
    return result


# ---------------------------------------------------------------------------
# Non-streaming helper
# ---------------------------------------------------------------------------

def _generate_sync(model, tokenizer, input_ids, req, seed, max_tokens, session_id, base_messages):
    if len(input_ids) == 0:
        raise HTTPException(400, "Empty input")

    prefix_ids = input_ids[:-1]
    last_input = input_ids[-1]

    out_ids: List[int] = []
    try:
        with _model_lock:
            _ensure_prefilled_prefix(model, session_id, prefix_ids)
            next_token = _infer_one(model, last_input, req, seed)
            out_ids.append(next_token)

            for _ in range(max_tokens - 1):
                if next_token == model.meta.end_token:
                    break
                next_token = _infer_one(model, next_token, req, seed)
                out_ids.append(next_token)

            global _kv_active_prefix
            _kv_active_prefix = list(input_ids) + out_ids
    except Exception as e:
        print(f"[Server] Inference error: {e}")
        raise HTTPException(500, f"Inference failed: {e}")

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    all_messages = list(base_messages) + [{"role": "assistant", "content": text}]
    _session_mgr.upsert(session_id, _build_session_title(all_messages), all_messages)

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
            completion_tokens=len(out_ids),
            total_tokens=len(input_ids) + len(out_ids),
        ),
    )


# ---------------------------------------------------------------------------
# Streaming helper
# ---------------------------------------------------------------------------

def _stream_sync(model, tokenizer, input_ids, req, seed, max_tokens, session_id, base_messages) -> Generator[str, None, None]:
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    ts = int(time.time())
    name = req.model

    yield _sse(ChatCompletionChunk(
        id=cid,
        created=ts,
        model=name,
        choices=[ChunkChoice(delta=DeltaMessage(role="assistant"))],
    ))

    out_ids: List[int] = []
    prev_text = ""
    error_occurred = False

    try:
        t_start = time.time()
        with _model_lock:
            prefix_ids = input_ids[:-1]
            last_input = input_ids[-1]
            _ensure_prefilled_prefix(model, session_id, prefix_ids)

            next_token = _infer_one(model, last_input, req, seed)
            for _ in range(max_tokens):
                if next_token == model.meta.end_token:
                    break
                if time.time() - t_start > GENERATION_TIMEOUT_S:
                    raise TimeoutError(f"Generation exceeded {GENERATION_TIMEOUT_S}s limit")

                out_ids.append(next_token)
                full_text = tokenizer.decode(out_ids, skip_special_tokens=True)
                delta = full_text[len(prev_text):]
                if delta:
                    prev_text = full_text
                    yield _sse(ChatCompletionChunk(
                        id=cid,
                        created=ts,
                        model=name,
                        choices=[ChunkChoice(delta=DeltaMessage(content=delta))],
                    ))

                next_token = _infer_one(model, next_token, req, seed)

            global _kv_active_prefix
            _kv_active_prefix = list(input_ids) + out_ids

    except Exception as e:
        error_occurred = True
        print(f"[Server] Streaming inference error: {e}")
        yield _sse(ChatCompletionChunk(
            id=cid,
            created=ts,
            model=name,
            choices=[ChunkChoice(delta=DeltaMessage(content=f"\n\n[错误: 推理失败 — {e}]"))],
        ))

    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    all_messages = list(base_messages) + [{"role": "assistant", "content": text}]
    _session_mgr.upsert(session_id, _build_session_title(all_messages), all_messages)

    yield _sse(ChatCompletionChunk(
        id=cid,
        created=ts,
        model=name,
        choices=[ChunkChoice(delta=DeltaMessage(), finish_reason="error" if error_occurred else "stop")],
    ))
    yield "data: [DONE]\n\n"


def _sse(chunk: ChatCompletionChunk) -> str:
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
