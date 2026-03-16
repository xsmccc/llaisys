"""OpenAI-compatible request / response schemas for LLAISYS Chat Server."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

import time
import uuid


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-r1-distill-qwen-1.5b"
    messages: List[ChatMessage]
    session_id: Optional[str] = None
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=500)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage


# ---------------------------------------------------------------------------
# Streaming (chunked) response
# ---------------------------------------------------------------------------

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChunkChoice]


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

class SessionSummary(BaseModel):
    id: str
    title: str
    updated_at: int
    message_count: int


class SessionDetail(BaseModel):
    id: str
    title: str
    messages: List[ChatMessage]
    updated_at: int


class CreateSessionRequest(BaseModel):
    title: str = "新对话"


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None
