"""OpenAI-compatible request/response schemas with Ruitong backend extension.

Only `backend` is a Ruitong extension; every other field mirrors the OpenAI
chat-completions shape so stock clients work unmodified.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Roles accepted by OpenAI-compatible servers including vLLM. Kept as a
# Literal so an unknown role fails at our edge with a clear error rather than
# surfacing as an opaque 400 from the backend.
Role = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel, frozen=True):
    """A single chat message."""

    role: Role
    content: str


class ChatRequest(BaseModel, frozen=True):
    """OpenAI-compatible chat request with Ruitong backend selector."""

    model: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    backend: Literal["cuda", "ascend", "auto"] = "auto"
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int | None = None
    stream: bool = False
    # An OpenAI-compatible server returns logprobs ONLY when asked. Without
    # these fields the request never asks, vLLM returns text alone, and every
    # equivalence comparison silently degrades to weak text matching — the
    # fixtures hid this because fakes return logprobs unconditionally.
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)


class TopLogprob(BaseModel, frozen=True):
    """One candidate token and its logprob at a single position."""

    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobEntry(BaseModel, frozen=True):
    """The chosen token at one position, plus the top-k alternatives."""

    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel, frozen=True):
    """OpenAI's `choices[].logprobs` — an OBJECT, not a list.

    Verified against a live OpenAI-shaped server: the wire format is
    `{"content": [{token, logprob, bytes, top_logprobs: [...]}, ...]}`.
    Declaring this as `list[float]` made every real response fail validation
    and silently degrade the run to text-only comparison. The fixtures hid it
    because they returned a bare list that nothing else produces.
    """

    content: list[LogprobEntry] = Field(default_factory=list)

    def flat(self) -> list[float]:
        """Chosen-token logprob per position."""
        return [entry.logprob for entry in self.content]

    def top_k_matrix(self) -> list[list[float]]:
        """Per position, the top-k logprobs — the real comparison input."""
        return [[t.logprob for t in e.top_logprobs] for e in self.content]

    def top_k_tokens(self) -> list[list[str]]:
        """Per position, the top-k token strings.

        Token identity is what top-1/top-5 agreement is supposed to compare;
        comparing positional indices instead is meaningless across backends.
        """
        return [[t.token for t in e.top_logprobs] for e in self.content]


class Choice(BaseModel, frozen=True):
    """A single choice in a chat response."""

    index: int = Field(ge=0)
    message: Message
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class Usage(BaseModel, frozen=True):
    """Token usage counts."""

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class ChatResponse(BaseModel, frozen=True):
    """OpenAI-compatible chat response with Ruitong backend extension."""

    id: str
    model: str
    backend: str
    choices: list[Choice]
    usage: Usage
    created: int | None = None


class Delta(BaseModel, frozen=True):
    """Streaming delta — fields may be None."""

    role: Role | None = None
    content: str | None = None


class DeltaChoice(BaseModel, frozen=True):
    """A single streaming choice."""

    index: int = Field(ge=0)
    delta: Delta
    finish_reason: str | None = None


class ChatChunk(BaseModel, frozen=True):
    """Streaming chat chunk."""

    id: str
    model: str
    backend: str
    choices: list[DeltaChoice]


class HealthStatus(BaseModel, frozen=True):
    """Backend health status."""

    healthy: bool
    backend: str
    models_served: int = Field(ge=0)
    uptime_seconds: float | None = Field(default=None, ge=0.0)


class ModelInfo(BaseModel, frozen=True):
    """Single model descriptor."""

    id: str
    backend: str
    max_model_len: int | None = Field(default=None, gt=0)
