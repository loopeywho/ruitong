"""Phase 5 — Port API schemas and endpoints.

Wraps EquivalenceRunner as a REST service.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from typing import Literal


class PortRequest(BaseModel):
    """Request to run an equivalence port comparison."""

    model: str = Field(..., min_length=1, description="Model name to compare")
    target: str = Field(
        default="auto",
        pattern=r"^(cuda|ascend|auto)$",
        description="Backend target: cuda, ascend, or auto",
    )
    prompts: list[str] | None = Field(
        default=None,
        min_length=1,
        description="Custom prompts (default: 3 built-in prompts)",
    )


class PortMetric(BaseModel):
    """Single aggregate metric from a port run."""

    name: str
    value: float | None = None


class PerPromptMetric(BaseModel):
    """Per-prompt comparison result."""

    prompt: str
    mode: str
    cosine_sim: float | None = None
    max_abs_diff: float | None = None
    top1_agreement: float | None = None
    top5_set_agreement: float | None = None
    response_parity: float | None = None


class PortReport(BaseModel):
    """Full port report response."""

    model: str
    mode: str
    total_prompts: int
    passed: bool
    validation_level: Literal["simulated", "staging", "production"] = "simulated"
    metrics: list[PortMetric]
    per_prompt: list[PerPromptMetric]
    warnings: list[str]
    thresholds: dict[str, float]


class PortError(BaseModel):
    """Error response for port operations."""

    error: str
    detail: str | None = None


# ── Async job types ──────────────────────────────────────────────────


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    error = "error"


class JobResult(BaseModel):
    """Result of a completed async job."""

    report: PortReport
    error: str | None = None


class JobInfo(BaseModel):
    """Status of an async port job."""

    job_id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    result: JobResult | None = None