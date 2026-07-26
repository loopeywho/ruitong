"""Port API router — equivalence comparison via REST."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from ..backends.fake import FakeAscend, FakeCuda
from ..equivalence.runner import EquivalenceRunner, EquivalenceReport
from ..jobs.persistence import JobStore
from . import (
    JobInfo,
    JobResult,
    JobStatus,
    PerPromptMetric,
    PortMetric,
    PortReport,
    PortRequest,
)

router = APIRouter(prefix="/v1/port", tags=["port"])


def _make_runner(
    model: str, target: str
) -> tuple[EquivalenceRunner, str | None, str | None]:
    """Build the runner and determine which backends to compare."""
    if target == "auto":
        cuda = FakeCuda(model_ids=[model])
        ascend = FakeAscend(model_ids=[model])
        return EquivalenceRunner(cuda, ascend), "cuda", "ascend"
    elif target == "cuda":
        cuda = FakeCuda(model_ids=[model])
        return EquivalenceRunner(cuda, cuda), "cuda", "cuda"
    elif target == "ascend":
        ascend = FakeAscend(model_ids=[model])
        return EquivalenceRunner(ascend, ascend), "ascend", "ascend"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown target: {target}")


def _default_prompts(model: str) -> list[str]:
    return [
        f"What is {model}? Answer in one sentence.",
        "Explain the concept of transformers.",
        "Write a short poem about AI.",
    ]


def _report_to_response(report: EquivalenceReport, validation_level: str = "simulated") -> PortReport:
    """Convert an EquivalenceReport to the API response model."""
    m = report.metrics
    metrics_list = [
        PortMetric(name="cosine_similarity", value=m.get("cosine_similarity")),
        PortMetric(name="max_absolute_difference", value=m.get("max_absolute_difference")),
        PortMetric(name="top1_agreement", value=m.get("top1_agreement")),
        PortMetric(name="top5_set_agreement", value=m.get("top5_set_agreement")),
        PortMetric(name="response_parity", value=m.get("response_parity")),
    ]
    per_prompt = [
        PerPromptMetric(
            prompt=r.prompt,
            mode=r.mode,
            cosine_sim=r.cosine_sim,
            max_abs_diff=r.max_abs_diff,
            top1_agreement=r.top1_agreement,
            top5_set_agreement=r.top5_set_agreement,
            response_parity=r.response_parity,
        )
        for r in report.per_prompt_results
    ]
    return PortReport(
        model=report.model,
        mode=report.mode,
        total_prompts=report.total_prompts,
        passed=report.passed,
        validation_level=validation_level,
        metrics=metrics_list,
        per_prompt=per_prompt,
        warnings=report.warnings,
        thresholds={
            "cosine_min": report.thresholds_used.COSINE_MIN,
            "max_abs_diff_max": report.thresholds_used.MAX_ABS_DIFF_MAX,
            "top1_min": report.thresholds_used.TOP1_MIN,
            "top5_min": report.thresholds_used.TOP5_MIN,
        },
    )


# ── Sync endpoint (deprecated) ──────────────────────────────────────


@router.post("", response_model=PortReport)
async def run_port(req: PortRequest, request: Request) -> PortReport:
    """Run an equivalence port comparison and return the report synchronously.

    Deprecated: prefer POST /v1/port/preview (async with job polling).
    """
    runner, _, _ = _make_runner(req.model, req.target)
    prompts = req.prompts if req.prompts is not None else _default_prompts(req.model)
    report: EquivalenceReport = await runner.run(req.model, prompts)
    return _report_to_response(report, validation_level="simulated")


# ── Async job endpoints ─────────────────────────────────────────────


@router.post("/preview", status_code=202, response_model=JobInfo)
async def submit_port_job(req: PortRequest, request: Request) -> JobInfo:
    """Submit an equivalence comparison as an async job.

    Returns immediately with a job_id. Poll GET /v1/port/preview/{job_id}
    to retrieve the result when status is ``done``.
    """
    job_id = uuid.uuid4().hex
    now = datetime.now(timezone.utc)

    job = JobInfo(
        job_id=job_id,
        status=JobStatus.pending,
        created_at=now,
        updated_at=now,
    )

    # Persist job to the store
    store = getattr(request.app.state, "job_store", None)
    if store is None:
        store = JobStore.default()
    store.create(job, model=req.model, target=req.target)

    # Launch background task
    import asyncio

    async def _run_job() -> None:
        try:
            store.update_status(job_id, JobStatus.running)

            runner, _, _ = _make_runner(req.model, req.target)
            prompts = (
                req.prompts
                if req.prompts is not None
                else _default_prompts(req.model)
            )
            report = await runner.run(req.model, prompts)

            result = JobResult(report=_report_to_response(report, validation_level="simulated"))
            store.update_result(job_id, JobStatus.done, result)
        except Exception as exc:
            result = JobResult(
                report=PortReport(
                    model=req.model,
                    mode="error",
                    total_prompts=0,
                    passed=False,
                    validation_level="simulated",
                    metrics=[],
                    per_prompt=[],
                    warnings=[],
                    thresholds={},
                ),
                error=f"{type(exc).__name__}: {exc}",
            )
            store.update_result(job_id, JobStatus.error, result)

    asyncio.create_task(_run_job())
    return job


@router.get("/preview/{job_id}", response_model=JobInfo)
async def get_port_job(job_id: str, request: Request) -> JobInfo:
    """Poll the status and result of an async port job."""
    store = getattr(request.app.state, "job_store", None)
    if store is None:
        store = JobStore.default()
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job