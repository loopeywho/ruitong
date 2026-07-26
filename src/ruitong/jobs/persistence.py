"""SQLite-backed job persistence for async port jobs.

Replaces the in-memory dict with a file-backed store that survives restarts.
Uses sqlite3 from stdlib — zero external dependencies.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..api import JobInfo, JobResult, JobStatus, PortReport


class JobStore:
    """Persistent job store backed by SQLite.

    Thread-safe via a reentrant lock. Compatible with FastAPI's async
    request lifecycle — SQLite operations are fast enough for low
    concurrency; for higher throughput wrap calls in run_in_executor.
    """

    _default_instance: JobStore | None = None

    @classmethod
    def default(cls) -> JobStore:
        """Return the module-level singleton (in-memory when no path set)."""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance

    def __init__(self, db_path: str = "") -> None:
        """Open or create the database.

        When *db_path* is empty, uses ``:memory:`` — jobs are lost on
        restart (same as the old in-memory dict, but with the same
        query interface).
        """
        path = db_path or ":memory:"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id     TEXT PRIMARY KEY,
                    status     TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    model      TEXT NOT NULL DEFAULT '',
                    target     TEXT NOT NULL DEFAULT 'auto',
                    result     TEXT,
                    error      TEXT
                );
            """)
            self._conn.commit()

    # ── CRUD ─────────────────────────────────────────────────────────

    def create(self, job: JobInfo, model: str = "", target: str = "") -> None:
        """Insert a new job record."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO jobs (job_id, status, created_at, updated_at, model, target)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    job.job_id,
                    job.status.value,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    model,
                    target,
                ),
            )
            self._conn.commit()

    def get(self, job_id: str) -> JobInfo | None:
        """Fetch a job by ID, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def update_status(self, job_id: str, status: JobStatus) -> None:
        """Update job status and updated_at."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status.value, now, job_id),
            )
            self._conn.commit()

    def update_result(self, job_id: str, status: JobStatus, result: JobResult) -> None:
        """Mark a job as completed/failed and store its result."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """UPDATE jobs SET status = ?, updated_at = ?, result = ?, error = ?
                   WHERE job_id = ?""",
                (
                    status.value,
                    now,
                    result.report.model_dump_json() if result.report else None,
                    result.error,
                    job_id,
                ),
            )
            self._conn.commit()

    def count_active(self) -> int:
        """Number of jobs that are pending or running."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM jobs WHERE status IN ('pending', 'running')"
            ).fetchone()
        return row["cnt"] if row else 0

    # ── Internal ─────────────────────────────────────────────────────

    def _row_to_job(self, row: sqlite3.Row) -> JobInfo:
        result: JobResult | None = None
        raw_result = row["result"]
        raw_error = row["error"]
        if raw_result is not None:
            try:
                report_dict: dict[str, Any] = json.loads(raw_result)
                report = PortReport(**report_dict)
                result = JobResult(report=report, error=raw_error)
            except (json.JSONDecodeError, TypeError, ValueError):
                result = JobResult(
                    report=PortReport(
                        model=row["model"],
                        mode="error",
                        total_prompts=0,
                        passed=False,
                        validation_level="simulated",
                        metrics=[],
                        per_prompt=[],
                        warnings=[],
                        thresholds={},
                    ),
                    error=f"DeserializationError: {raw_error or 'failed to parse result'}",
                )
        elif raw_error:
            result = JobResult(
                report=PortReport(
                    model=row["model"],
                    mode="error",
                    total_prompts=0,
                    passed=False,
                    validation_level="simulated",
                    metrics=[],
                    per_prompt=[],
                    warnings=[],
                    thresholds={},
                ),
                error=raw_error,
            )

        return JobInfo(
            job_id=row["job_id"],
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            result=result,
        )

    def close(self) -> None:
        self._conn.close()