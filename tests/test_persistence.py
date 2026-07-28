"""JobStore fail-closed and tenant isolation tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ruitong.api import JobInfo, JobStatus
from ruitong.jobs.persistence import JobStore


@pytest.fixture
def store() -> JobStore:
    """In-memory JobStore for test isolation."""
    return JobStore(":memory:")


def _make_job(job_id: str = "job-1") -> JobInfo:
    """Create a minimal JobInfo for testing."""
    now = datetime.now(timezone.utc)
    return JobInfo(
        job_id=job_id,
        status=JobStatus.pending,
        created_at=now,
        updated_at=now,
    )


class TestFailClosed:
    """Store methods reject empty/None owner — belt and braces."""

    def test_get_rejects_empty(self, store: JobStore) -> None:
        """get() with empty owner raises ValueError."""
        job = _make_job()
        store.create(job, owner="tenant-a")
        with pytest.raises(ValueError, match="owner is required"):
            store.get(job.job_id, owner="")

    def test_list_by_owner_rejects_empty(self, store: JobStore) -> None:
        """list_by_owner() with empty owner raises ValueError."""
        job = _make_job()
        store.create(job, owner="tenant-a")
        with pytest.raises(ValueError, match="owner is required"):
            store.list_by_owner(owner="")

    def test_delete_rejects_empty(self, store: JobStore) -> None:
        """delete() with empty owner raises ValueError."""
        job = _make_job()
        store.create(job, owner="tenant-a")
        with pytest.raises(ValueError, match="owner is required"):
            store.delete(job.job_id, owner="")

    def test_delete_rejects_none(self, store: JobStore) -> None:
        """delete() with None owner raises ValueError."""
        job = _make_job()
        store.create(job, owner="tenant-a")
        with pytest.raises(ValueError, match="owner is required"):
            store.delete(job.job_id, owner=None)  # type: ignore[arg-type]


class TestTenantIsolation:
    """Cross-tenant data is never visible."""

    def test_owner_matched_exactly(self, store: JobStore) -> None:
        """tenant-b cannot see tenant-a's jobs via get(), list, or delete."""
        job_a = _make_job("job-a")
        job_b = _make_job("job-b")
        store.create(job_a, owner="tenant-a")
        store.create(job_b, owner="tenant-b")

        # tenant-b cannot get tenant-a's job
        result = store.get("job-a", owner="tenant-b")
        assert result is None

        # tenant-b's list does not include tenant-a's job
        b_jobs = store.list_by_owner(owner="tenant-b")
        assert len(b_jobs) == 1
        assert b_jobs[0].job_id == "job-b"
        assert "job-a" not in {j.job_id for j in b_jobs}

        # tenant-b cannot delete tenant-a's job
        deleted = store.delete("job-a", owner="tenant-b")
        assert deleted is False

        # tenant-a can still get their own job
        a_job = store.get("job-a", owner="tenant-a")
        assert a_job is not None
        assert a_job.job_id == "job-a"


class TestJobInfoModel:
    """JobInfo response model does not expose owner."""

    def test_owner_not_in_response(self) -> None:
        """JobInfo has no 'owner' field — owner is never leaked."""
        job = _make_job()
        data = job.model_dump()
        assert "owner" not in data
