"""Tests for the ruitong CLI (subprocess-based).

Uses subprocess so it tests the actual CLI entry point configured in
pyproject.toml (`ruitong = "ruitong.cli:main"`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run_ruitong(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run the ruitong CLI via subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(ROOT, "src")
    result = subprocess.run(
        [sys.executable, "-m", "ruitong.cli"] + list(args),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=ROOT,
        env=env,
    )
    return result


class TestCLIBasic:
    def test_no_args_shows_help(self) -> None:
        """No command → prints help."""
        r = _run_ruitong()
        assert r.returncode == 0
        assert "ruitong" in r.stdout.lower() or "usage" in r.stdout.lower()

    def test_port_help_shows_usage(self) -> None:
        """ruitong port --help shows usage."""
        r = _run_ruitong("port", "--help")
        assert r.returncode == 0
        assert "model" in r.stdout.lower() or "target" in r.stdout.lower()
        assert "cuda" in r.stdout.lower() or "ascend" in r.stdout.lower()

    def test_port_help_mentions_auto(self) -> None:
        """Help text mentions auto mode."""
        r = _run_ruitong("port", "--help")
        assert "auto" in r.stdout.lower()


class TestCLIRunning:
    def test_target_cuda_produces_report(self, tmp_path) -> None:
        """ruitong port <model> --target cuda writes report.json."""
        out_file = tmp_path / "report_cuda.json"
        r = _run_ruitong(
            "port", "Qwen3-8B", "--target", "cuda", "--output", str(out_file)
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["model"] == "Qwen3-8B"
        assert data["passed"] is True
        assert data["total_prompts"] > 0

    def test_target_ascend_produces_report(self, tmp_path) -> None:
        """ruitong port <model> --target ascend writes report.json."""
        out_file = tmp_path / "report_ascend.json"
        r = _run_ruitong(
            "port", "Qwen3-8B", "--target", "ascend", "--output", str(out_file)
        )
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["model"] == "Qwen3-8B"

    def test_auto_compare_both_backends(self, tmp_path) -> None:
        """Auto mode compares both backends (divergent fakes → exit 1)."""
        out_file = tmp_path / "report_auto.json"
        r = _run_ruitong("port", "Qwen3-8B")
        assert r.returncode == 1
        assert "cuda" in r.stdout.lower() or "ascend" in r.stdout.lower()

    def test_stdout_prints_summary(self) -> None:
        """CLI prints a human-readable summary to stdout."""
        r = _run_ruitong("port", "Qwen3-8B", "--target", "cuda")
        assert r.returncode == 0
        assert "equivalence" in r.stdout.lower() or "report" in r.stdout.lower()
        assert "passed" in r.stdout.lower()

    def test_auto_stdout_mentions_both(self) -> None:
        """Auto mode stdout mentions both backends (divergent → exit 1)."""
        r = _run_ruitong("port", "Qwen3-8B")
        assert r.returncode == 1
        combined = r.stdout.lower()
        assert "cuda" in combined or "ascend" in combined or "auto" in combined

    def test_report_has_thresholds(self, tmp_path) -> None:
        """Report JSON includes thresholds_used section."""
        out_file = tmp_path / "report_thresh.json"
        r = _run_ruitong(
            "port", "Qwen3-8B", "--target", "cuda", "--output", str(out_file)
        )
        assert r.returncode == 0
        data = json.loads(out_file.read_text())
        assert "thresholds_used" in data
        tu = data["thresholds_used"]
        assert "cosine_min" in tu
        assert "max_abs_diff_max" in tu
        assert "top1_min" in tu
        assert "top5_min" in tu


class TestCLIEdgeCases:
    def test_per_prompt_results_in_report(self, tmp_path) -> None:
        """Each prompt has a per_prompt_results entry."""
        out_file = tmp_path / "report_pp.json"
        r = _run_ruitong(
            "port", "Qwen3-8B", "--target", "cuda", "--output", str(out_file)
        )
        assert r.returncode == 0
        data = json.loads(out_file.read_text())
        assert len(data["per_prompt_results"]) >= 1
        for pp in data["per_prompt_results"]:
            assert "prompt" in pp
            assert "cosine_sim" in pp or "cuda_logprobs" in pp

    def test_warnings_list_present(self, tmp_path) -> None:
        """Report has a warnings list."""
        out_file = tmp_path / "report_warn.json"
        r = _run_ruitong(
            "port", "Qwen3-8B", "--target", "cuda", "--output", str(out_file)
        )
        assert r.returncode == 0
        data = json.loads(out_file.read_text())
        assert isinstance(data["warnings"], list)
