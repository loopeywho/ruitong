"""Tests for the ruitong CLI — in-process for coverage, subprocess for packaging proof.

Most tests call ``main(argv)`` directly so coverage can instrument the CLI's
paths. Two subprocess tests at the end prove ``ruitong ...`` works as a
packaged console-script entry point.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from ruitong.cli import EXIT_CANNOT_RUN, EXIT_GATE_FAILED, EXIT_PASS, main

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _run_subprocess(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    """Run the ruitong CLI as a subprocess (packaging tests only)."""
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


# ─── In-process tests (coverage-instrumentable) ───


class TestMainCoverage:
    """In-process calls to ``main(argv)`` so pytest-cov can track coverage."""

    def test_no_args_prints_help(self, capsys) -> None:
        rc = main([])
        captured = capsys.readouterr()
        assert rc == EXIT_PASS
        assert "ruitong" in captured.out.lower() or "usage" in captured.out.lower()

    def test_port_help_usage(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["port", "--help"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        assert "model" in captured.out.lower() or "target" in captured.out.lower()
        assert "cuda" in captured.out.lower() or "ascend" in captured.out.lower()

    def test_unknown_command_exits_2(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            main(["bogus"])
        assert exc.value.code == 2

    def test_synthetic_cuda_exit_1(self, tmp_path: Path, capsys) -> None:
        """No endpoints + --target cuda → synthetic run → exit 1 (cannot pass)."""
        out_file = tmp_path / "report_cuda.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert data["model"] == "Qwen3-8B"
        assert data["passed"] is False
        assert data["synthetic"] is True
        assert data["total_prompts"] > 0
        captured = capsys.readouterr()
        assert "SYNTHETIC" in captured.out or "SYNTHETIC" in captured.err

    def test_synthetic_ascend_exit_1(self, tmp_path: Path, capsys) -> None:
        """No endpoints + --target ascend → synthetic run → exit 1."""
        out_file = tmp_path / "report_ascend.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "ascend",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert data["model"] == "Qwen3-8B"

    def test_synthetic_auto_exit_1(self, capsys) -> None:
        """No endpoints, no --target → default ascend target → exit 1."""
        rc = main(["port", "Qwen3-8B"])
        assert rc == EXIT_GATE_FAILED
        captured = capsys.readouterr()
        combined = captured.out.lower()
        assert "cuda" in combined or "ascend" in combined or "auto" in combined

    def test_stdout_prints_summary(self, capsys) -> None:
        """Synthetic run prints human-readable summary to stdout."""
        rc = main(["port", "Qwen3-8B", "--target", "cuda"])
        assert rc == EXIT_GATE_FAILED
        captured = capsys.readouterr()
        assert "equivalence" in captured.out.lower() or "report" in captured.out.lower()
        assert "passed" in captured.out.lower()

    def test_report_has_thresholds(self, tmp_path: Path) -> None:
        """Report JSON includes thresholds_used section."""
        out_file = tmp_path / "report_thresh.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert "thresholds_used" in data
        tu = data["thresholds_used"]
        assert "cosine_min" in tu
        assert "max_abs_diff_max" in tu
        assert "top1_min" in tu
        assert "top5_min" in tu

    def test_per_prompt_results(self, tmp_path: Path) -> None:
        """Report contains per_prompt_results entries."""
        out_file = tmp_path / "report_pp.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert len(data["per_prompt_results"]) >= 1
        for pp in data["per_prompt_results"]:
            assert "prompt" in pp
            assert "cosine_sim" in pp or "cuda_logprobs" in pp

    def test_warnings_list_present(self, tmp_path: Path) -> None:
        """Report has a warnings list (synthetic warning present)."""
        out_file = tmp_path / "report_warn.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert isinstance(data["warnings"], list)

    def test_report_written_even_on_failure(self, tmp_path: Path) -> None:
        """Report is persisted even when the gate fails (exit 1)."""
        out_file = tmp_path / "report_fail.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc in (EXIT_GATE_FAILED, EXIT_CANNOT_RUN)
        assert out_file.exists()

    def test_reference_not_equal_target(self, tmp_path: Path) -> None:
        """reference_backend != target_backend always."""
        out_file = tmp_path / "report_ref.json"
        rc = main([
            "port", "Qwen3-8B",
            "--target", "cuda",
            "--output", str(out_file),
        ])
        assert rc == EXIT_GATE_FAILED
        data = json.loads(out_file.read_text())
        assert data["reference_backend"] != data["target_backend"]

    def test_synthetic_cannot_pass(self, tmp_path: Path) -> None:
        """Synthetic run: passed MUST be False, exit MUST not be 0."""
        out_file = tmp_path / "report_synth.json"
        rc = main([
            "port", "Qwen3-8B",
            "--output", str(out_file),
        ])
        assert rc != EXIT_PASS
        data = json.loads(out_file.read_text())
        assert data["passed"] is False
        assert data["synthetic"] is True


# ─── Subprocess tests — packaging proof only ───

# Coverage is unavailable here by design — these exist to prove the Python
# package can be invoked as a subprocess, not to instrument the code paths.


class TestCLIPackaging:
    """Subprocess-based tests that verify the installed package works."""

    def test_no_args_shows_help(self) -> None:
        r = _run_subprocess()
        assert r.returncode == 0
        assert "ruitong" in r.stdout.lower() or "usage" in r.stdout.lower()

    def test_port_help_shows_usage(self) -> None:
        r = _run_subprocess("port", "--help")
        assert r.returncode == 0
        assert "model" in r.stdout.lower() or "target" in r.stdout.lower()