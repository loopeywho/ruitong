"""Ruitong CLI — equivalence harness entry point.

Usage:
    ruitong port <model> --cuda-endpoint URL --ascend-endpoint URL [--output report.json]
    ruitong port <model> --target cuda   # reverse direction: Ascend is the reference
    ruitong port <model>                 # no endpoints -> SYNTHETIC dry run, cannot pass

Exit codes (CI contract):
    0  equivalence gate passed
    1  gate failed — the port is not equivalent
    2  could not run — backend unreachable, no comparisons made, or write error

`2` is deliberately distinct from `1`: "the port is broken" and "we could not
tell" demand opposite responses, and collapsing them lets an infrastructure
outage read as a clean bill of health.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, NoReturn

from .backends.base import Backend
from .backends.fake import FakeAscend, FakeCuda
from .backends.vllm_http import VllmHttpBackend
from .equivalence.runner import EquivalenceReport, EquivalenceRunner

EXIT_PASS = 0
EXIT_GATE_FAILED = 1
EXIT_CANNOT_RUN = 2


def _fail(msg: str) -> NoReturn:
    """Exit 2 — we could not run the comparison. Not a gate failure."""
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(EXIT_CANNOT_RUN)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruitong",
        description="Ruitong Bridge — CANN/CUDA equivalence harness",
    )
    sub = parser.add_subparsers(dest="command")

    port_parser = sub.add_parser(
        "port",
        help="Compare a model across CUDA and Ascend backends",
    )
    port_parser.add_argument("model", help="Model name (e.g. Qwen3-8B)")
    port_parser.add_argument(
        "--target",
        choices=["ascend", "cuda"],
        default="ascend",
        help=(
            "Backend being ported TO. The other side is the reference. "
            "Default: ascend (CUDA reference -> Ascend target)."
        ),
    )
    port_parser.add_argument(
        "--cuda-endpoint",
        default=None,
        help="Base URL of the CUDA vLLM server (e.g. http://gpu-host:8000)",
    )
    port_parser.add_argument(
        "--ascend-endpoint",
        default=None,
        help="Base URL of the Ascend vllm-ascend server",
    )
    # Vendor-neutral form. The harness only ever sees logprobs over an
    # OpenAI-compatible endpoint, so it has no idea what silicon is behind it —
    # ROCm, Gaudi, Trainium, MetaX and Moore Threads all work unchanged. This
    # is what lets one tool serve both the China market (Ascend) and the
    # Western market (AMD/Intel/AWS) without a second codebase.
    port_parser.add_argument(
        "--reference",
        default=None,
        metavar="NAME=URL",
        help="Reference backend, e.g. cuda=http://gpu:8000 (overrides --cuda-endpoint)",
    )
    port_parser.add_argument(
        "--candidate",
        default=None,
        metavar="NAME=URL",
        help="Candidate backend, e.g. rocm=http://mi300:8000 (overrides --ascend-endpoint)",
    )
    port_parser.add_argument(
        "--api-key",
        default=os.environ.get("RUITONG_API_KEY"),
        help=(
            "Bearer token sent to both endpoints. Defaults to $RUITONG_API_KEY. "
            "Pass via the environment, not the command line, so it stays out of "
            "shell history and process listings."
        ),
    )
    # The two sides are, by definition, different clusters — a CUDA host and an
    # Ascend host are never one deployment and rarely share a credential. A
    # single shared key only ever works in a demo.
    port_parser.add_argument(
        "--reference-api-key",
        default=os.environ.get("RUITONG_REF_API_KEY"),
        help="Bearer token for the reference endpoint only. Defaults to "
        "$RUITONG_REF_API_KEY, then --api-key.",
    )
    port_parser.add_argument(
        "--candidate-api-key",
        default=os.environ.get("RUITONG_CAND_API_KEY"),
        help="Bearer token for the candidate endpoint only. Defaults to "
        "$RUITONG_CAND_API_KEY, then --api-key.",
    )
    port_parser.add_argument(
        "--output", default=None, help="Write the JSON report to this path"
    )
    return parser


def _parse_endpoint(spec: str, flag: str) -> tuple[str, str]:
    """Parse a NAME=URL backend spec."""
    name, separator, url = spec.partition("=")
    if not separator or not name.strip() or not url.strip():
        _fail(f"{flag} expects NAME=URL (e.g. rocm=http://host:8000), got {spec!r}")
    return name.strip(), url.strip()


def _make_backends(
    model: str,
    cuda_endpoint: str | None,
    ascend_endpoint: str | None,
    api_key: str | None = None,
    *,
    reference_api_key: str | None = None,
    candidate_api_key: str | None = None,
) -> tuple[Backend, Backend, bool]:
    """Return (reference, candidate, synthetic).

    Endpoints are used when supplied. When either is missing we fall back to
    fixtures and report `synthetic=True` — such a run can never pass, because
    a report computed from fixtures certifies nothing about real hardware.
    """
    synthetic = cuda_endpoint is None or ascend_endpoint is None

    cuda: Backend = (
        VllmHttpBackend(
            name="cuda",
            base_url=cuda_endpoint,
            api_key=reference_api_key or api_key,
        )
        if cuda_endpoint
        else FakeCuda(model_ids=[model])
    )
    ascend: Backend = (
        VllmHttpBackend(
            name="ascend",
            base_url=ascend_endpoint,
            api_key=candidate_api_key or api_key,
        )
        if ascend_endpoint
        else FakeAscend(model_ids=[model])
    )
    return cuda, ascend, synthetic


def _default_prompts(model: str) -> list[str]:
    """Default probe prompts.

    Deliberately varied — the fixture backends previously ignored the prompt
    entirely, so three prompts measured the same thing once.
    """
    return [
        f"What is {model}? Answer in one sentence.",
        "Explain the concept of transformers.",
        "Write a short poem about AI.",
    ]


def _print_summary(
    report: EquivalenceReport,
    reference: str,
    target: str,
    synthetic: bool,
    passed: bool,
) -> None:
    print("=== Ruitong Equivalence Report ===")
    print(f"Model:     {report.model}")
    print(f"Mode:      {report.mode}")
    print(f"Prompts:   {report.total_prompts}")
    print(f"Compare:   {reference} (reference) vs {target} (target)")
    if synthetic:
        print("Data:      SYNTHETIC — fixture backends, no hardware contacted")
    print(f"Passed:    {'YES' if passed else 'NO'}")
    print()

    # Gated metrics first, and marked. Without this the summary showed only
    # cosine and full-vocab max-abs-diff — neither of which gates — so a
    # reader could not tell from the output why a run passed or failed.
    labels = {
        "token_matched_prob_diff": "Token-matched prob diff [GATE]",
        "probability_mass_delta": "Prob-mass delta         [GATE]",
        "top1_agreement": "Top-1 agreement         [GATE]",
        "top5_set_agreement": "Top-5 set agreement     (reported)",
        "topk_max_abs_diff": "Top-k max abs diff      (reported)",
        "cosine_similarity": "Cosine similarity       (reported)",
        "max_absolute_difference": "Max abs diff            (reported)",
        "response_parity": "Response parity         (reported)",
    }
    for key, label in labels.items():
        value = report.metrics.get(key)
        if value is not None:
            print(f"{label + ':':<38}{value:.9f}")
    print()


def _write_report(path: str, payload: dict[str, Any]) -> None:
    try:
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Report written to {path}")
    except OSError as exc:
        _fail(f"Cannot write report: {exc}")


def _run_port(args: argparse.Namespace) -> int:
    model: str = args.model

    reference: Backend
    target: Backend

    if args.reference or args.candidate:
        # Vendor-neutral path: any two OpenAI-compatible endpoints.
        if not (args.reference and args.candidate):
            _fail("--reference and --candidate must be supplied together")
        ref_name, ref_url = _parse_endpoint(args.reference, "--reference")
        cand_name, cand_url = _parse_endpoint(args.candidate, "--candidate")
        if ref_name == cand_name:
            _fail(
                f"--reference and --candidate are both named {ref_name!r}; "
                "a backend compared with itself always agrees and certifies nothing"
            )
        # Distinct names are not enough: two labels pointing at the SAME URL
        # is still a backend compared with itself, and it always passes.
        if ref_url.rstrip("/") == cand_url.rstrip("/"):
            _fail(
                f"--reference and --candidate both point at {ref_url!r}; "
                "comparing an endpoint with itself certifies nothing"
            )
        reference = VllmHttpBackend(
            name=ref_name,
            base_url=ref_url,
            api_key=args.reference_api_key or args.api_key,
        )
        target = VllmHttpBackend(
            name=cand_name,
            base_url=cand_url,
            api_key=args.candidate_api_key or args.api_key,
        )
        synthetic = False
    else:
        cuda, ascend, synthetic = _make_backends(
            model,
            args.cuda_endpoint,
            args.ascend_endpoint,
            args.api_key,
            reference_api_key=args.reference_api_key,
            candidate_api_key=args.candidate_api_key,
        )
        # The target is the side being ported TO; the other is the reference.
        # Self-comparison is structurally impossible here — a backend compared
        # with itself always agrees, so it certifies nothing.
        if args.target == "ascend":
            reference, target = cuda, ascend
        else:
            reference, target = ascend, cuda

    runner = EquivalenceRunner(reference, target)

    try:
        report: EquivalenceReport = asyncio.run(runner.run(model, _default_prompts(model)))
    except Exception as exc:  # noqa: BLE001 — surfaced as "could not run", not a gate failure
        _fail(f"Comparison could not run: {type(exc).__name__}: {exc}")

    # Coverage gate. Any errored prompt means we did not measure what we
    # claimed to measure, so the run cannot pass. If NOTHING was compared the
    # answer is "could not run" (exit 2), never "the port failed" (exit 1) —
    # an outage and a broken port demand opposite responses.
    incomplete = report.errored_prompts > 0
    nothing_compared = report.compared_prompts == 0

    passed = report.passed and not synthetic and not incomplete

    payload = report.to_dict()
    payload["synthetic"] = synthetic
    payload["reference_backend"] = reference.name
    payload["target_backend"] = target.name
    if synthetic:
        payload["passed"] = False
        payload.setdefault("warnings", []).append(
            "SYNTHETIC RUN: no endpoints supplied, so fixture backends were used. "
            "This report certifies nothing about real hardware and cannot pass. "
            "Supply --cuda-endpoint and --ascend-endpoint."
        )
    if incomplete:
        payload["passed"] = False
        payload.setdefault("warnings", []).append(
            f"INCOMPLETE: {report.errored_prompts} of {report.total_prompts} "
            f"prompts errored, so only {report.compared_prompts} were actually "
            "compared. A verdict computed from a subset is not a verdict."
        )

    _print_summary(report, reference.name, target.name, synthetic, passed)

    for warning in payload.get("warnings", []):
        print(f"  WARN: {warning}", file=sys.stderr)

    # Write the report BEFORE deciding the exit code. Writing only on success
    # produces an archive containing exclusively passing runs, which reads as a
    # flawless track record — precisely when the failing artifact matters most.
    if args.output:
        _write_report(args.output, payload)

    if nothing_compared:
        print(
            "Error: no prompts could be compared — the backends were "
            "unreachable or returned nothing usable. This is NOT an "
            "equivalence failure; nothing was measured.",
            file=sys.stderr,
        )
        return EXIT_CANNOT_RUN

    return EXIT_PASS if passed else EXIT_GATE_FAILED


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns an exit code rather than calling sys.exit().

    Returning the code keeps `main` callable in-process, so the CLI's paths are
    measurable by coverage instead of only reachable via subprocess.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return EXIT_PASS

    if args.command == "port":
        return _run_port(args)

    parser.print_help()
    return EXIT_CANNOT_RUN


if __name__ == "__main__":
    sys.exit(main())
