"""Ruitong CLI — equivalence harness entry point.

Usage:
    ruitong port <model> --target ascend [--output report.json]
    ruitong port <model> --target cuda    [--output report.json]
    ruitong port <model>                   # auto — compare both backends
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import NoReturn

from .backends.fake import FakeAscend, FakeCuda
from .equivalence.runner import EquivalenceRunner, EquivalenceReport


def _print_error(msg: str) -> NoReturn:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ruitong",
        description="Ruitong Bridge — CANN/CUDA equivalence harness",
    )
    sub = parser.add_subparsers(dest="command")

    port_parser = sub.add_parser(
        "port",
        help="Run equivalence comparison between CUDA and Ascend backends",
    )
    port_parser.add_argument(
        "model",
        help="Model name to compare (e.g. Qwen3-8B)",
    )
    port_parser.add_argument(
        "--target",
        choices=["cuda", "ascend", "auto"],
        default="auto",
        help="Target backend(s): cuda, ascend, or auto (default: auto)",
    )
    port_parser.add_argument(
        "--output",
        default=None,
        help="Write JSON report to this file path",
    )
    return parser


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
        _print_error(f"Unknown target: {target}")  # unreachable


def _default_prompts(model: str) -> list[str]:
    """Return a small set of default prompts for equivalence testing."""
    return [
        f"What is {model}? Answer in one sentence.",
        "Explain the concept of transformers.",
        "Write a short poem about AI.",
    ]


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "port":
        _run_port(args)


def _run_port(args: argparse.Namespace) -> None:
    model = args.model
    target = args.target

    runner, name_a, name_b = _make_runner(model, target)
    prompts = _default_prompts(model)

    report: EquivalenceReport = asyncio.run(runner.run(model, prompts))

    # Print summary to stdout
    print(f"=== Ruitong Equivalence Report ===")
    print(f"Model:  {report.model}")
    print(f"Mode:   {report.mode}")
    print(f"Prompt: {report.total_prompts}")
    if name_a != name_b:
        print(f"Compare: {name_a} vs {name_b}")
    else:
        print(f"Target:  {name_a}")
    print(f"Passed: {'YES' if report.passed else 'NO'}")
    print()

    m = report.metrics
    if m.get("cosine_similarity") is not None:
        print(f"Cosine similarity:    {m['cosine_similarity']:.6f}")
    if m.get("max_absolute_difference") is not None:
        print(f"Max abs diff:         {m['max_absolute_difference']:.6f}")
    if m.get("top1_agreement") is not None:
        print(f"Top-1 agreement:      {m['top1_agreement']:.6f}")
    if m.get("top5_set_agreement") is not None:
        print(f"Top-5 set agreement:  {m['top5_set_agreement']:.6f}")
    if m.get("response_parity") is not None:
        print(f"Response parity:      {m['response_parity']:.6f}")
    print()

    if report.warnings:
        for w in report.warnings:
            print(f"  WARN: {w}", file=sys.stderr)

    # P1 #2 — exit with non-zero when comparison fails
    if not report.passed:
        sys.exit(1)

    # Write JSON report if requested
    if args.output:
        try:
            with open(args.output, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            print(f"Report written to {args.output}")
        except OSError as exc:
            _print_error(f"Cannot write report: {exc}")


if __name__ == "__main__":
    main()
