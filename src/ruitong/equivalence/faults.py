"""Fault injection for calibrating the equivalence gate.

The metrics in `metrics.py` say how *similar* two runs are. They do not say
whether the gate built on them can actually tell a good port from a broken one.
This module supplies the known-bad inputs needed to measure that.

Two uses:

1. **Calibration.** Establish what "equivalent" looks like numerically
   (`bfloat16_round`, run-to-run noise) versus what "broken" looks like
   (the fault injectors), and set thresholds from the gap between them.
2. **Regression.** A CI suite asserting every injected fault still fails the
   gate. A fault the gate stops detecting is a false PASS the product would
   ship — the worst failure mode for a tool whose output is a trust claim.

Pure stdlib on purpose: this must run anywhere, including a customer's machine
reproducing our sensitivity table.
"""

from __future__ import annotations

import math
import struct

Row = list[float]
Tensor = list[Row]

# ── Precision ─────────────────────────────────────────────────────────


def bfloat16_round(value: float) -> float:
    """Round a float to bfloat16 precision, round-half-to-even.

    bfloat16 keeps float32's 8-bit exponent but truncates the mantissa from 23
    bits to 7. Rounding a float32 through bf16 reproduces the *precision* loss
    a BF16 inference run incurs, which is the closest CPU-only analogue to the
    numeric difference between two hardware kernels.
    """
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    # Round-half-to-even at bit 16.
    bias = 0x7FFF + ((bits >> 16) & 1)
    bits = (bits + bias) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def to_bfloat16(tensor: Tensor) -> Tensor:
    """Apply bfloat16 rounding element-wise."""
    return [[bfloat16_round(v) for v in row] for row in tensor]


# ── Realistic inputs ──────────────────────────────────────────────────


def synthetic_logprobs(
    vocab: int = 512, positions: int = 8, seed: int = 7
) -> Tensor:
    """Generate log-softmax rows shaped like real LLM output.

    Real next-token logprobs are sharply peaked: a handful of plausible tokens
    near 0, then a long tail decaying steeply. Uniform random values would make
    every metric look better than it does in practice, so the shape matters.

    Deterministic for a given seed — a linear congruential generator is used
    rather than `random` so results do not depend on interpreter state.
    """
    state = seed
    tensor: Tensor = []
    for _ in range(positions):
        state = (1103515245 * state + 12345) % (2**31)
        logits: Row = []
        for rank in range(vocab):
            state = (1103515245 * state + 12345) % (2**31)
            jitter = (state / (2**31)) * 0.5
            # Steep decay: top few tokens dominate, tail falls away fast.
            logits.append(-0.35 * rank + jitter)
        # log_softmax, computed stably.
        peak = max(logits)
        total = sum(math.exp(x - peak) for x in logits)
        shift = peak + math.log(total)
        tensor.append([x - shift for x in logits])
    return tensor


# ── Fault injectors ───────────────────────────────────────────────────
#
# Each mimics a real porting failure. Every one MUST fail the gate.


def swap_tokens(tensor: Tensor, i: int = 0, j: int = 1) -> Tensor:
    """Swap two tokens' logprobs — mimics a transposed operator output."""
    out = [row[:] for row in tensor]
    for row in out:
        row[i], row[j] = row[j], row[i]
    return out


def scale_logprobs(tensor: Tensor, factor: float = 1.05) -> Tensor:
    """Scale every logprob — mimics a temperature or softmax scaling bug."""
    return [[v * factor for v in row] for row in tensor]


def truncate_vocab(tensor: Tensor, keep: int, floor: float = -30.0) -> Tensor:
    """Flatten the vocabulary tail — mimics wrong vocab-size handling."""
    return [
        [v if idx < keep else floor for idx, v in enumerate(row)]
        for row in tensor
    ]


def shift_positions(tensor: Tensor, by: int = 1) -> Tensor:
    """Rotate rows — mimics an off-by-one in KV-cache indexing."""
    if not tensor:
        return []
    by %= len(tensor)
    return [row[:] for row in tensor[by:] + tensor[:by]]


def corrupt_fraction(
    tensor: Tensor, fraction: float = 0.01, magnitude: float = 5.0, seed: int = 3
) -> Tensor:
    """Perturb a fraction of positions — mimics an intermittent kernel fault.

    This is the hardest fault to catch and the most dangerous in production:
    most positions agree, so any metric that *averages* over positions dilutes
    the damage. If the gate misses this, it will pass a flaky accelerator.
    """
    out = [row[:] for row in tensor]
    state = seed
    n = max(1, int(len(out) * fraction)) if out else 0
    for k in range(n):
        state = (1103515245 * state + 12345) % (2**31)
        idx = state % len(out)
        out[idx] = [v - magnitude * ((i % 3) - 1) for i, v in enumerate(out[idx])]
    return out
