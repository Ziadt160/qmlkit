"""Shot-budget arithmetic.

Simulator-only means shots are opt-in, not mandatory. When they are on, every
sampled number should be reportable with its uncertainty — that is what makes
"would this survive on a real device?" an answerable question rather than a guess.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "standard_error",
    "variance",
    "shots_for_precision",
    "p0_from_z",
    "z_from_p0",
    "runtime_estimate",
]


def variance(z: float) -> float:
    """Single-shot variance of a +-1 observable with mean ``z``: ``1 - z**2``."""
    return float(1.0 - np.clip(z, -1.0, 1.0) ** 2)


def standard_error(z: float, shots: int) -> float:
    """Standard error of an expectation estimated from ``shots`` samples."""
    if shots <= 0:
        raise ValueError("shots must be positive")
    return float(np.sqrt(variance(z) / shots))


def shots_for_precision(eps: float, z: float = 0.0) -> int:
    """Shots needed to reach standard error ``eps`` — the ``1/eps**2`` price."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    return int(np.ceil(variance(z) / eps**2))


def p0_from_z(z: float) -> float:
    """P(0) from <Z>."""
    return float((1.0 + z) / 2.0)


def z_from_p0(p0: float) -> float:
    """<Z> from P(0)."""
    return float(2.0 * p0 - 1.0)


def runtime_estimate(shots: int, rate_hz: float) -> float:
    """Wall-clock seconds for a shot budget at a given sampling rate."""
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")
    return float(shots / rate_hz)
