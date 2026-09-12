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


def standard_error(z: float, shots: int, scale: float = 1.0) -> float:
    """Standard error of a **single** Pauli term's expectation over ``shots`` samples.

    ``scale`` is the term's coefficient: ``(cP)^2 = c^2 I``, so the variance is
    ``c^2 - z^2`` and the error scales with ``|c|``.

    This formula is only correct for one term. A sum needs ``<O^2>``, which is a
    different measurement and not recoverable from ``<O>`` — feeding a sum in here
    gives an error bar that is too tight, too loose, or exactly zero once ``|<O>|``
    reaches ``|c|``. :func:`~qmlkit.core.execute.expectation` checks the observable
    before calling this.
    """
    if shots <= 0:
        raise ValueError("shots must be positive")
    magnitude = abs(float(scale))
    spread = max(magnitude**2 - float(z) ** 2, 0.0)
    return float(np.sqrt(spread / shots))


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
