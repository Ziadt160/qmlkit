r"""SPSA — a gradient estimate in two evaluations, whatever ``P`` is.

Parameter-shift costs ``2P`` circuits. SPSA perturbs *every* parameter at once
along a random :math:`\pm 1` direction and costs two, forever:

.. math::  \hat{g} = \frac{f(\theta + c\Delta) - f(\theta - c\Delta)}{2c}\,\Delta^{-1}

The estimate is noisy but **unbiased in expectation**, and stochastic optimisers
tolerate that well — which is why it is the standard answer once ``P`` gets large
enough that ``2P`` circuits per step stops being affordable.

The decay schedules are Spall's. ``A``, the stability constant, is the one people
leave out: without it the first steps are far too large, and the run diverges
before the schedule has a chance to settle it. Roughly 10 % of the planned
iteration count is the usual choice.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["spsa_grad", "spsa_step", "SPSASchedule", "minimize_spsa"]

LossFn = Callable[[np.ndarray], float]


def spsa_grad(
    f: LossFn,
    theta: ArrayLike,
    c: float = 0.1,
    n_avg: int = 1,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """A stochastic gradient estimate from ``2 * n_avg`` evaluations.

    ``n_avg`` averages several random directions, trading evaluations for variance
    — still constant in ``P``.
    """
    arr = np.asarray(theta, dtype=float).ravel()
    if c <= 0:
        raise ValueError("c must be positive")
    if n_avg < 1:
        raise ValueError("n_avg must be at least 1")
    generator = rng if rng is not None else np.random.default_rng(seed)

    total = np.zeros_like(arr)
    for _ in range(n_avg):
        delta = generator.choice([-1.0, 1.0], size=arr.shape)
        plus = float(f(arr + c * delta))
        minus = float(f(arr - c * delta))
        total += (plus - minus) / (2.0 * c) * delta  # delta^-1 == delta for +-1
    return total / n_avg


class SPSASchedule:
    """Spall's decay schedules for the step size and the perturbation size."""

    def __init__(
        self,
        a: float = 0.2,
        c: float = 0.1,
        A: float | None = None,
        alpha: float = 0.602,
        gamma: float = 0.101,
        n_iterations: int = 100,
    ) -> None:
        self.a = a
        self.c = c
        # the stability constant the lecture's version omits
        self.A = A if A is not None else max(1.0, 0.1 * n_iterations)
        self.alpha = alpha
        self.gamma = gamma

    def step_size(self, k: int) -> float:
        return self.a / (k + 1 + self.A) ** self.alpha

    def perturbation(self, k: int) -> float:
        return self.c / (k + 1) ** self.gamma

    def __repr__(self) -> str:
        return (
            f"SPSASchedule(a={self.a}, c={self.c}, A={self.A}, "
            f"alpha={self.alpha}, gamma={self.gamma})"
        )


def spsa_step(
    f: LossFn,
    theta: ArrayLike,
    k: int,
    schedule: SPSASchedule | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One SPSA update at iteration ``k``."""
    sched = schedule or SPSASchedule()
    arr = np.asarray(theta, dtype=float).ravel()
    g = spsa_grad(f, arr, c=sched.perturbation(k), rng=rng)
    return arr - sched.step_size(k) * g


def minimize_spsa(
    f: LossFn,
    theta0: ArrayLike,
    n_iterations: int = 100,
    schedule: SPSASchedule | None = None,
    seed: int | None = None,
    callback: Callable[[int, np.ndarray, float], None] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Minimise ``f`` with SPSA. Returns the final parameters and the loss history.

    Two evaluations per iteration regardless of how many parameters there are.
    """
    sched = schedule or SPSASchedule(n_iterations=n_iterations)
    rng = np.random.default_rng(seed)
    theta = np.asarray(theta0, dtype=float).ravel().copy()
    history: list[float] = []
    for k in range(n_iterations):
        value = float(f(theta))
        history.append(value)
        if callback is not None:
            callback(k, theta, value)
        theta = spsa_step(f, theta, k, sched, rng)
    history.append(float(f(theta)))
    return theta, history
