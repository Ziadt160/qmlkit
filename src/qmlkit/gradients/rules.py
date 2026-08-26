r"""Shift rules, *derived* rather than remembered.

A circuit expectation as a function of one gate angle is a finite Fourier series
whose frequencies are the unique positive gaps between the eigenvalues of that
gate's generator:

.. math::  f(\theta) = a_0 + \sum_{k} a_k \cos(\omega_k \theta) + b_k \sin(\omega_k \theta)

We want coefficients :math:`c_i` and shifts :math:`s_i` with
:math:`\sum_i c_i f(\theta + s_i) = f'(\theta)` for **every** such :math:`f`.
Expanding and matching the :math:`\cos(\omega\theta)` and :math:`\sin(\omega\theta)`
terms gives, for each frequency :math:`\omega`:

.. math::  \sum_i c_i \cos(\omega s_i) = 0, \qquad \sum_i c_i \sin(\omega s_i) = \omega

Choosing antisymmetric shifts :math:`\{+s_1, -s_1, \dots\}` with antisymmetric
coefficients satisfies the cosine equation and :math:`\sum_i c_i = 0` identically,
leaving an :math:`R \times R` linear system in the positive half. We solve it.

Deriving the rule instead of hardcoding constants means a new gate needs only its
``frequencies`` declared — no new gradient code, and no chance of a transcribed
constant being subtly wrong. :func:`four_term_rule` reproduces the textbook
controlled-rotation constants, and the test suite asserts exactly that.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.gates import get_gate

__all__ = [
    "ShiftRule",
    "general_shift_rule",
    "two_term_rule",
    "four_term_rule",
    "rule_for_frequencies",
    "rule_for_gate",
]


@dataclass(frozen=True)
class ShiftRule:
    """``f'(theta) = sum_i coeffs[i] * f(theta + shifts[i])`` (+ an unshifted term)."""

    shifts: tuple[float, ...]
    coeffs: tuple[float, ...]
    #: some rules (notably second-derivative ones) also need f(theta) itself
    unshifted_coeff: float = 0.0

    @property
    def needs_unshifted(self) -> bool:
        return self.unshifted_coeff != 0.0

    @property
    def n_evaluations(self) -> int:
        return len(self.shifts) + (1 if self.needs_unshifted else 0)

    def __repr__(self) -> str:
        pairs = ", ".join(
            f"{c:+.4f}@{s:+.4f}" for s, c in zip(self.shifts, self.coeffs, strict=False)
        )
        return f"ShiftRule({pairs})"


def _positive_shifts(frequencies: Sequence[float]) -> npt.NDArray[Any]:
    """Equidistant positive shifts: (2j-1)*pi / (2*max_freq), j = 1..R."""
    r = len(frequencies)
    scale = float(max(frequencies))
    return np.array([(2 * j - 1) * np.pi / (2 * scale) for j in range(1, r + 1)], dtype=float)


def general_shift_rule(
    frequencies: Sequence[float], shifts: Sequence[float] | None = None
) -> ShiftRule:
    """Build the exact shift rule for a generator with these frequencies."""
    freqs = tuple(sorted({float(f) for f in frequencies}))
    if not freqs:
        raise ValueError("a shift rule needs at least one generator frequency")
    if any(f <= 0 for f in freqs):
        raise ValueError(f"frequencies must be positive, got {freqs}")

    pos = np.asarray(shifts, dtype=float) if shifts is not None else _positive_shifts(freqs)
    if pos.size != len(freqs):
        raise ValueError(f"need exactly {len(freqs)} positive shifts, got {pos.size}")

    # 2 * sum_j c_j sin(w_k s_j) = w_k
    a = 2.0 * np.sin(np.outer(np.asarray(freqs), pos))
    if abs(np.linalg.det(a)) < 1e-12:
        raise ValueError(
            f"degenerate shift choice {pos.tolist()} for frequencies {list(freqs)}; "
            "pick different shifts"
        )
    c = np.linalg.solve(a, np.asarray(freqs, dtype=float))

    out_shifts: list[float] = []
    out_coeffs: list[float] = []
    for s, ci in zip(pos, c, strict=False):
        out_shifts += [float(s), float(-s)]
        out_coeffs += [float(ci), float(-ci)]
    return ShiftRule(tuple(out_shifts), tuple(out_coeffs))


@cache
def rule_for_frequencies(frequencies: tuple[float, ...]) -> ShiftRule:
    """Cached :func:`general_shift_rule` — rules are pure functions of the spectrum."""
    return general_shift_rule(frequencies)


def two_term_rule() -> ShiftRule:
    """The familiar Pauli-rotation rule: shifts +-pi/2, coefficients +-1/2."""
    return rule_for_frequencies((1.0,))


def four_term_rule() -> ShiftRule:
    """Controlled rotations: generator eigenvalues {0, 0, +-1/2} => frequencies {1/2, 1}."""
    return rule_for_frequencies((0.5, 1.0))


def rule_for_gate(gate: str) -> ShiftRule:
    """Look the rule up from the gate's declared generator frequencies.

    This per-gate lookup is the whole point: a circuit mixing ``ry`` with ``crz``
    needs two *different* rules, and applying one uniform rule to both returns a
    plausible, wrong gradient with no error raised.
    """
    g = get_gate(gate)
    if not g.is_parametric:
        raise ValueError(f"gate {gate!r} has no parameters to differentiate")
    if not g.is_differentiable:
        raise ValueError(
            f"gate {gate!r} declares no generator frequencies, so no exact shift rule "
            "can be derived. Register it with frequencies=(...) to make it differentiable."
        )
    return rule_for_frequencies(g.frequencies)


def second_derivative_rule(frequencies: tuple[float, ...] = (1.0,)) -> ShiftRule:
    """Diagonal Hessian rule. For one frequency: f'' = (f(theta+pi) - f(theta)) / 2."""
    if frequencies != (1.0,):
        raise NotImplementedError(
            "the closed-form second-derivative rule is implemented for single-frequency "
            "(Pauli) generators only; apply the first-order rule twice otherwise"
        )
    return ShiftRule(shifts=(np.pi,), coeffs=(0.5,), unshifted_coeff=-0.5)
