"""The parameter-shift rule — exact gradients from measurements alone.

Two things here are easy to get wrong, and both fail *silently* — a plausible
number, no exception:

**Per-gate rules.** The shift rule is a property of the gate's generator, not of
the call. A circuit mixing ``ry`` (one frequency, two-term rule) with ``crz`` (two
frequencies, four-term rule) needs both. We look the rule up per slot.

**Per-occurrence shifting.** When one logical parameter drives several gates —
weight tying, as in a QCNN's shared convolution block — the derivative is the sum
over occurrences, each shifted *on its own*. Shifting them together computes
something else. The slot abstraction in :mod:`qmlkit.core.ir` makes this fall out
naturally: several slots simply map back to the same parameter index.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.registry import get_backend
from qmlkit.core.execute import BackendLike
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z
from qmlkit.gradients.rules import ShiftRule, rule_for_gate
from qmlkit.utils.errors import unknown

__all__ = [
    "param_shift_grad",
    "param_shift_grad_circuit",
    "grad_circuit_cost",
    "finite_diff_grad",
]

SlotFn = Callable[[npt.NDArray[Any]], float]


def param_shift_grad(
    f_slots: SlotFn,
    spec: CircuitSpec,
    theta: Sequence[float],
    rules: dict[int, ShiftRule] | None = None,
    f0: float | None = None,
) -> npt.NDArray[Any]:
    """Exact gradient of ``f`` with respect to the logical parameter vector.

    Parameters
    ----------
    f_slots
        Evaluates the circuit given **slot angles** — one angle per parameterised
        gate site. Taking slot angles rather than the logical vector is what makes
        per-occurrence shifting expressible at all.
    spec
        Supplies the slot map and each slot's gate (hence its shift rule).
    theta
        The logical parameter vector.
    rules
        Optional ``{slot_index: ShiftRule}`` override.
    f0
        The unshifted value, if you already have it. Unused by the standard
        two-term rule; required by rules with ``needs_unshifted``.
    """
    angles = spec.bind_slots(theta)
    slots = spec.slots()
    grad = np.zeros(spec.n_params, dtype=float)

    for i, slot in enumerate(slots):
        rule = (rules or {}).get(i) or rule_for_gate(slot.gate)
        total = 0.0
        for shift, coeff in zip(rule.shifts, rule.coeffs, strict=False):
            shifted = angles.copy()
            shifted[i] += shift
            total += coeff * f_slots(shifted)
        if rule.needs_unshifted:
            base = f0 if f0 is not None else f_slots(angles)
            total += rule.unshifted_coeff * base
        # Chain rule for a rescaled reference, and — crucially — ``+=``, which is
        # what sums a weight-tied parameter's several occurrences.
        grad[slot.ref.index] += total * slot.ref.scale

    return grad


def param_shift_grad_circuit(
    spec: CircuitSpec,
    theta: Sequence[float],
    obs: Observable | None = None,
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """Convenience wrapper: parameter-shift gradient of ``<obs>`` for a circuit."""
    obs = Z(0) if obs is None else obs
    be = get_backend(backend)

    def f_slots(angles: npt.NDArray[Any]) -> float:
        return be.expectation(spec.with_slot_angles(angles), obs, shots, seed)

    return param_shift_grad(f_slots, spec, theta)


def grad_circuit_cost(spec: CircuitSpec) -> int:
    """Circuit evaluations for one full parameter-shift gradient.

    Sums each slot's *real* rule cost. This is deliberately not a flat ``2P``:
    a controlled rotation costs four evaluations, not two, and a weight-tied
    parameter costs its rule once per occurrence.
    """
    return sum(rule_for_gate(s.gate).n_evaluations for s in spec.slots())


def finite_diff_grad(
    f: Callable[[npt.NDArray[Any]], float],
    theta: Sequence[float],
    eps: float = 1e-6,
    mode: str = "central",
) -> npt.NDArray[Any]:
    """Finite differences. For debugging and tests only — never for training.

    Carries an ``O(eps**2)`` bias and amplifies any sampling noise by ``1/eps``,
    which is why there is no good choice of ``eps`` on a noisy device.
    """
    arr = np.asarray(theta, dtype=float).ravel()
    grad = np.zeros_like(arr)
    base = f(arr) if mode == "forward" else None
    for k in range(arr.size):
        plus = arr.copy()
        plus[k] += eps
        if mode == "central":
            minus = arr.copy()
            minus[k] -= eps
            grad[k] = (f(plus) - f(minus)) / (2 * eps)
        elif mode == "forward":
            assert base is not None
            grad[k] = (f(plus) - base) / eps
        else:
            raise unknown("mode", mode, ("central", "forward"))
    return grad
