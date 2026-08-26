r"""Adjoint differentiation — exact gradients in one backward pass.

Parameter-shift needs ``2P`` circuit evaluations because a real device can only be
*run*, never inspected. On a simulator the state is right there, so the whole
gradient comes out of a single forward and a single backward sweep — independent
of ``P``.

The sweep. With :math:`|\psi_j\rangle = U_j \cdots U_1 |0\rangle` and
:math:`E = \langle\psi_n| O |\psi_n\rangle`,

.. math::  \frac{\partial E}{\partial\theta_k}
           = 2\,\mathrm{Re}\,\langle\lambda_k|\,\partial_k U_k\,|\psi_{k-1}\rangle,
           \qquad \lambda_k = (U_n\cdots U_{k+1})^\dagger O |\psi_n\rangle

so walking backwards and undoing one gate at a time keeps both states current at
``O(1)`` extra memory.

This is exact — the gate derivatives are closed-form, not finite differences — but
it is **simulator-only**, because no device will hand you an amplitude. Since
``0.x`` is simulator-only anyway, it is the right default for training;
parameter-shift remains what the course teaches, what validates this, and what
keeps the library hardware-ready.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend
from qmlkit.core.backends.numpy_backend import _apply
from qmlkit.core.backends.registry import get_backend
from qmlkit.core.gates import gate_derivative, gate_matrix
from qmlkit.core.ir import CircuitSpec, ParamRef
from qmlkit.core.observables import Observable, PauliString, Z, as_sum

__all__ = ["adjoint_grad", "supports_adjoint"]

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def supports_adjoint(spec: CircuitSpec, backend: Backend | str | None = None) -> bool:
    """True if every parameterised gate in ``spec`` has a closed-form derivative."""
    from qmlkit.core.gates import get_gate

    be = get_backend(backend)
    if not be.supports_statevector:
        return False
    return all(get_gate(s.gate).has_derivative for s in spec.slots())


def _apply_observable(state: npt.NDArray[Any], obs: Observable, n_qubits: int) -> npt.NDArray[Any]:
    """``O|psi>`` for a Pauli sum, in tensor shape."""
    total = np.zeros_like(state)
    for term in as_sum(obs).terms:
        out = state
        for q, p in term.paulis:
            if p == "I":
                continue
            out = np.tensordot(_PAULI[p], out, axes=([1], [q]))
            out = np.moveaxis(out, 0, q)
        total = total + term.coeff * out
    return total


def adjoint_grad(
    spec: CircuitSpec,
    theta: npt.NDArray[Any],
    obs: Observable | None = None,
    backend: Backend | str | None = None,
) -> npt.NDArray[Any]:
    """Exact gradient of ``<obs>`` with respect to the logical parameter vector.

    One forward pass and one backward pass, whatever ``P`` is. Weight-tied
    parameters accumulate across their occurrences, exactly as parameter-shift does.
    """
    obs = Z(0) if obs is None else obs
    be = get_backend(backend)
    if not be.supports_statevector:
        raise ValueError(
            f"the {be.name!r} backend has no statevector, so it cannot differentiate by "
            'the adjoint method; use grad_method="parameter-shift"'
        )

    theta = np.asarray(theta, dtype=float).ravel()
    slots = spec.slots()
    slot_angles = spec.bind_slots(theta)
    n = spec.n_qubits
    shape = (2,) * n

    # forward: run the bound circuit once
    psi = be.statevector(spec.with_slot_angles(slot_angles)).reshape(shape)
    lam = _apply_observable(psi, obs, n)

    grad = np.zeros(spec.n_params, dtype=float)
    slot_of_op = {s.op_index: i for i, s in enumerate(slots)}

    # backward: undo one gate at a time, reading off each parameter's contribution
    cursor = len(slot_angles)
    for op_index in range(len(spec.ops) - 1, -1, -1):
        op = spec.ops[op_index]
        slot_i = slot_of_op.get(op_index)
        if slot_i is not None:
            cursor -= 1
            angles = [float(slot_angles[cursor])]
        else:
            angles = [float(p) for p in op.params if not isinstance(p, ParamRef)]

        u = gate_matrix(op.gate, tuple(angles))
        psi = _apply(psi, u.conj().T, op.qubits)  # psi is now the state before this gate

        if slot_i is not None:
            du = gate_derivative(op.gate, tuple(angles))
            mu = _apply(psi, du, op.qubits)
            contribution = 2.0 * float(np.real(np.vdot(lam, mu)))
            ref = slots[slot_i].ref
            grad[ref.index] += contribution * ref.scale  # += ties occurrences together

        lam = _apply(lam, u.conj().T, op.qubits)

    return grad


def adjoint_grad_terms(
    spec: CircuitSpec,
    theta: npt.NDArray[Any],
    obs: Observable,
    backend: Backend | str | None = None,
) -> dict[PauliString, npt.NDArray[Any]]:  # pragma: no cover - convenience for diagnostics
    """Per-term gradients, for diagnosing which observable term drives a parameter."""
    return {t: adjoint_grad(spec, theta, t, backend) for t in as_sum(obs).terms}
