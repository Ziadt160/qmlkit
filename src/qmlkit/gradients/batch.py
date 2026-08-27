r"""Gradients for a whole training batch, in one pass rather than one per sample.

A training step evaluates the same circuit *structure* at one parameter vector per
sample. The forward pass has been batched for a while; the backward pass had not, and
the backward pass is where the time goes — measured on a 4-qubit ``VQC`` at batch 128,
the forward pass was 13 ms of an 1170 ms step. **99% of training was still running one
sample at a time.**

Two routes, and the difference between them is the point of this module.

:func:`param_shift_grad_batch` is **backend-agnostic**. A shift rule only ever needs
the circuit *run* at shifted angles, so a whole batch's gradient is one big set of
evaluations — ``batch x 2P`` of them — with no inspection of the state at all. It goes
through :meth:`~qmlkit.core.backends.base.Backend.expectation_over_slots`, which every
backend has, so this works on NumPy, Qiskit, Cirq, SpinQit, a sampling-only device, and
anything registered later. On hardware it is exactly the batched submission a provider
wants: one job instead of ``batch x 2P`` blocking calls.

:func:`adjoint_grad_batch` is faster and simulator-only, because it reads the state.

Both are exact, and both are asserted equal to the per-sample functions they replace.

The shifts happen in **slot** space, not logical-parameter space, which is why the
batch primitives underneath are slot-based. A shift rule moves one *occurrence* of a
parameter and a weight-tied parameter has several; a batched routine written against
logical parameters cannot express that, and would silently compute a different
derivative for any circuit with shared weights.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.core.backends.base import Backend
from qmlkit.core.backends.numpy_backend import (
    _VECTORISED_D,
    _apply_batch,
    _batched_matrices,
)
from qmlkit.core.backends.registry import get_backend
from qmlkit.core.gates import gate_derivative
from qmlkit.core.ir import CircuitSpec, Op
from qmlkit.core.observables import Observable, Z, as_sum
from qmlkit.gradients.rules import rule_for_gate

__all__ = ["grad_batch", "param_shift_grad_batch", "adjoint_grad_batch"]

_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


# --------------------------------------------------------------------------- #
# the backend-agnostic route
# --------------------------------------------------------------------------- #
def param_shift_grad_batch(
    spec: CircuitSpec,
    thetas: ArrayLike,
    obs: Observable | None = None,
    shots: int | None = None,
    backend: Backend | str | None = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """Parameter-shift gradients for a batch, as **one** set of evaluations.

    Returns ``(batch, n_params)``.

    Every shifted circuit the whole batch needs is assembled first and handed to the
    backend in a single call, so a backend that can evaluate many circuits at once —
    or submit them as one job — gets to. Nothing here inspects a state, so this is
    valid on hardware and under sampling.

    Weight-tied parameters accumulate across occurrences and rescaled references get
    their chain-rule factor, exactly as the per-sample
    :func:`~qmlkit.gradients.parameter_shift.param_shift_grad` does.
    """
    obs = Z(0) if obs is None else obs
    be = get_backend(backend)
    angles = spec.bind_slots_batch(np.atleast_2d(np.asarray(thetas, dtype=float)))
    batch = angles.shape[0]
    slots = spec.slots()
    if not slots:
        return np.zeros((batch, spec.n_params), dtype=float)

    # ---- assemble every shifted row the batch needs, once ------------------ #
    blocks: list[npt.NDArray[Any]] = []
    plan: list[tuple[int, float]] = []  # (slot index, coefficient)
    needs_unshifted = False
    for i, slot in enumerate(slots):
        rule = rule_for_gate(slot.gate)
        for shift, coeff in zip(rule.shifts, rule.coeffs, strict=False):
            shifted = angles.copy()
            shifted[:, i] += shift
            blocks.append(shifted)
            plan.append((i, coeff))
        needs_unshifted = needs_unshifted or rule.needs_unshifted
    if needs_unshifted:
        blocks.append(angles)

    values = be.expectation_over_slots(
        spec, np.concatenate(blocks, axis=0), obs, shots, seed
    ).reshape(len(blocks), batch)

    # ---- fold the evaluations back into per-parameter gradients ------------ #
    grad = np.zeros((batch, spec.n_params), dtype=float)
    for k, (i, coeff) in enumerate(plan):
        ref = slots[i].ref
        grad[:, ref.index] += coeff * values[k] * ref.scale
    if needs_unshifted:
        base = values[-1]
        for slot in slots:
            rule = rule_for_gate(slot.gate)
            if rule.needs_unshifted:
                grad[:, slot.ref.index] += rule.unshifted_coeff * base * slot.ref.scale
    return grad


def param_shift_batch_cost(spec: CircuitSpec, batch: int) -> int:
    """Circuit evaluations a batched parameter-shift gradient submits, in total."""
    per_sample = sum(rule_for_gate(s.gate).n_evaluations for s in spec.slots())
    return per_sample * batch


# --------------------------------------------------------------------------- #
# the simulator route
# --------------------------------------------------------------------------- #
def _apply_observable_batch(
    state: npt.NDArray[Any], obs: Observable, n_qubits: int
) -> npt.NDArray[Any]:
    """``O|psi>`` for a stack of states, Pauli term by Pauli term."""
    total = np.zeros_like(state)
    for term in as_sum(obs).terms:
        out = state
        for q, p in term.paulis:
            if p == "I":
                continue
            out = np.tensordot(_PAULI[p], out, axes=([1], [q + 1]))
            out = np.moveaxis(out, 0, q + 1)
        total = total + term.coeff * out
    return total


def _dagger(matrices: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Conjugate transpose, whether the array is one matrix or a stack of them."""
    if matrices.ndim == 2:
        return matrices.conj().T
    return matrices.conj().transpose(0, 2, 1)


def _batched_derivatives(op: Op, columns: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """``dU/dtheta`` per row, in closed form where one is registered.

    Profiling the batched adjoint put four times as much time in building these one
    row at a time as in the contraction they feed, which is why the vectorised table
    exists. A gate outside it still works, one build per row.
    """
    builder = _VECTORISED_D.get(op.gate)
    if builder is not None and columns.shape[1] == 1:
        return builder(columns[:, 0])
    return np.stack([gate_derivative(op.gate, tuple(row)) for row in columns])


def adjoint_grad_batch(
    spec: CircuitSpec,
    thetas: ArrayLike,
    obs: Observable | None = None,
    backend: Backend | str | None = None,
) -> npt.NDArray[Any]:
    """Adjoint gradients for a batch in one forward and one backward sweep.

    Returns ``(batch, n_params)``. Exact, and independent of ``P`` in cost, but it
    needs the statevector — so it is simulator-only, and
    :func:`param_shift_grad_batch` is what a device uses.

    The sweep is the one in :mod:`~qmlkit.gradients.adjoint` with the batch carried as
    a leading axis: every gate is undone for the whole stack at once, and each
    parameter's contribution comes out as one inner product per row.
    """
    obs = Z(0) if obs is None else obs
    be = get_backend(backend)
    if not be.supports_statevector:
        raise ValueError(
            f"the {be.name!r} backend has no statevector, so it cannot differentiate by "
            'the adjoint method; use method="parameter-shift", which batches too'
        )

    values = np.atleast_2d(np.asarray(thetas, dtype=float))
    angles = spec.bind_slots_batch(values)
    batch, n = angles.shape[0], spec.n_qubits
    slots = spec.slots()

    psi = be.statevector_batch_slots(spec, angles).reshape((batch,) + (2,) * n)
    lam = _apply_observable_batch(psi, obs, n)
    grad = np.zeros((batch, spec.n_params), dtype=float)

    # walk the slots backwards in step with the ops, the way the scalar version does
    cursor = len(slots)
    for op_index in range(len(spec.ops) - 1, -1, -1):
        op = spec.ops[op_index]
        first = cursor
        while first > 0 and slots[first - 1].op_index == op_index:
            first -= 1
        columns = angles[:, first:cursor] if first < cursor else None
        cursor = first

        u = _batched_matrices(op, columns)
        psi = _apply_batch(psi, _dagger(u), op.qubits)  # the state before this gate

        if columns is not None and columns.shape[1] == 1:
            du = _batched_derivatives(op, columns)
            mu = _apply_batch(psi, du, op.qubits)
            overlap = np.einsum(
                "bi,bi->b", lam.reshape(batch, -1).conj(), mu.reshape(batch, -1)
            )
            ref = slots[first].ref
            grad[:, ref.index] += 2.0 * np.real(overlap) * ref.scale
        elif columns is not None and columns.shape[1] > 1:  # pragma: no cover
            raise NotImplementedError(
                f"gate {op.gate!r} has {columns.shape[1]} parameters; batched adjoint "
                'handles one per gate. Use method="parameter-shift".'
            )

        lam = _apply_batch(lam, _dagger(u), op.qubits)
    return grad


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def grad_batch(
    spec: CircuitSpec,
    thetas: ArrayLike,
    obs: Observable | None = None,
    method: str = "auto",
    shots: int | None = None,
    backend: Backend | str | None = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """``(batch, n_params)`` gradients, by whichever route fits the backend.

    ``"auto"`` picks adjoint on an exact simulator and parameter-shift otherwise —
    the same rule :func:`~qmlkit.gradients.dispatch.choose_method` uses for one
    sample, since sampling puts the statevector out of reach by definition.

        >>> import numpy as np, qmlkit as qk
        >>> a = qk.hardware_efficient(3, 2)
        >>> qk.grad_batch(a.build(), np.zeros((5, a.n_params)), qk.Z(0)).shape
        (5, 12)
    """
    if method == "auto":
        from qmlkit.gradients.adjoint import supports_adjoint

        method = (
            "adjoint"
            if shots is None and supports_adjoint(spec, backend)
            else "parameter-shift"
        )
    if method == "adjoint":
        if shots is not None:
            raise ValueError("adjoint differentiation is exact; it cannot take shots")
        return adjoint_grad_batch(spec, thetas, obs, backend)
    if method == "parameter-shift":
        return param_shift_grad_batch(spec, thetas, obs, shots, backend, seed)

    from qmlkit.utils.errors import unknown

    raise unknown(
        "batched gradient method",
        method,
        ("auto", "adjoint", "parameter-shift"),
        hint=" Other methods have no batched form yet; loop over qk.grad for those.",
    )
