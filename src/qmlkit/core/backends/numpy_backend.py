"""Exact statevector simulator — the reference implementation.

This backend is the yardstick every other backend is measured against, and it is
what lets the library (and CI) run on Python 3.11+, where SpinQit has no wheel.
It is deliberately double-precision throughout: ``torch.autograd.gradcheck``
requires float64, and so does any honest claim that parameter-shift is *exact*.

Sampling, basis rotation and expectation semantics all live in
:class:`~qmlkit.core.backends.base.Backend`; this class supplies the statevector
and nothing else.

Convention: qubit 0 is the **most significant** bit of a bitstring, matching the
lecture notebooks where ``basis_encode([1, 0, 1])`` yields ``'101'``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend
from qmlkit.core.gates import gate_matrix
from qmlkit.core.ir import CircuitSpec, Op, ParamRef


def _apply(
    state: npt.NDArray[Any], matrix: npt.NDArray[Any], qubits: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Apply a k-qubit gate to the tensor-shaped state."""
    k = len(qubits)
    op = matrix.reshape((2,) * (2 * k))
    # contract the gate's input legs with the state's qubit axes
    state = np.tensordot(op, state, axes=(list(range(k, 2 * k)), list(qubits)))
    # tensordot puts the gate's output legs first; move them back into place
    return np.moveaxis(state, list(range(k)), list(qubits))


#: einsum subscripts. 'a' and 'b' are reserved for the batch and for spare output legs.
_LETTERS = "cdefghijklmnopqrstuvwxyz"


def _apply_batch(
    state: npt.NDArray[Any], matrices: npt.NDArray[Any], qubits: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Apply a k-qubit gate to a whole stack of states at once.

    ``state`` is ``(batch,) + (2,)*n``, so axis 0 is the batch and qubit ``q`` lives on
    axis ``q+1``. ``matrices`` is either one ``(d, d)`` matrix used for every sample —
    a gate with literal angles — or ``(batch, d, d)`` when the angle varies across the
    batch, which is the case for every encoding gate.

    The contraction is expressed directly in ``einsum`` subscripts rather than by
    moving the qubit axes to the end and reshaping. Both are correct; the reshape
    forces a copy of the whole stack twice per gate, and at 8 qubits that copying costs
    more than the arithmetic (measured: 41 ms against 27 ms for the same work).
    """
    n = state.ndim - 1
    if n > len(_LETTERS) - len(qubits):  # pragma: no cover - max_qubits bites first
        raise ValueError(f"{n} qubits is more than the batched contraction can subscript")
    state_subs = list(_LETTERS[:n])
    out_subs = list(state_subs)
    gate_subs = []
    for i, q in enumerate(qubits):
        fresh = _LETTERS[n + i]
        gate_subs.append(fresh)
        out_subs[q] = fresh
    inputs = "".join(state_subs[q] for q in qubits)
    legs = (2,) * (2 * len(qubits))

    if matrices.ndim == 2:
        spec = f"{''.join(gate_subs)}{inputs},b{''.join(state_subs)}->b{''.join(out_subs)}"
        shared: npt.NDArray[Any] = np.einsum(spec, matrices.reshape(legs), state)
        return shared
    spec = f"b{''.join(gate_subs)}{inputs},b{''.join(state_subs)}->b{''.join(out_subs)}"
    per_sample: npt.NDArray[Any] = np.einsum(spec, matrices.reshape((-1, *legs)), state)
    return per_sample


def _rx_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    c, s = np.cos(a / 2), np.sin(a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = c
    m[:, 0, 1] = m[:, 1, 0] = -1j * s
    return m


def _ry_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    c, s = np.cos(a / 2), np.sin(a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = c
    m[:, 0, 1] = -s
    m[:, 1, 0] = s
    return m


def _rz_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    e = np.exp(-1j * a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = e
    m[:, 1, 1] = np.conj(e)
    return m


def _phase_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = 1.0
    m[:, 1, 1] = np.exp(1j * a)
    return m


def _controlled_batch(sub: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.tile(np.eye(4, dtype=complex), (sub.shape[0], 1, 1))
    m[:, 2:, 2:] = sub
    return m


#: Closed-form batched builders for the rotations that dominate every ansatz. Each is
#: asserted equal to the scalar ``gate_matrix`` for that gate in ``tests/test_batch.py``
#: — a vectorised matrix that disagrees with the reference by a sign is exactly the
#: plausible-wrong-number bug this library exists to catch.
_VECTORISED: dict[str, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]] = {
    "rx": _rx_batch,
    "ry": _ry_batch,
    "rz": _rz_batch,
    "phase": _phase_batch,
    "crx": lambda a: _controlled_batch(_rx_batch(a)),
    "cry": lambda a: _controlled_batch(_ry_batch(a)),
    "crz": lambda a: _controlled_batch(_rz_batch(a)),
}


class NumpyBackend(Backend):
    """Exact statevector simulation in NumPy."""

    name = "numpy"
    supports_statevector = True
    supports_exact = True

    #: Above this width, :meth:`statevector_batch` falls back to simulating one
    #: sample at a time. Batching trades per-sample Python overhead for worse memory
    #: locality, so it wins while the overhead dominates and loses once ``2**n`` does.
    #: Measured on a hardware-efficient ansatz against the one-at-a-time loop:
    #: 30x at 4 qubits, 11x at 6, 3.9x at 8, 1.3x at 10, and 0.7x at 11 — the
    #: crossover sits between 10 and 11 and does not move with batch size, which is
    #: what you would expect if it is set by ``2**n`` alone. Raise it if you measure
    #: otherwise on your own hardware.
    batch_max_qubits = 10

    def __init__(self, seed: int | None = None, max_qubits: int = 24) -> None:
        super().__init__(seed)
        self.max_qubits = max_qubits

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        self._check_bound(spec)
        if spec.n_qubits > self.max_qubits:
            raise ValueError(
                f"{spec.n_qubits} qubits exceeds max_qubits={self.max_qubits}; "
                "raise it explicitly if you really mean to allocate that much memory"
            )
        state = np.zeros((2,) * spec.n_qubits, dtype=complex)
        state[(0,) * spec.n_qubits] = 1.0
        for op in spec.ops:
            state = _apply(state, self._matrix(op), op.qubits)
        return state.reshape(-1)


    def statevector_batch(
        self, spec: CircuitSpec, thetas: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Every sample's state in one pass, batch carried as a leading axis.

        The loop is over *gates*, not over samples: each gate is applied to the whole
        stack at once. A gate whose angle is the same for every sample contributes one
        matrix; one whose angle varies — every encoding gate — contributes a stack of
        them, built in closed form for the common rotations and by falling back to
        :func:`~qmlkit.core.gates.gate_matrix` for anything registered later.
        """
        values = np.atleast_2d(np.asarray(thetas, dtype=float))
        if values.shape[1] != spec.n_params:
            raise ValueError(
                f"circuit has {spec.n_params} parameters; got vectors of length "
                f"{values.shape[1]}"
            )
        if spec.n_qubits > self.max_qubits:
            raise ValueError(
                f"{spec.n_qubits} qubits exceeds max_qubits={self.max_qubits}; "
                "raise it explicitly if you really mean to allocate that much memory"
            )
        if spec.n_qubits > self.batch_max_qubits:
            # wider than the crossover: the one-at-a-time loop is genuinely faster
            return super().statevector_batch(spec, values)
        batch = values.shape[0]
        state = np.zeros((batch,) + (2,) * spec.n_qubits, dtype=complex)
        state[(slice(None),) + (0,) * spec.n_qubits] = 1.0

        for op in spec.ops:
            state = _apply_batch(state, self._matrices(op, values), op.qubits)
        return state.reshape(batch, -1)

    @staticmethod
    def _matrices(op: Op, values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """One matrix for the batch, or one per sample when an angle varies."""
        literals = [p for p in op.params if not isinstance(p, ParamRef)]
        if len(literals) == len(op.params):
            return gate_matrix(op.gate, tuple(float(p) for p in literals))

        angles = np.stack(
            [
                p.scale * values[:, p.index] + p.offset
                if isinstance(p, ParamRef)
                else np.full(values.shape[0], float(p))
                for p in op.params
            ],
            axis=1,
        )
        builder = _VECTORISED.get(op.gate)
        if builder is not None and angles.shape[1] == 1:
            return builder(angles[:, 0])
        # a registered custom gate: correctness first, one build per sample
        return np.stack([gate_matrix(op.gate, tuple(row)) for row in angles])

    @staticmethod
    def _matrix(op: Op) -> npt.NDArray[Any]:
        angles: list[float] = []
        for p in op.params:
            if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
                raise ValueError(f"unbound parameter reached the backend in {op.gate!r}")
            angles.append(float(p))
        return gate_matrix(op.gate, tuple(angles))
