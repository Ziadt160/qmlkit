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

import numpy as np

from qmlkit.core.backends.base import Backend
from qmlkit.core.gates import gate_matrix
from qmlkit.core.ir import CircuitSpec, Op, ParamRef


def _apply(state: np.ndarray, matrix: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    """Apply a k-qubit gate to the tensor-shaped state."""
    k = len(qubits)
    op = matrix.reshape((2,) * (2 * k))
    # contract the gate's input legs with the state's qubit axes
    state = np.tensordot(op, state, axes=(list(range(k, 2 * k)), list(qubits)))
    # tensordot puts the gate's output legs first; move them back into place
    return np.moveaxis(state, list(range(k)), list(qubits))


class NumpyBackend(Backend):
    """Exact statevector simulation in NumPy."""

    name = "numpy"
    supports_statevector = True
    supports_exact = True

    def __init__(self, seed: int | None = None, max_qubits: int = 24) -> None:
        super().__init__(seed)
        self.max_qubits = max_qubits

    def statevector(self, spec: CircuitSpec) -> np.ndarray:
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

    @staticmethod
    def _matrix(op: Op) -> np.ndarray:
        angles: list[float] = []
        for p in op.params:
            if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
                raise ValueError(f"unbound parameter reached the backend in {op.gate!r}")
            angles.append(float(p))
        return gate_matrix(op.gate, tuple(angles))
