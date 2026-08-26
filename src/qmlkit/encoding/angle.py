"""Angle and basis encoding — getting classical numbers into a circuit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qmlkit.core.builder import QCircuit
from qmlkit.core.ir import CircuitSpec, ParamRef

__all__ = ["angle_encode", "basis_encode", "basis_index", "n_qubits_for"]


def angle_encode(
    x: Sequence[float],
    rotation: str = "ry",
    trainable: bool = False,
) -> CircuitSpec:
    """One feature per qubit, written into a rotation angle.

    ``trainable=False`` bakes the values in as literals. ``trainable=True`` makes
    them circuit *parameters* instead — which is what lets the same shift rule
    deliver ``df/dx``, the gradient a classical pre-net needs in a hybrid stack.
    """
    values = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    if values.size == 0:
        raise ValueError("angle_encode needs at least one feature")
    qc = QCircuit(values.size)
    for i, xi in enumerate(values):
        qc.apply(rotation, i, ParamRef(i) if trainable else float(xi))
    return qc.to_spec()


def basis_encode(bits: Sequence[int]) -> CircuitSpec:
    """Computational-basis encoding: flip a qubit wherever the bit is 1."""
    values = [int(b) for b in bits]
    if not values:
        raise ValueError("basis_encode needs at least one bit")
    if any(b not in (0, 1) for b in values):
        raise ValueError(f"basis_encode takes 0/1 values, got {values}")
    qc = QCircuit(len(values))
    for i, b in enumerate(values):
        if b == 1:
            qc.x(i)
    return qc.to_spec()


def basis_index(bits: Sequence[int]) -> int:
    """``[1, 0, 1] -> 5``. Qubit 0 is the most significant bit."""
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def n_qubits_for(n_values: int) -> int:
    """Qubits needed to hold ``n_values`` amplitudes: ``ceil(log2 N)``."""
    if n_values <= 0:
        raise ValueError("n_values must be positive")
    return max(1, int(np.ceil(np.log2(n_values))))
