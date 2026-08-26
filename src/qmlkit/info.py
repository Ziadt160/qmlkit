"""Quantum information quantities — the ``qml.qinfo`` equivalent.

Reduced density matrices, purity, entropies and state fidelity. These are the
building blocks the trainability metrics and the projected kernel are made of, and
they are useful on their own for looking at what a circuit is actually doing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qmlkit.core.execute import BackendLike, statevector
from qmlkit.core.ir import CircuitSpec

__all__ = [
    "density_matrix",
    "reduced_dm",
    "purity",
    "vn_entropy",
    "mutual_info",
    "state_fidelity",
    "concurrence",
    "bloch_vector",
]


def _as_state(state: CircuitSpec | np.ndarray, backend: BackendLike = None) -> np.ndarray:
    if isinstance(state, CircuitSpec):
        return statevector(state, backend=backend)
    return np.asarray(state, dtype=complex).ravel()


def density_matrix(state: CircuitSpec | np.ndarray, backend: BackendLike = None) -> np.ndarray:
    """``|psi><psi|`` for a pure state."""
    psi = _as_state(state, backend)
    return np.outer(psi, psi.conj())


def reduced_dm(
    state: CircuitSpec | np.ndarray,
    wires: Sequence[int],
    n_qubits: int | None = None,
    backend: BackendLike = None,
) -> np.ndarray:
    """Trace out everything except ``wires``.

    Qubit 0 is the most significant bit, matching the rest of the library.
    """
    psi = _as_state(state, backend)
    n = n_qubits if n_qubits is not None else int(np.log2(psi.size))
    if 2**n != psi.size:
        raise ValueError(f"state of size {psi.size} is not {n} qubits")
    keep = sorted(set(wires))
    if any(not 0 <= w < n for w in keep):
        raise ValueError(f"wires {list(wires)} out of range for {n} qubits")

    tensor = psi.reshape((2,) * n)
    traced = [q for q in range(n) if q not in keep]
    # move kept axes to the front, then contract the rest against their conjugate
    perm = keep + traced
    tensor = np.transpose(tensor, perm)
    k = len(keep)
    tensor = tensor.reshape(2**k, -1)
    return tensor @ tensor.conj().T


def purity(
    state: CircuitSpec | np.ndarray,
    wires: Sequence[int] | None = None,
    n_qubits: int | None = None,
    backend: BackendLike = None,
) -> float:
    """``Tr(rho^2)`` — 1 for a pure state, ``1/d`` for the maximally mixed one."""
    if wires is None:
        return 1.0  # a statevector is pure by construction
    rho = reduced_dm(state, wires, n_qubits, backend)
    return float(np.real(np.trace(rho @ rho)))


def vn_entropy(
    state: CircuitSpec | np.ndarray,
    wires: Sequence[int],
    n_qubits: int | None = None,
    base: float | None = None,
    backend: BackendLike = None,
) -> float:
    """Von Neumann entropy of a subsystem — how entangled it is with the rest."""
    rho = reduced_dm(state, wires, n_qubits, backend)
    eig = np.linalg.eigvalsh(rho)
    eig = eig[eig > 1e-12]
    entropy = float(-np.sum(eig * np.log(eig)))
    return entropy / np.log(base) if base is not None else entropy


def mutual_info(
    state: CircuitSpec | np.ndarray,
    wires_a: Sequence[int],
    wires_b: Sequence[int],
    n_qubits: int | None = None,
    backend: BackendLike = None,
) -> float:
    """``S(A) + S(B) - S(AB)`` — total correlation between two subsystems."""
    a, b = sorted(set(wires_a)), sorted(set(wires_b))
    if set(a) & set(b):
        raise ValueError(f"subsystems overlap on {sorted(set(a) & set(b))}")
    sa = vn_entropy(state, a, n_qubits, backend=backend)
    sb = vn_entropy(state, b, n_qubits, backend=backend)
    sab = vn_entropy(state, a + b, n_qubits, backend=backend)
    return sa + sb - sab


def state_fidelity(
    state_a: CircuitSpec | np.ndarray,
    state_b: CircuitSpec | np.ndarray,
    backend: BackendLike = None,
) -> float:
    """``|<a|b>|^2`` — the quantity a fidelity kernel estimates."""
    a = _as_state(state_a, backend)
    b = _as_state(state_b, backend)
    if a.size != b.size:
        raise ValueError(f"states have different widths: {a.size} vs {b.size}")
    return float(abs(np.vdot(a, b)) ** 2)


def concurrence(state: CircuitSpec | np.ndarray, backend: BackendLike = None) -> float:
    """Two-qubit concurrence — 0 for a product state, 1 for a Bell state."""
    psi = _as_state(state, backend)
    if psi.size != 4:
        raise ValueError("concurrence is defined here for two qubits only")
    yy = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
    return float(abs(psi @ yy @ psi))


def bloch_vector(
    state: CircuitSpec | np.ndarray,
    wire: int = 0,
    n_qubits: int | None = None,
    backend: BackendLike = None,
) -> np.ndarray:
    """``(<X>, <Y>, <Z>)`` for one qubit — its point on (or in) the Bloch sphere."""
    rho = reduced_dm(state, [wire], n_qubits, backend)
    x = 2 * np.real(rho[0, 1])
    y = 2 * np.imag(rho[1, 0])
    z = np.real(rho[0, 0] - rho[1, 1])
    return np.array([x, y, z], dtype=float)
