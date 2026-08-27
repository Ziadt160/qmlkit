r"""Hamiltonians to hand to VQE, and an exact answer to check it against.

A Hamiltonian here is just a :class:`~qmlkit.core.observables.PauliSum` — the same
type an expectation value takes — so nothing new has to learn about it. These are
constructors, not a new class hierarchy.

:func:`exact_ground_energy` diagonalises the dense matrix. That is exponential and
useless past ~14 qubits, which is exactly the point: it is the oracle a variational
result gets *checked* against on small systems, not a method to compete with.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.builder import entangler_pairs
from qmlkit.core.observables import Observable, PauliString, PauliSum, as_sum

__all__ = [
    "pauli_hamiltonian",
    "ising_hamiltonian",
    "heisenberg_hamiltonian",
    "max_cut_hamiltonian",
    "hamiltonian_matrix",
    "exact_ground_energy",
    "exact_ground_state",
]

_PAULI: dict[str, npt.NDArray[Any]] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


def pauli_hamiltonian(terms: Iterable[tuple[str, Sequence[int], float]]) -> PauliSum:
    """Build from ``(paulis, qubits, coefficient)`` triples.

    >>> pauli_hamiltonian([("ZZ", (0, 1), 1.0), ("X", (0,), -0.5)])
    Z0 Z1 + -0.5*X0
    """
    out: list[PauliString] = []
    for letters, qubits, coeff in terms:
        # A constant term is the identity: no letters and no qubits. Allowed, because
        # dropping it would silently shift every energy the Hamiltonian reports.
        if not letters and not tuple(qubits):
            out.append(PauliString((), float(coeff)))
            continue
        if len(letters) != len(qubits):
            raise ValueError(f"{letters!r} needs {len(letters)} qubits, got {tuple(qubits)}")
        paulis = tuple(
            (int(q), p.upper()) for q, p in zip(qubits, letters, strict=True) if p.upper() != "I"
        )
        out.append(PauliString(tuple(sorted(paulis)), float(coeff)))
    return PauliSum(tuple(out))


def ising_hamiltonian(
    n_qubits: int,
    j: float = 1.0,
    h: float = 1.0,
    edges: Sequence[tuple[int, int]] | None = None,
    pattern: str = "chain",
) -> PauliSum:
    r"""Transverse-field Ising model, :math:`H = J\sum Z_iZ_j + h\sum X_i`.

    The standard first test for any variational eigensolver: it is exactly solvable,
    frustration-free at ``h=0``, and its ground state becomes genuinely entangled as
    ``h`` grows, so a working VQE has to do real work.
    """
    graph = list(edges) if edges is not None else list(entangler_pairs(n_qubits, pattern))
    terms = [("ZZ", (a, b), j) for a, b in graph]
    terms += [("X", (q,), h) for q in range(n_qubits)]
    return pauli_hamiltonian(terms)


def heisenberg_hamiltonian(
    n_qubits: int,
    jx: float = 1.0,
    jy: float = 1.0,
    jz: float = 1.0,
    h: float = 0.0,
    edges: Sequence[tuple[int, int]] | None = None,
    pattern: str = "chain",
) -> PauliSum:
    r"""Heisenberg model, :math:`\sum J_\alpha \sigma^\alpha_i\sigma^\alpha_j + h\sum Z_i`."""
    graph = list(edges) if edges is not None else list(entangler_pairs(n_qubits, pattern))
    terms: list[tuple[str, Sequence[int], float]] = []
    for a, b in graph:
        for letter, coupling in (("XX", jx), ("YY", jy), ("ZZ", jz)):
            if coupling:
                terms.append((letter, (a, b), coupling))
    terms += [("Z", (q,), h) for q in range(n_qubits) if h]
    return pauli_hamiltonian(terms)


def max_cut_hamiltonian(edges: Sequence[tuple[int, int]], n_qubits: int | None = None) -> PauliSum:
    r"""MaxCut cost, :math:`\tfrac12\sum_{(i,j)\in E}(Z_iZ_j - 1)`.

    Minimising this maximises the cut, and its ground-state energy is
    ``-(number of edges cut)``. The constant is kept rather than dropped so the
    energy VQE or QAOA reports *is* the negated cut size, with nothing to add back.
    """
    if not edges:
        raise ValueError("MaxCut needs at least one edge")
    _ = n_qubits  # width comes from the edges themselves; kept for a symmetric API
    terms: list[tuple[str, Sequence[int], float]] = [("ZZ", (a, b), 0.5) for a, b in edges]
    terms.append(("", (), -0.5 * len(edges)))
    return pauli_hamiltonian(terms)


def hamiltonian_matrix(obs: Observable, n_qubits: int) -> npt.NDArray[Any]:
    """Dense ``2**n x 2**n`` matrix. Exponential — for checking, not for running."""
    if n_qubits > 14:
        raise ValueError(
            f"a dense matrix for {n_qubits} qubits needs {4**n_qubits * 16 / 1e9:.0f} GB; "
            "this function exists to verify small cases, not to solve large ones"
        )
    dim = 2**n_qubits
    total = np.zeros((dim, dim), dtype=complex)
    for term in as_sum(obs).terms:
        letters = dict(term.paulis)
        matrix = np.eye(1, dtype=complex)
        for qubit in range(n_qubits):
            matrix = np.kron(matrix, _PAULI[letters.get(qubit, "I")])
        total += complex(term.coeff) * matrix
    return total


def exact_ground_energy(obs: Observable, n_qubits: int) -> float:
    """Lowest eigenvalue, by dense diagonalisation. The oracle, not the method."""
    return float(np.linalg.eigvalsh(hamiltonian_matrix(obs, n_qubits))[0])


def exact_ground_state(obs: Observable, n_qubits: int) -> tuple[float, npt.NDArray[Any]]:
    """Lowest eigenvalue and its eigenvector."""
    values, vectors = np.linalg.eigh(hamiltonian_matrix(obs, n_qubits))
    return float(values[0]), np.asarray(vectors[:, 0])
