r"""Molecular Hamiltonians, computed here rather than quoted.

VQE's canonical demonstration is the ground-state energy of H\ :sub:`2`, and most
tutorials get the Hamiltonian by importing coefficients from a chemistry package or
copying a table out of a paper. This module computes it: STO-3G integrals over
Gaussian primitives, symmetry-adapted molecular orbitals, second quantisation, and a
Jordan–Wigner map to four qubits.

That matters for a library whose whole argument is that you should be able to see the
cost of what you run. It is also checkable — the curve below reproduces the published
FCI/STO-3G result to five decimals:

    >>> from qmlkit.algorithms.chemistry import h2_hamiltonian
    >>> from qmlkit.algorithms import exact_ground_energy
    >>> h, info = h2_hamiltonian(0.735)
    >>> round(exact_ground_energy(h, 4), 5)
    -1.13731

Minimal basis only, and two centres only. Anything larger wants PySCF or
OpenFermion, and the point here is transparency rather than coverage.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.observables import PauliString, PauliSum

__all__ = [
    "h2_hamiltonian",
    "h2_curve",
    "BOHR_PER_ANGSTROM",
    "HARTREE_TO_KCAL",
    "CHEMICAL_ACCURACY",
]

#: STO-3G contraction for the hydrogen 1s orbital
_ALPHA = np.array([3.42525091, 0.62391373, 0.16885540])
_COEFF = np.array([0.15432897, 0.53532814, 0.44463454])
_D = _COEFF * (2 * _ALPHA / np.pi) ** 0.75  # fold in the primitive normalisation

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903
HARTREE_TO_KCAL = 627.509474
#: 1 kcal/mol in hartree — the accuracy chemistry actually cares about
CHEMICAL_ACCURACY = 1.0 / HARTREE_TO_KCAL

_I2 = np.eye(2)
_Z = np.diag([1.0, -1.0])
_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]])
_PAULI: dict[str, npt.NDArray[Any]] = {
    "I": _I2,
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1j], [1j, 0.0]]),
    "Z": _Z,
}


def _boys(t: float) -> float:
    """Boys function :math:`F_0`, which is all s-type integrals need.

    `math.erf` rather than SciPy's: qmlkit depends on NumPy alone, and reaching for
    SciPy here would have added a runtime dependency for one scalar special function.
    """
    value = float(t)
    if value < 1e-12:
        return 1.0
    return float(np.sqrt(np.pi / (4 * value)) * math.erf(np.sqrt(value)))


def _kron(*matrices: npt.NDArray[Any]) -> npt.NDArray[Any]:
    out = np.eye(1, dtype=complex)
    for m in matrices:
        out = np.kron(out, m)
    return out


def _ao_integrals(r_bohr: float) -> tuple[npt.NDArray[Any], ...]:
    """Overlap, kinetic, nuclear attraction and two-electron integrals."""
    centres = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, r_bohr])]
    n = 2
    overlap = np.zeros((n, n))
    kinetic = np.zeros((n, n))
    nuclear = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        diff = centres[i] - centres[j]
        d2 = float(diff @ diff)
        for a, da in zip(_ALPHA, _D, strict=True):
            for b, db in zip(_ALPHA, _D, strict=True):
                p = a + b
                mu = a * b / p
                gaussian = np.exp(-mu * d2)
                s = (np.pi / p) ** 1.5 * gaussian
                overlap[i, j] += da * db * s
                kinetic[i, j] += da * db * mu * (3 - 2 * mu * d2) * s
                centre = (a * centres[i] + b * centres[j]) / p
                for nucleus in centres:  # both hydrogens carry Z = 1
                    pc = centre - nucleus
                    nuclear[i, j] -= (
                        da * db * 2 * np.pi / p * gaussian * float(_boys(p * float(pc @ pc)))
                    )

    eri = np.zeros((n, n, n, n))
    for i, j, k, m in itertools.product(range(n), repeat=4):
        total = 0.0
        dij = centres[i] - centres[j]
        dkm = centres[k] - centres[m]
        for a, da in zip(_ALPHA, _D, strict=True):
            for b, db in zip(_ALPHA, _D, strict=True):
                p = a + b
                centre_p = (a * centres[i] + b * centres[j]) / p
                k_ab = np.exp(-a * b / p * float(dij @ dij))
                for c, dc in zip(_ALPHA, _D, strict=True):
                    for d, dd in zip(_ALPHA, _D, strict=True):
                        q = c + d
                        centre_q = (c * centres[k] + d * centres[m]) / q
                        k_cd = np.exp(-c * d / q * float(dkm @ dkm))
                        pq = centre_p - centre_q
                        total += (
                            da
                            * db
                            * dc
                            * dd
                            * 2
                            * np.pi**2.5
                            / (p * q * np.sqrt(p + q))
                            * k_ab
                            * k_cd
                            * float(_boys(p * q / (p + q) * float(pq @ pq)))
                        )
        eri[i, j, k, m] = total
    return overlap, kinetic, nuclear, eri


def _annihilator(orbital: int, n_spin_orbitals: int) -> npt.NDArray[Any]:
    """Jordan-Wigner: a Z string for the fermionic sign, then a lowering operator."""
    factors = [_Z] * orbital + [_SIGMA_MINUS] + [_I2] * (n_spin_orbitals - orbital - 1)
    return _kron(*factors)


def _matrix(bond_length_angstrom: float) -> tuple[npt.NDArray[Any], float]:
    """The 16x16 Hamiltonian in the occupation-number basis, plus nuclear repulsion."""
    r = bond_length_angstrom * BOHR_PER_ANGSTROM
    overlap, kinetic, nuclear, eri = _ao_integrals(r)
    core = kinetic + nuclear

    # For a symmetric two-centre minimal basis the molecular orbitals are fixed by
    # symmetry, so no SCF iteration is needed: sigma_g and sigma_u, normalised.
    s = overlap[0, 1]
    coefficients = np.array(
        [
            [1 / np.sqrt(2 * (1 + s)), 1 / np.sqrt(2 * (1 - s))],
            [1 / np.sqrt(2 * (1 + s)), -1 / np.sqrt(2 * (1 - s))],
        ]
    )
    h = coefficients.T @ core @ coefficients
    g = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        eri,
        optimize=True,
    )

    n = 4  # two spatial orbitals, two spins
    a = [_annihilator(p, n) for p in range(n)]
    adag = [x.conj().T for x in a]
    spin = (0, 1, 0, 1)
    spatial = (0, 0, 1, 1)

    matrix = np.zeros((2**n, 2**n), dtype=complex)
    for p, q in itertools.product(range(n), repeat=2):
        if spin[p] == spin[q]:
            matrix += h[spatial[p], spatial[q]] * (adag[p] @ a[q])
    for p, q, r_, s_ in itertools.product(range(n), repeat=4):
        if spin[p] == spin[q] and spin[r_] == spin[s_]:
            matrix += (
                0.5
                * g[spatial[p], spatial[q], spatial[r_], spatial[s_]]
                * (adag[p] @ adag[r_] @ a[s_] @ a[q])
            )
    repulsion = 1.0 / r
    return matrix + repulsion * np.eye(2**n), repulsion


def h2_hamiltonian(
    bond_length: float = 0.735, tol: float = 1e-10
) -> tuple[PauliSum, dict[str, Any]]:
    """The H2 qubit Hamiltonian at a given bond length in angstrom.

    Returns the observable and a dictionary of what went into it. The Pauli
    coefficients come from projecting the dense matrix, ``c_P = Tr(P H) / 2^n``,
    which needs no symbolic algebra and is trivially checkable in the other
    direction with :func:`~qmlkit.algorithms.hamiltonian_matrix`.
    """
    matrix, repulsion = _matrix(bond_length)
    terms: list[PauliString] = []
    for letters in itertools.product("IXYZ", repeat=4):
        operator = _kron(*[_PAULI[c] for c in letters])
        coefficient = float(np.real(np.trace(operator @ matrix)) / 16)
        if abs(coefficient) > tol:
            paulis = tuple((q, c) for q, c in enumerate(letters) if c != "I")
            terms.append(PauliString(paulis, coefficient))
    info = {
        "bond_length": bond_length,
        "n_qubits": 4,
        "n_terms": len(terms),
        "nuclear_repulsion": repulsion,
        "hartree_fock_occupation": [1, 1, 0, 0],  # both electrons in sigma_g
    }
    return PauliSum(tuple(terms)), info


def h2_curve(bond_lengths: npt.NDArray[Any] | list[float]) -> list[tuple[float, PauliSum]]:
    """``(bond_length, hamiltonian)`` pairs — the dissociation curve as input data."""
    return [(float(r), h2_hamiltonian(float(r))[0]) for r in bond_lengths]
