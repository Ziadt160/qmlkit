r"""Molecular Hamiltonians for any molecule, by two routes.

**Route one — bring your own integrals.** This is the general one, and the one to
reach for past a couple of light atoms. Anything that can produce one- and
two-electron integrals in a molecular-orbital basis — PySCF, OpenFermion, Psi4 — hands
them to :func:`from_integrals` and gets a qubit Hamiltonian back::

    from pyscf import gto, scf, ao2mo
    import numpy as np

    mol = gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g")
    mf = scf.RHF(mol).run()
    c = mf.mo_coeff
    h1 = c.T @ mf.get_hcore() @ c
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, c), c.shape[1])

    hamiltonian, info = from_integrals(h1, h2, n_electrons=mol.nelectron,
                                       nuclear_repulsion=mol.energy_nuc())

That decoupling is deliberate. A quantum ML library should not also be a quantum
chemistry package, and pretending otherwise would mean shipping a worse version of
software that already exists.

**Route two — the built-in SCF.** For molecules built only from *s*-orbital atoms
(hydrogen and helium in STO-3G) the integrals are computed here, with a real
restricted Hartree–Fock loop rather than a symmetry shortcut. That covers the systems
VQE is usually benchmarked on — H\ :sub:`2`, H\ :sub:`3`\ :sup:`+`, H\ :sub:`4` chains
and rings, HeH\ :sup:`+` — at arbitrary geometry::

    from qmlkit.algorithms import Molecule, molecular_hamiltonian

    h4 = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.9)),
                   ("H", (0, 0, 1.8)), ("H", (0, 0, 2.7))])
    hamiltonian, info = molecular_hamiltonian(h4)

Anything with *p* orbitals needs route one. That boundary is stated rather than
papered over.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.observables import PauliString, PauliSum

__all__ = [
    "Molecule",
    "MolecularInfo",
    "molecular_hamiltonian",
    "from_integrals",
    "hydrogen_chain",
    "hydrogen_ring",
    "SUPPORTED_ELEMENTS",
    "BOHR_PER_ANGSTROM",
]

BOHR_PER_ANGSTROM = 1.0 / 0.529177210903

#: STO-3G 1s contractions. Only elements whose occupied shells are pure *s*, because
#: p-orbital integrals need a different (and much longer) set of recursions.
SUPPORTED_ELEMENTS: dict[str, dict[str, Any]] = {
    "H": {"z": 1, "electrons": 1, "alpha": (3.42525091, 0.62391373, 0.16885540)},
    "He": {"z": 2, "electrons": 2, "alpha": (6.36242139, 1.15892300, 0.31364979)},
}
_CONTRACTION = np.array([0.15432897, 0.53532814, 0.44463454])

_I2 = np.eye(2)
_Zmat = np.diag([1.0, -1.0])
_SIGMA_MINUS = np.array([[0.0, 1.0], [0.0, 0.0]])
_PAULI: dict[str, npt.NDArray[Any]] = {
    "I": _I2,
    "X": np.array([[0.0, 1.0], [1.0, 0.0]]),
    "Y": np.array([[0.0, -1j], [1j, 0.0]]),
    "Z": _Zmat,
}


@dataclass
class Molecule:
    """Atoms and where they are. Positions in angstrom."""

    atoms: list[tuple[str, tuple[float, float, float]]]
    charge: int = 0

    def __post_init__(self) -> None:
        unknown = sorted({s for s, _ in self.atoms} - set(SUPPORTED_ELEMENTS))
        if unknown:
            raise ValueError(
                f"the built-in SCF handles only s-orbital elements "
                f"{sorted(SUPPORTED_ELEMENTS)}, not {unknown}. Compute the integrals "
                "with PySCF or OpenFermion and pass them to from_integrals() instead."
            )

    @property
    def n_electrons(self) -> int:
        total = sum(SUPPORTED_ELEMENTS[s]["electrons"] for s, _ in self.atoms)
        return int(total - self.charge)

    @property
    def n_orbitals(self) -> int:
        """One 1s function per atom, in this basis."""
        return len(self.atoms)

    def __repr__(self) -> str:
        formula = "".join(s for s, _ in self.atoms)
        return f"Molecule({formula}, charge={self.charge}, {self.n_electrons} electrons)"


@dataclass
class MolecularInfo:
    """Everything that went into the Hamiltonian, so the result can be audited."""

    n_qubits: int
    n_electrons: int
    n_orbitals: int
    n_terms: int
    nuclear_repulsion: float
    hartree_fock_energy: float | None = None
    hartree_fock_occupation: list[int] = field(default_factory=list)
    active_space: tuple[int, ...] | None = None

    def __repr__(self) -> str:
        return (
            f"MolecularInfo(qubits={self.n_qubits}, electrons={self.n_electrons}, "
            f"orbitals={self.n_orbitals}, terms={self.n_terms})"
        )


def hydrogen_chain(n: int, spacing: float = 0.74) -> Molecule:
    """``n`` hydrogens in a line — the standard scaling benchmark for VQE."""
    return Molecule([("H", (0.0, 0.0, i * spacing)) for i in range(n)])


def hydrogen_ring(n: int, radius: float = 1.0) -> Molecule:
    """``n`` hydrogens on a circle; frustrated, and harder than the chain."""
    return Molecule(
        [
            ("H", (radius * np.cos(2 * np.pi * i / n), radius * np.sin(2 * np.pi * i / n), 0.0))
            for i in range(n)
        ]
    )


# --------------------------------------------------------------------------- #
# integrals over contracted s-type Gaussians
# --------------------------------------------------------------------------- #
def _boys(t: float) -> float:
    """Boys function F0, from the standard library rather than SciPy.

    `math.erf` has been there since Python 3.2, and using it keeps qmlkit's only
    runtime dependency NumPy -- which is a promise the README makes and which CI
    checks by installing nothing else.
    """
    if t < 1e-12:
        return 1.0
    return float(np.sqrt(np.pi / (4 * t)) * math.erf(np.sqrt(t)))


def _basis(molecule: Molecule) -> list[tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]]:
    """``(centre, exponents, coefficients)`` per basis function."""
    out = []
    for symbol, position in molecule.atoms:
        alpha = np.array(SUPPORTED_ELEMENTS[symbol]["alpha"])
        coeff = _CONTRACTION * (2 * alpha / np.pi) ** 0.75
        out.append((np.array(position) * BOHR_PER_ANGSTROM, alpha, coeff))
    return out


def _ao_integrals(molecule: Molecule) -> tuple[npt.NDArray[Any], ...]:
    basis = _basis(molecule)
    n = len(basis)
    charges = [
        (SUPPORTED_ELEMENTS[s]["z"], np.array(p) * BOHR_PER_ANGSTROM) for s, p in molecule.atoms
    ]

    overlap = np.zeros((n, n))
    core = np.zeros((n, n))
    for i, j in itertools.product(range(n), repeat=2):
        ci, ai, di = basis[i]
        cj, aj, dj = basis[j]
        d2 = float((ci - cj) @ (ci - cj))
        for a, da in zip(ai, di, strict=True):
            for b, db in zip(aj, dj, strict=True):
                p = a + b
                mu = a * b / p
                gauss = np.exp(-mu * d2)
                s = (np.pi / p) ** 1.5 * gauss
                overlap[i, j] += da * db * s
                core[i, j] += da * db * mu * (3 - 2 * mu * d2) * s  # kinetic
                centre = (a * ci + b * cj) / p
                for z, nucleus in charges:
                    pc = centre - nucleus
                    core[i, j] -= da * db * 2 * np.pi / p * z * gauss * _boys(p * float(pc @ pc))

    eri = np.zeros((n, n, n, n))
    for i, j, k, m in itertools.product(range(n), repeat=4):
        ci, ai, di = basis[i]
        cj, aj, dj = basis[j]
        ck, ak, dk = basis[k]
        cm, am, dm = basis[m]
        dij = float((ci - cj) @ (ci - cj))
        dkm = float((ck - cm) @ (ck - cm))
        total = 0.0
        for a, da in zip(ai, di, strict=True):
            for b, db in zip(aj, dj, strict=True):
                p = a + b
                cp = (a * ci + b * cj) / p
                kab = np.exp(-a * b / p * dij)
                for c, dc in zip(ak, dk, strict=True):
                    for d, dd in zip(am, dm, strict=True):
                        q = c + d
                        cq = (c * ck + d * cm) / q
                        kcd = np.exp(-c * d / q * dkm)
                        pq = cp - cq
                        total += (
                            da
                            * db
                            * dc
                            * dd
                            * 2
                            * np.pi**2.5
                            / (p * q * np.sqrt(p + q))
                            * kab
                            * kcd
                            * _boys(p * q / (p + q) * float(pq @ pq))
                        )
        eri[i, j, k, m] = total

    repulsion = 0.0
    for (z1, r1), (z2, r2) in itertools.combinations(charges, 2):
        repulsion += z1 * z2 / float(np.linalg.norm(r1 - r2))
    return overlap, core, eri, np.array(repulsion)


def _rhf(
    overlap: npt.NDArray[Any],
    core: npt.NDArray[Any],
    eri: npt.NDArray[Any],
    n_electrons: int,
    max_iterations: int = 100,
    tol: float = 1e-10,
) -> tuple[npt.NDArray[Any], float]:
    """Restricted Hartree-Fock. Returns MO coefficients and the electronic energy.

    Symmetric orthogonalisation, then the usual diagonalise-build-repeat loop. No
    symmetry shortcut, so this works for any geometry rather than only symmetric ones.
    """
    if n_electrons % 2:
        raise ValueError(f"restricted Hartree-Fock needs an even electron count, got {n_electrons}")
    n_occupied = n_electrons // 2
    values, vectors = np.linalg.eigh(overlap)
    x = vectors @ np.diag(values**-0.5) @ vectors.T  # S^{-1/2}

    density = np.zeros_like(core)
    energy = 0.0
    for _ in range(max_iterations):
        coulomb = np.einsum("ls,mnls->mn", density, eri, optimize=True)
        exchange = np.einsum("ls,mlns->mn", density, eri, optimize=True)
        fock = core + coulomb - 0.5 * exchange
        _, c_prime = np.linalg.eigh(x.T @ fock @ x)
        coefficients = x @ c_prime
        occupied = coefficients[:, :n_occupied]
        new_density = 2 * occupied @ occupied.T
        new_energy = 0.5 * float(np.sum(new_density * (core + fock)))
        if abs(new_energy - energy) < tol and np.abs(new_density - density).max() < tol:
            density, energy = new_density, new_energy
            break
        density, energy = new_density, new_energy
    return coefficients, energy


# --------------------------------------------------------------------------- #
# second quantisation and the Jordan-Wigner map
# --------------------------------------------------------------------------- #
def _kron(*matrices: npt.NDArray[Any]) -> npt.NDArray[Any]:
    out = np.eye(1, dtype=complex)
    for m in matrices:
        out = np.kron(out, m)
    return out


def _annihilator(orbital: int, n_spin_orbitals: int) -> npt.NDArray[Any]:
    return _kron(*([_Zmat] * orbital + [_SIGMA_MINUS] + [_I2] * (n_spin_orbitals - orbital - 1)))


#: (m00, m01, m10, m11) -> (c_I, c_X, c_Y, c_Z), i.e. Tr(P M)/2 for each Pauli
_TO_PAULI = 0.5 * np.array(
    [
        [1, 0, 0, 1],  # I
        [0, 1, 1, 0],  # X
        [0, 1j, -1j, 0],  # Y
        [1, 0, 0, -1],  # Z
    ],
    dtype=complex,
)


def _decompose(matrix: npt.NDArray[Any], n_qubits: int, tol: float) -> list[PauliString]:
    r"""``c_P = Tr(P H) / 2^n`` for every Pauli string at once.

    Done term by term this is ``Tr(P @ H)`` over ``4^n`` strings, each a
    ``2^n x 2^n`` matrix product — ``O(16^n)``, which is about a minute for a
    four-atom molecule and hopeless past that.

    The same numbers come out of ``n`` applications of one 4x4 change of basis: pair
    each row index with its column index, and rotate that pair from the
    matrix-element basis into the Pauli basis. That is ``O(n * 4^n)``, and it takes
    milliseconds where the direct form took minutes.
    """
    tensor = np.asarray(matrix, dtype=complex).reshape((2,) * (2 * n_qubits))
    # interleave row/column indices so each qubit's (r, c) pair sits on one axis
    order = [i for q in range(n_qubits) for i in (q, q + n_qubits)]
    tensor = tensor.transpose(order).reshape((4,) * n_qubits)
    for axis in range(n_qubits):
        tensor = np.moveaxis(np.tensordot(_TO_PAULI, tensor, axes=([1], [axis])), 0, axis)

    letters = "IXYZ"
    terms: list[PauliString] = []
    for index, value in enumerate(tensor.reshape(-1)):
        coefficient = float(np.real(value))
        if abs(coefficient) <= tol:
            continue
        digits = np.base_repr(index, base=4).rjust(n_qubits, "0")
        terms.append(
            PauliString(
                tuple((q, letters[int(d)]) for q, d in enumerate(digits) if d != "0"), coefficient
            )
        )
    return terms


def from_integrals(
    one_body: npt.NDArray[Any],
    two_body: npt.NDArray[Any],
    n_electrons: int,
    nuclear_repulsion: float = 0.0,
    active_space: tuple[int, ...] | None = None,
    tol: float = 1e-10,
) -> tuple[PauliSum, MolecularInfo]:
    r"""A qubit Hamiltonian from molecular-orbital integrals.

    Parameters
    ----------
    one_body
        ``h[p, q]``, the one-electron integrals in the MO basis.
    two_body
        ``g[p, q, r, s]`` in **chemist notation** ``(pq|rs)``, which is what PySCF's
        ``ao2mo.restore(1, ...)`` returns.
    active_space
        Spatial orbitals to keep. Everything else is dropped, which is the usual way
        to fit a molecule onto a machine you actually have — ``2 * len(active_space)``
        qubits instead of ``2 * n_orbitals``.

    Notes
    -----
    This is the general entry point: it never asks where the integrals came from.
    """
    h = np.asarray(one_body, dtype=float)
    g = np.asarray(two_body, dtype=float)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"one_body must be square, got shape {h.shape}")
    if g.shape != (h.shape[0],) * 4:
        raise ValueError(f"two_body must have shape {(h.shape[0],) * 4}, got {g.shape}")

    if active_space is not None:
        keep = list(active_space)
        h = h[np.ix_(keep, keep)]
        g = g[np.ix_(keep, keep, keep, keep)]

    n_spatial = h.shape[0]
    n_spin = 2 * n_spatial
    if n_spin > 12:
        raise ValueError(
            f"{n_spin} qubits means a {2**n_spin}-dimensional matrix. Use active_space "
            "to pick the orbitals that matter."
        )

    a = [_annihilator(p, n_spin) for p in range(n_spin)]
    adag = [x.conj().T for x in a]
    spin = [p % 2 for p in range(n_spin)]
    spatial = [p // 2 for p in range(n_spin)]

    matrix = np.zeros((2**n_spin, 2**n_spin), dtype=complex)
    for p, q in itertools.product(range(n_spin), repeat=2):
        if spin[p] == spin[q]:
            matrix += h[spatial[p], spatial[q]] * (adag[p] @ a[q])
    for p, q, r, s in itertools.product(range(n_spin), repeat=4):
        if spin[p] == spin[q] and spin[r] == spin[s]:
            matrix += (
                0.5
                * g[spatial[p], spatial[q], spatial[r], spatial[s]]
                * (adag[p] @ adag[r] @ a[s] @ a[q])
            )
    matrix += float(nuclear_repulsion) * np.eye(2**n_spin)

    terms = _decompose(matrix, n_spin, tol)
    occupation = [1 if i < n_electrons else 0 for i in range(n_spin)]
    info = MolecularInfo(
        n_qubits=n_spin,
        n_electrons=n_electrons,
        n_orbitals=n_spatial,
        n_terms=len(terms),
        nuclear_repulsion=float(nuclear_repulsion),
        hartree_fock_occupation=occupation,
        active_space=active_space,
    )
    return PauliSum(tuple(terms)), info


def molecular_hamiltonian(
    molecule: Molecule,
    active_space: tuple[int, ...] | None = None,
    tol: float = 1e-10,
) -> tuple[PauliSum, MolecularInfo]:
    """Compute the integrals here, then hand them to :func:`from_integrals`.

    Restricted to s-orbital elements — see the module docstring for why, and for the
    route to take when that is not enough.
    """
    overlap, core, eri, repulsion = _ao_integrals(molecule)
    coefficients, electronic = _rhf(overlap, core, eri, molecule.n_electrons)
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
    hamiltonian, info = from_integrals(
        h, g, molecule.n_electrons, float(repulsion), active_space=active_space, tol=tol
    )
    info.hartree_fock_energy = float(electronic + repulsion)
    return hamiltonian, info
