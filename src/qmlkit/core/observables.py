"""Pauli observables.

One ``expectation()`` that takes an observable and is correct for any register
width — replacing the per-lecture ``expz(counts)`` helpers, which divide by
``n0 + n1`` and so silently misreport on more than one qubit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["PauliString", "PauliSum", "I", "X", "Y", "Z", "ZZ", "Observable"]

_PAULIS = ("I", "X", "Y", "Z")


@dataclass(frozen=True)
class PauliString:
    """A weighted tensor product of Paulis, e.g. ``0.5 * Z0 X2``.

    Qubits not named act as identity.
    """

    paulis: tuple[tuple[int, str], ...] = ()
    coeff: complex = 1.0

    def __post_init__(self) -> None:
        seen = set()
        for q, p in self.paulis:
            if p not in _PAULIS:
                raise ValueError(f"unknown Pauli {p!r}; expected one of {_PAULIS}")
            if q < 0:
                raise ValueError(f"negative qubit index {q}")
            if q in seen:
                raise ValueError(f"qubit {q} appears twice in a Pauli string")
            seen.add(q)

    # ------------------------------------------------------------- construction
    @classmethod
    def from_label(cls, label: str, coeff: complex = 1.0) -> PauliString:
        """``PauliString.from_label("ZIX")`` -> Z on qubit 0, X on qubit 2."""
        items = tuple((i, ch.upper()) for i, ch in enumerate(label) if ch.upper() != "I")
        return cls(items, coeff)

    @property
    def label_qubits(self) -> tuple[int, ...]:
        return tuple(q for q, p in self.paulis if p != "I")

    def support(self) -> tuple[int, ...]:
        return tuple(sorted(self.label_qubits))

    # ---------------------------------------------------------------- algebra --
    def __mul__(self, other: PauliString | float | complex) -> PauliString:
        if isinstance(other, (int, float, complex)):
            return PauliString(self.paulis, self.coeff * other)
        overlap = {q for q, _ in self.paulis} & {q for q, _ in other.paulis}
        if overlap:
            raise ValueError(
                f"multiplying Pauli strings that overlap on qubits {sorted(overlap)} is not "
                "supported; build the product term explicitly"
            )
        return PauliString(self.paulis + other.paulis, self.coeff * other.coeff)

    __rmul__ = __mul__

    def __neg__(self) -> PauliString:
        return PauliString(self.paulis, -self.coeff)

    def __add__(self, other: Observable) -> PauliSum:
        return PauliSum((self,)) + other

    def __repr__(self) -> str:
        body = "I" if not self.paulis else " ".join(f"{p}{q}" for q, p in sorted(self.paulis))
        return body if self.coeff == 1 else f"{self.coeff:g}*{body}"


@dataclass(frozen=True)
class PauliSum:
    """A linear combination of Pauli strings."""

    terms: tuple[PauliString, ...] = ()

    def __add__(self, other: Observable) -> PauliSum:
        if isinstance(other, PauliString):
            return PauliSum(self.terms + (other,))
        return PauliSum(self.terms + other.terms)

    def __mul__(self, k: float | complex) -> PauliSum:
        return PauliSum(tuple(t * k for t in self.terms))

    __rmul__ = __mul__

    def support(self) -> tuple[int, ...]:
        return tuple(sorted({q for t in self.terms for q in t.support()}))

    def __repr__(self) -> str:
        return " + ".join(repr(t) for t in self.terms) or "0"


Observable = PauliString | PauliSum


# ------------------------------------------------------------------ shorthands
def I() -> PauliString:  # noqa: E743 - deliberate single-letter API
    """The identity observable."""
    return PauliString()


def X(q: int) -> PauliString:
    return PauliString(((q, "X"),))


def Y(q: int) -> PauliString:
    return PauliString(((q, "Y"),))


def Z(q: int) -> PauliString:
    return PauliString(((q, "Z"),))


def ZZ(a: int, b: int) -> PauliString:
    return PauliString(((a, "Z"), (b, "Z")))


def as_sum(obs: Observable) -> PauliSum:
    return PauliSum((obs,)) if isinstance(obs, PauliString) else obs


# ------------------------------------------------------------------ evaluation
def _pauli_matrix(p: str) -> np.ndarray:
    return {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }[p]


def expectation_from_statevector(obs: Observable, state: np.ndarray, n_qubits: int) -> float:
    """Exact <psi|O|psi>. ``state`` is a flat 2**n complex vector, qubit 0 most significant."""
    psi = np.asarray(state, dtype=complex).reshape((2,) * n_qubits)
    total = 0.0 + 0.0j
    for term in as_sum(obs).terms:
        out = psi
        for q, p in term.paulis:
            if p == "I":
                continue
            out = np.tensordot(_pauli_matrix(p), out, axes=([1], [q]))
            out = np.moveaxis(out, 0, q)
        total += term.coeff * np.vdot(psi, out)
    if abs(total.imag) > 1e-9:  # pragma: no cover - guards a genuine bug
        raise ValueError(f"non-real expectation {total!r}; observable is not Hermitian")
    return float(total.real)


def diagonal_eigenvalues(term: PauliString, n_qubits: int) -> np.ndarray:
    """+-1 eigenvalue per computational basis state, for a Z-only Pauli string."""
    if any(p not in ("I", "Z") for _, p in term.paulis):
        raise ValueError("diagonal_eigenvalues expects a Z-only string (rotate the basis first)")
    idx = np.arange(2**n_qubits)
    signs = np.ones(2**n_qubits, dtype=float)
    for q, p in term.paulis:
        if p == "Z":
            bit = (idx >> (n_qubits - 1 - q)) & 1  # qubit 0 is most significant
            signs *= np.where(bit == 0, 1.0, -1.0)
    return signs


def expectation_from_counts(term: PauliString, counts: Mapping[str, int], n_qubits: int) -> float:
    """<P> from measurement counts already taken in the term's own basis."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot take an expectation from zero shots")
    acc = 0.0
    support = {q for q, p in term.paulis if p != "I"}
    for bits, n in counts.items():
        if len(bits) != n_qubits:
            raise ValueError(f"bitstring {bits!r} does not match {n_qubits} qubits")
        parity = sum(bits[q] == "1" for q in support) % 2
        acc += n * (1.0 if parity == 0 else -1.0)
    return float(term.coeff.real * acc / total)


def basis_rotation(term: PauliString) -> list[tuple[str, tuple[int, ...]]]:
    """Gates that rotate ``term`` into the computational (Z) basis.

    X is diagonalised by H; Y by S-dagger then H.
    """
    ops: list[tuple[str, tuple[int, ...]]] = []
    for q, p in term.paulis:
        if p == "X":
            ops.append(("h", (q,)))
        elif p == "Y":
            ops.append(("sdg", (q,)))
            ops.append(("h", (q,)))
    return ops


def group_qubit_wise_commuting(obs: Observable) -> list[list[PauliString]]:
    """Partition terms into qubit-wise-commuting groups (one circuit per group).

    Simple greedy first-fit. Cheap, and enough to matter for multi-term observables.
    """
    groups: list[list[PauliString]] = []
    for term in as_sum(obs).terms:
        placed = False
        for g in groups:
            if all(_qwc(term, other) for other in g):
                g.append(term)
                placed = True
                break
        if not placed:
            groups.append([term])
    return groups


def _qwc(a: PauliString, b: PauliString) -> bool:
    da: dict[int, str] = dict(a.paulis)
    db: dict[int, str] = dict(b.paulis)
    return all(da[q] == db[q] for q in set(da) & set(db))


def observable_support(obs: Observable) -> tuple[int, ...]:
    return as_sum(obs).support()


def iter_terms(obs: Observable) -> Iterable[PauliString]:
    return as_sum(obs).terms


def required_qubits(obs: Observable) -> int:
    sup: Sequence[int] = observable_support(obs)
    return (max(sup) + 1) if sup else 1
