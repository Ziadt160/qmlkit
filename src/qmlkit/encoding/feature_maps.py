r"""Pauli feature maps.

Each term :math:`S` of a feature map contributes

.. math::  \exp\!\left(-i\,\phi_S(x) \prod_{j \in S} P_j\right)

realised as ``W . (CX ladder, Rz(2 phi), CX ladder) . W^dagger``, where ``W`` is the
basis change that sends each Pauli to ``Z``. The default data map is the standard
one: :math:`\phi_{\{i\}}(x) = x_i` for singletons and
:math:`\phi_S(x) = \prod_{j \in S}(\pi - x_j)` for higher-order terms.

**This module supplies the two helpers the lecture notebook is missing.**
``Lecture3``'s ``pauli_feature_map`` calls ``_basis(...)`` and ``_phi(...)``; neither
is defined anywhere in that repository, so the cell cannot run. Here they are
:func:`basis_change` and :func:`default_data_map`, and the map is tested against
the analytic kernel it is supposed to induce.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations

import numpy as np

from qmlkit.core.builder import QCircuit, entangler_pairs
from qmlkit.core.ir import CircuitSpec, ParamRef

__all__ = [
    "FeatureMap",
    "PauliFeatureMap",
    "ZFeatureMap",
    "ZZFeatureMap",
    "AngleFeatureMap",
    "default_data_map",
    "basis_change",
    "pauli_terms",
]

DataMap = Callable[[np.ndarray, tuple[int, ...]], float]


# --------------------------------------------------------------------------- #
# the two helpers Lecture 3 references but never defines
# --------------------------------------------------------------------------- #
def default_data_map(x: np.ndarray, indices: tuple[int, ...]) -> float:
    """The standard data map: ``x_i`` for one index, ``prod(pi - x_j)`` for more.

    The product form is what makes higher-order terms *nonlinear* in the features —
    a linear map there would leave the kernel factorisable and the entanglers
    pointless.
    """
    if len(indices) == 1:
        return float(x[indices[0]])
    out = 1.0
    for i in indices:
        out *= float(np.pi - x[i])
    return out


def basis_change(pauli: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Gates that rotate ``pauli`` into the Z basis, and the gates that undo it.

    Returns ``(forward, inverse)`` as gate-name tuples applied in circuit order.
    ``X = H Z H`` so ``W = H``; ``Y = (SH) Z (SH)^dagger`` so ``W = H S^dagger``,
    which in circuit order is ``sdg`` then ``h``.
    """
    p = pauli.upper()
    if p in ("I", "Z"):
        return (), ()
    if p == "X":
        return ("h",), ("h",)
    if p == "Y":
        return ("sdg", "h"), ("h", "s")
    raise ValueError(f"unknown Pauli {pauli!r}; expected I, X, Y or Z")


def pauli_terms(
    paulis: Sequence[str], n_features: int, entanglement: str = "linear"
) -> list[tuple[tuple[int, ...], str]]:
    """Expand Pauli strings into concrete ``(qubit indices, pauli string)`` terms.

    A one-character string like ``"Z"`` becomes one term per qubit. A two-character
    string like ``"ZZ"`` follows the entanglement pattern. Longer strings enumerate
    combinations of that size.
    """
    terms: list[tuple[tuple[int, ...], str]] = []
    for pauli in paulis:
        k = len(pauli)
        if k == 0:
            raise ValueError("empty Pauli string")
        if k == 1:
            idx_sets: list[tuple[int, ...]] = [(i,) for i in range(n_features)]
        elif k == 2:
            pattern = "chain" if entanglement == "linear" else entanglement
            idx_sets = [tuple(p) for p in entangler_pairs(n_features, pattern)]
        else:
            idx_sets = list(combinations(range(n_features), k))
        for idx in idx_sets:
            terms.append((idx, pauli.upper()))
    return terms


# --------------------------------------------------------------------------- #
# base
# --------------------------------------------------------------------------- #
class FeatureMap:
    """Turns a feature vector into a circuit.

    Subclasses implement :meth:`build`. ``adjoint`` comes free from the IR, which
    is what the fidelity kernel's compute-uncompute test needs.
    """

    n_features: int
    n_qubits: int

    def build(self, x: Sequence[float]) -> CircuitSpec:
        """The circuit for a concrete feature vector."""
        return self._emit(self.angles(x))

    # -- the three methods that make df/dx work through a nonlinear map --------
    def angles(self, x: Sequence[float]) -> np.ndarray:  # pragma: no cover - abstract
        """The rotation angles this map derives from ``x``."""
        raise NotImplementedError

    @property
    def n_angles(self) -> int:  # pragma: no cover - abstract
        """How many distinct angles the map uses."""
        raise NotImplementedError

    def _emit(self, angles: Sequence[object]) -> CircuitSpec:  # pragma: no cover - abstract
        """Build the circuit from angle values, which may be floats or ParamRefs."""
        raise NotImplementedError

    def build_parametric(self, offset: int = 0) -> CircuitSpec:
        """The circuit with each encoding angle as a free parameter.

        This is what lets a gradient flow back to the *data*: the circuit is
        differentiated with respect to its angles, and the chain rule to ``x`` is
        finished classically by :meth:`angle_jacobian`.
        """
        return self._emit([ParamRef(offset + i) for i in range(self.n_angles)])

    def angle_jacobian(self, x: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """``d(angle) / d(feature)``, shape ``(n_angles, n_features)``.

        The default differences the *classical* data map — no circuits involved, so
        it costs nothing quantum. Override it when a closed form is available.
        """
        arr = self._validate(x)
        jac = np.zeros((self.n_angles, arr.size), dtype=float)
        for i in range(arr.size):
            plus = arr.copy()
            minus = arr.copy()
            plus[i] += eps
            minus[i] -= eps
            jac[:, i] = (self.angles(plus) - self.angles(minus)) / (2 * eps)
        return jac

    def adjoint(self, x: Sequence[float]) -> CircuitSpec:
        """``U(x)^dagger`` — the second half of an inversion-test kernel."""
        return self.build(x).adjoint()

    def __call__(self, x: Sequence[float]) -> CircuitSpec:
        return self.build(x)

    def _validate(self, x: Sequence[float]) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
        if arr.size != self.n_features:
            raise ValueError(
                f"{type(self).__name__} expects {self.n_features} features, got {arr.size}"
            )
        return arr

    def resources(self) -> dict[str, object]:
        """Gate counts and depth for a representative input."""
        return self.build(np.zeros(self.n_features)).resources()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_features={self.n_features}, n_qubits={self.n_qubits})"


# --------------------------------------------------------------------------- #
# the Pauli family
# --------------------------------------------------------------------------- #
class PauliFeatureMap(FeatureMap):
    """The general Pauli feature map — a runnable version of ``Lecture3`` cell 42.

    Parameters
    ----------
    n_features
        One qubit per feature.
    paulis
        Pauli strings to include, e.g. ``("Z", "ZZ")``.
    reps
        How many times to repeat the whole block. More reps means higher reachable
        frequencies, at proportional depth.
    entanglement
        Pattern for two-body terms: ``linear``/``chain``, ``ring``, ``full``, or
        ``alternating``.
    data_map
        Override the default :func:`default_data_map`.
    """

    def __init__(
        self,
        n_features: int,
        paulis: Sequence[str] = ("Z", "ZZ"),
        reps: int = 2,
        entanglement: str = "linear",
        data_map: DataMap | None = None,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if reps < 1:
            raise ValueError("reps must be at least 1")
        self.n_features = n_features
        self.n_qubits = n_features
        self.paulis = tuple(paulis)
        self.reps = reps
        self.entanglement = entanglement
        self.data_map = data_map or default_data_map
        self.terms = pauli_terms(self.paulis, n_features, entanglement)

    @property
    def n_angles(self) -> int:
        """One angle per term. Reps reuse the same angles, so they add depth only."""
        return len(self.terms)

    def angles(self, x: Sequence[float]) -> np.ndarray:
        arr = self._validate(x)
        return np.array([2.0 * self.data_map(arr, idx) for idx, _ in self.terms], dtype=float)

    def angle_jacobian(self, x: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """Closed form for the standard data map; falls back to differencing otherwise."""
        if self.data_map is not default_data_map:
            return super().angle_jacobian(x, eps)
        arr = self._validate(x)
        jac = np.zeros((len(self.terms), arr.size), dtype=float)
        for row, (indices, _) in enumerate(self.terms):
            if len(indices) == 1:
                jac[row, indices[0]] = 2.0  # d(2 x_i)/dx_i
            else:
                for i in indices:  # d/dx_i of 2 * prod_j (pi - x_j)
                    others = 1.0
                    for j in indices:
                        if j != i:
                            others *= float(np.pi - arr[j])
                    jac[row, i] = -2.0 * others
        return jac

    def _emit(self, angles: Sequence[object]) -> CircuitSpec:
        qc = QCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qc.h(i)
            for term_index, (indices, pauli) in enumerate(self.terms):
                self._append_term(qc, angles[term_index], indices, pauli)
        return qc.to_spec()

    def _append_term(
        self, qc: QCircuit, angle: object, indices: tuple[int, ...], pauli: str
    ) -> None:
        letters = pauli if len(pauli) == len(indices) else pauli * len(indices)
        # rotate every qubit in the term into the Z basis
        undo: list[tuple[str, int]] = []
        for q, p in zip(indices, letters, strict=True):
            forward, inverse = basis_change(p)
            for gate in forward:
                qc.apply(gate, q)
            undo.extend((gate, q) for gate in inverse)

        if len(indices) == 1:
            qc.rz(indices[0], angle)
        else:
            pairs = list(zip(indices[:-1], indices[1:], strict=False))  # deliberately offset
            for a, b in pairs:
                qc.cx(a, b)
            qc.rz(indices[-1], angle)
            for a, b in reversed(pairs):
                qc.cx(a, b)

        for gate, q in undo:
            qc.apply(gate, q)


class ZFeatureMap(PauliFeatureMap):
    """First-order, no entanglement — so its kernel factorises over features."""

    def __init__(self, n_features: int, reps: int = 2) -> None:
        super().__init__(n_features, paulis=("Z",), reps=reps)


class ZZFeatureMap(PauliFeatureMap):
    """First order plus entangling ZZ couplings — the kernel stops factorising."""

    def __init__(self, n_features: int, reps: int = 2, entanglement: str = "linear") -> None:
        super().__init__(n_features, paulis=("Z", "ZZ"), reps=reps, entanglement=entanglement)


class AngleFeatureMap(FeatureMap):
    """One rotation per feature, optionally followed by an entangling layer.

    The plainest map there is, and the one whose kernel has a closed form:
    ``cos^2((x - x')/2)`` per feature when ``entangle=False``.
    """

    def __init__(
        self,
        n_features: int,
        rotation: str = "ry",
        entangle: bool = True,
        entanglement: str = "chain",
        reps: int = 1,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        self.n_features = n_features
        self.n_qubits = n_features
        self.rotation = rotation
        self.entangle = entangle
        self.entanglement = entanglement
        self.reps = reps

    @property
    def n_angles(self) -> int:
        return self.n_features

    def angles(self, x: Sequence[float]) -> np.ndarray:
        return self._validate(x)

    def angle_jacobian(self, x: Sequence[float], eps: float = 1e-6) -> np.ndarray:
        """The map is the identity, so the Jacobian is too."""
        self._validate(x)
        return np.eye(self.n_features)

    def _emit(self, angles: Sequence[object]) -> CircuitSpec:
        qc = QCircuit(self.n_qubits)
        for _ in range(self.reps):
            for i in range(self.n_qubits):
                qc.apply(self.rotation, i, angles[i])
            if self.entangle and self.n_qubits > 1:
                qc.entangle(self.entanglement)
        return qc.to_spec()
