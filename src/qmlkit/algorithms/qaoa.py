"""QAOA as a solver, not just an ansatz.

The ansatz has been in the zoo since Phase 3. What was missing is the part that
makes it an *algorithm*: turn a combinatorial problem into a cost Hamiltonian,
optimise the angles, then sample the state and read off a bitstring you can act on.

    from qmlkit.algorithms import QAOA

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    result = QAOA(edges, p=2).run(seed=0)
    print(result.bitstring, result.cut_value)

The cost Hamiltonian is an argument, so anything expressible as a Pauli sum — MaxCut,
Max-2-SAT, a weighted graph, a portfolio constraint — is the same call.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.algorithms.hamiltonians import exact_ground_energy, max_cut_hamiltonian
from qmlkit.algorithms.vqe import OPTIMIZERS, Optimizer
from qmlkit.ansatz.library import Ansatz, qaoa_ansatz
from qmlkit.core.execute import BackendLike, expectation, probabilities
from qmlkit.core.observables import Observable, observable_support
from qmlkit.optim import supports_rotosolve

__all__ = ["QAOA", "QAOAResult"]


@dataclass
class QAOAResult:
    """The angles, and — more usefully — the bitstring they point at."""

    energy: float
    theta: npt.NDArray[Any]
    bitstring: str
    probability: float
    history: list[float] = field(default_factory=list)
    exact: float | None = None
    top: list[tuple[str, float]] = field(default_factory=list)

    @property
    def cut_value(self) -> float:
        """For a MaxCut cost, the number of edges the returned bitstring cuts."""
        return -self.energy

    @property
    def approximation_ratio(self) -> float | None:
        """Energy reached over the best possible, when the exact answer is known."""
        if self.exact is None or self.exact == 0:
            return None
        return float(self.energy / self.exact)

    def __repr__(self) -> str:
        ratio = (
            "" if self.approximation_ratio is None else f", ratio={self.approximation_ratio:.4f}"
        )
        return (
            f"QAOAResult(bitstring={self.bitstring!r}, energy={self.energy:.6f}"
            f", p={self.probability:.4f}{ratio})"
        )


class QAOA:
    """Optimise QAOA angles, then sample a solution out of the state.

    Parameters
    ----------
    problem
        Either an edge list (treated as MaxCut) or any cost ``Observable``.
    p
        Rounds. Two angles per round regardless of problem size — which is the
        whole appeal, and also why more rounds is the only way to improve.
    mixer, ansatz
        The structure. ``mixer`` is passed to :func:`qaoa_ansatz`; pass ``ansatz``
        directly to use a warm-started or otherwise non-standard construction.
    """

    def __init__(
        self,
        problem: Observable | Sequence[tuple[int, int]],
        p: int = 1,
        n_qubits: int | None = None,
        mixer: str = "x",
        ansatz: Ansatz | None = None,
        optimizer: str | Optimizer = "gradient-descent",
        backend: BackendLike = None,
        shots: int | None = None,
    ) -> None:
        if isinstance(problem, (list, tuple)) and not problem:
            raise ValueError(
                "QAOA needs a problem: an edge list for MaxCut, or a cost observable. "
                "An empty edge list defines nothing to optimise."
            )
        if isinstance(problem, (list, tuple)) and isinstance(problem[0], tuple):
            edges = [(int(a), int(b)) for a, b in problem]  # type: ignore[misc]
            width = n_qubits or max(max(e) for e in edges) + 1
            self.cost: Observable = max_cut_hamiltonian(edges)
            self.edges: list[tuple[int, int]] | None = edges
        else:
            self.cost = problem  # type: ignore[assignment]
            support = observable_support(self.cost)
            width = n_qubits or (max(support) + 1 if support else 1)
            self.edges = None

        self.n_qubits = width
        self.p = p
        self.ansatz = ansatz or qaoa_ansatz(width, edges=self.edges, p=p, mixer=mixer)
        self.optimizer = optimizer
        self.backend = backend
        self.shots = shots
        self._spec = self.ansatz.build()
        self.n_evaluations = 0

    def energy(self, theta: Sequence[float]) -> float:
        self.n_evaluations += 1
        return float(
            expectation(
                self._spec,
                self.cost,
                theta=np.asarray(theta, dtype=float),
                shots=self.shots,
                backend=self.backend,
            )
        )

    def gradient_of_energy(self, theta: Sequence[float]) -> npt.NDArray[Any]:
        from qmlkit.gradients.dispatch import grad

        return grad(
            self._spec,
            np.asarray(theta, dtype=float),
            self.cost,
            backend=self.backend,
            shots=self.shots,
        )

    def distribution(self, theta: Sequence[float]) -> npt.NDArray[Any]:
        """Outcome probabilities of the optimised state."""
        return probabilities(self._spec.bind(np.asarray(theta, dtype=float)), backend=self.backend)

    def run(
        self,
        theta0: Sequence[float] | None = None,
        seed: int | None = None,
        n_top: int = 5,
        compare_exact: bool | None = None,
        **optimizer_kwargs: Any,
    ) -> QAOAResult:
        start = (
            np.asarray(theta0, dtype=float)
            if theta0 is not None
            else self.ansatz.init("uniform", seed=seed)
        )
        fn = OPTIMIZERS[self.optimizer] if isinstance(self.optimizer, str) else self.optimizer
        if fn is OPTIMIZERS["spsa"]:
            optimizer_kwargs.setdefault("seed", seed)
        if fn is OPTIMIZERS["gradient-descent"]:
            optimizer_kwargs.setdefault("grad", self.gradient_of_energy)
        if fn is OPTIMIZERS["rotosolve"] and not supports_rotosolve(self._spec):
            warnings.warn(
                "Rotosolve assumes each angle drives a single Pauli rotation, but this "
                "circuit shares an angle across gates that do not compose - QAOA's cost "
                "angle drives one rz per edge. Rotosolve will converge immediately on "
                "the wrong point. Use 'gradient-descent' or 'spsa'.",
                UserWarning,
                stacklevel=2,
            )
        self.n_evaluations = 0
        theta, history = fn(self.energy, start, **optimizer_kwargs)

        # The angles are a means; the bitstring is the answer. Read the most likely
        # outcomes off the optimised state rather than reporting only an energy.
        probs = self.distribution(theta)
        order = np.argsort(probs)[::-1]
        top = [(format(int(i), f"0{self.n_qubits}b"), float(probs[i])) for i in order[:n_top]]
        best = top[0]

        if compare_exact is None:
            compare_exact = self.n_qubits <= 12
        exact = exact_ground_energy(self.cost, self.n_qubits) if compare_exact else None

        return QAOAResult(
            energy=float(history[-1]),
            theta=theta,
            bitstring=best[0],
            probability=best[1],
            history=list(history),
            exact=exact,
            top=top,
        )

    def cut_size(self, bitstring: str) -> int:
        """Edges cut by an assignment — the classical objective, computed classically."""
        if self.edges is None:
            raise ValueError("cut_size only applies when the problem was given as edges")
        return sum(1 for a, b in self.edges if bitstring[a] != bitstring[b])

    def __repr__(self) -> str:
        return f"QAOA(n_qubits={self.n_qubits}, p={self.p}, P={self.ansatz.n_weights})"
