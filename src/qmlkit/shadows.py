r"""Classical shadows — estimate many observables from few measurements.

Huang, Kueng & Preskill (2020). Measure in a *randomly chosen* basis each shot, invert
the resulting depolarising channel, and the collection of snapshots predicts
:math:`M` observables to fixed accuracy from :math:`O(\log M)` measurements — rather
than measuring each one separately.

On an exact simulator this buys nothing: ``shots=None`` returns every observable
exactly. It earns its place because **measurement cost is what binds on hardware**,
and because it makes that cost visible: :func:`shadow_shot_cost` against
:func:`qmlkit.kernels.kernel_shot_cost` is the comparison worth looking at before
committing to a device run.

    shadow = ClassicalShadow(spec, n_snapshots=2000, seed=0)
    shadow.expectation(qk.Z(0) + 0.5 * qk.ZZ(0, 2))
"""

from __future__ import annotations

from typing import Any

import numpy as np

from qmlkit.core.builder import QCircuit
from qmlkit.core.execute import BackendLike, run_counts
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, as_sum

__all__ = ["ClassicalShadow", "shadow_shot_cost"]

#: single-qubit inverse channel: 3 |b><b| - I, in the measured basis
_BASES = ("X", "Y", "Z")


class ClassicalShadow:
    """A set of randomised single-qubit measurements, and what they predict.

    Each snapshot picks a random Pauli basis per qubit, measures once, and stores
    ``(basis, outcome)``. An observable's estimate then averages over only the
    snapshots whose bases happen to match its support — which is why the cost grows
    with the observable's *locality*, not with how many observables you ask for.
    """

    def __init__(
        self,
        spec: CircuitSpec,
        n_snapshots: int = 1000,
        seed: int | None = None,
        backend: BackendLike = None,
    ) -> None:
        if not spec.is_bound:
            raise ValueError("bind the circuit before taking shadows")
        self.spec = spec
        self.n_qubits = spec.n_qubits
        self.n_snapshots = n_snapshots
        self.backend = backend
        rng = np.random.default_rng(seed)
        #: (n_snapshots, n_qubits) of basis indices, and of +-1 outcomes
        self.bases = rng.integers(0, 3, size=(n_snapshots, self.n_qubits))
        self.outcomes = np.empty((n_snapshots, self.n_qubits), dtype=np.int8)
        self._collect(rng)

    def _collect(self, rng: np.random.Generator) -> None:
        """One shot per snapshot, in that snapshot's randomly chosen basis."""
        for index in range(self.n_snapshots):
            builder = QCircuit(self.n_qubits)
            for op in self.spec.ops:
                builder.apply(op.gate, op.qubits, *[float(p) for p in op.params])
            for qubit in range(self.n_qubits):
                basis = _BASES[self.bases[index, qubit]]
                if basis == "X":
                    builder.h(qubit)
                elif basis == "Y":
                    builder.sdg(qubit)
                    builder.h(qubit)
            counts = run_counts(
                builder.to_spec(), shots=1, seed=int(rng.integers(2**31)), backend=self.backend
            )
            bits = next(iter(counts))
            self.outcomes[index] = [1 if b == "0" else -1 for b in bits]

    def expectation(self, obs: Observable) -> float:
        """Estimate ``<O>`` from the stored snapshots."""
        total = 0.0
        for term in as_sum(obs).terms:
            total += float(term.coeff.real) * self._term_estimate(term)
        return total

    def _term_estimate(self, term: Any) -> float:
        support = [(q, p) for q, p in term.paulis if p != "I"]
        if not support:
            return 1.0
        wanted = np.array([_BASES.index(p) for _, p in support])
        wires = np.array([q for q, _ in support])

        # Only snapshots that happened to measure every wire in the right basis carry
        # information; the inverse channel contributes a factor of 3 per matched wire.
        matched = np.all(self.bases[:, wires] == wanted, axis=1)
        if not matched.any():
            return 0.0
        products = np.prod(self.outcomes[np.ix_(matched, wires)], axis=1)
        return float(3 ** len(support) * products.sum() / self.n_snapshots)

    def __repr__(self) -> str:
        return f"ClassicalShadow(n_qubits={self.n_qubits}, snapshots={self.n_snapshots})"


def shadow_shot_cost(locality: int, n_observables: int, epsilon: float = 0.1) -> int:
    r"""Snapshots for :math:`M` observables of given locality to accuracy ``epsilon``.

    The headline scaling: :math:`O(3^k \log M / \epsilon^2)` — **logarithmic** in how
    many observables you want, exponential only in their locality. Measuring each one
    separately is instead linear in ``M``, which is the trade this whole method makes.
    """
    if locality < 1 or n_observables < 1:
        raise ValueError("locality and n_observables must be at least 1")
    return int(np.ceil(3**locality * np.log(2 * n_observables) / epsilon**2))
