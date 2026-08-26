"""The backend protocol.

A backend supplies *primitives*; this base class supplies *semantics*. A simulator
backend needs to implement only :meth:`statevector` — sampling, basis rotation and
expectation values are derived here, so every backend agrees on what a shot is and
what an expectation means. A sampling-only device overrides :meth:`counts` instead.

That split is what makes cross-backend equivalence testable rather than hopeful:
if two backends disagree, the disagreement is in the circuit translation, not in
four separate re-implementations of the measurement logic.
"""

from __future__ import annotations

import numpy as np

from qmlkit.core.backends._sampling import sample_counts_from_probs
from qmlkit.core.ir import CircuitSpec, Op
from qmlkit.core.observables import (
    Observable,
    PauliString,
    as_sum,
    basis_rotation,
    expectation_from_counts,
    expectation_from_statevector,
)


class BackendNotAvailable(RuntimeError):
    """Raised when a backend's underlying SDK is not installed or not importable."""


class Backend:
    """A device or simulator that can run a :class:`CircuitSpec`."""

    #: short name used by the registry
    name: str = "backend"
    #: can hand back a full statevector (simulators only)
    supports_statevector: bool = False
    #: can produce exact, shot-free expectation values
    supports_exact: bool = False

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------ primitives --
    def statevector(self, spec: CircuitSpec) -> np.ndarray:
        """Final state as a flat ``2**n`` complex vector, qubit 0 most significant."""
        raise NotImplementedError(
            f"the {self.name!r} backend cannot produce a statevector; use shots=N to sample instead"
        )

    def counts(self, spec: CircuitSpec, shots: int, seed: int | None = None) -> dict[str, int]:
        """Sample the computational basis. Keys are ``n_qubits``-wide bitstrings.

        The default samples the exact probability distribution — correct for any
        statevector simulator. A shot-based device overrides this.
        """
        self._check_bound(spec)
        if shots <= 0:
            raise ValueError("shots must be positive")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        return sample_counts_from_probs(self.probabilities(spec), shots, spec.n_qubits, rng)

    def probabilities(self, spec: CircuitSpec) -> np.ndarray:
        """Exact outcome probabilities over the ``2**n`` basis states."""
        return np.abs(self.statevector(spec)) ** 2

    # ------------------------------------------------------------- semantics --
    def expectation(
        self,
        spec: CircuitSpec,
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """``<O>``. ``shots=None`` means exact, where the backend supports it."""
        self._check_bound(spec)
        if shots is None:
            if not self.supports_exact:
                raise ValueError(
                    f"the {self.name!r} backend has no exact mode; pass shots=N to sample"
                )
            return expectation_from_statevector(obs, self.statevector(spec), spec.n_qubits)
        return sum(self._sampled_term(spec, t, shots, seed) for t in as_sum(obs).terms)

    def _sampled_term(
        self, spec: CircuitSpec, term: PauliString, shots: int, seed: int | None
    ) -> float:
        if not term.paulis:  # identity term carries no measurement
            return float(term.coeff.real)
        rotated = CircuitSpec(
            n_qubits=spec.n_qubits,
            ops=spec.ops + tuple(Op(g, q) for g, q in basis_rotation(term)),
            n_params=0,
        )
        return expectation_from_counts(term, self.counts(rotated, shots, seed), spec.n_qubits)

    # ------------------------------------------------------------- utilities --
    @staticmethod
    def _check_bound(spec: CircuitSpec) -> None:
        if not spec.is_bound:
            raise ValueError(
                "circuit still has free parameters; call spec.bind(theta) before running it"
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
