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

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends._sampling import sample_counts_from_probs
from qmlkit.core.ir import CircuitSpec, Op
from qmlkit.core.observables import (
    Observable,
    PauliString,
    basis_rotation,
    expectation_from_counts,
    expectation_from_statevector,
    expectation_from_statevectors,
    group_qubit_wise_commuting,
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
    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
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

    def probabilities(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        """Exact outcome probabilities over the ``2**n`` basis states."""
        return np.abs(self.statevector(spec)) ** 2

    # ----------------------------------------------------------------- batch --
    #: Rows evaluated per chunk. A batched parameter-shift gradient fans out to
    #: ``batch x 2P`` circuits, which is where an unbounded batch would exhaust memory
    #: on a simulator and exceed a provider's job limit on a device. Chunking is the
    #: same answer to both.
    max_batch_rows: int = 4096

    def statevector_batch_slots(
        self, spec: CircuitSpec, slot_angles: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """States for one circuit at many **slot-angle** vectors: ``(batch, 2**n)``.

        Slot space rather than logical-parameter space is the primitive because that is
        what differentiation needs: a shift rule moves one *occurrence* of a parameter,
        and a weight-tied parameter has several. Anything expressed in logical
        parameters cannot say that.

        The default binds and simulates one row at a time, so every backend has a
        working implementation the moment it can produce a statevector. A backend that
        can do better overrides this one method and everything above it — batched
        expectations, batched gradients, the torch layer — speeds up with it.
        """
        rows = np.atleast_2d(np.asarray(slot_angles, dtype=float))
        states: npt.NDArray[Any] = np.stack(
            [self.statevector(spec.with_slot_angles(row)) for row in rows]
        )
        return states

    def statevector_batch(
        self, spec: CircuitSpec, thetas: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """States for one circuit at many logical parameter vectors."""
        values = np.atleast_2d(np.asarray(thetas, dtype=float))
        return self.statevector_batch_slots(spec, spec.bind_slots_batch(values))

    def expectation_over_slots(
        self,
        spec: CircuitSpec,
        slot_angles: npt.NDArray[Any],
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[Any]:
        """``<O>`` at many slot-angle vectors, in chunks of :attr:`max_batch_rows`.

        This is the one call a batched gradient makes, and the one a device would turn
        into a job submission.
        """
        rows = np.atleast_2d(np.asarray(slot_angles, dtype=float))
        if shots is not None:
            return np.array(
                [
                    self.expectation(spec.with_slot_angles(row), obs, shots, seed)
                    for row in rows
                ],
                dtype=float,
            )
        if not self.supports_exact:
            raise ValueError(
                f"the {self.name!r} backend has no exact mode; pass shots=N to sample"
            )
        out = np.empty(rows.shape[0], dtype=float)
        for start in range(0, rows.shape[0], self.max_batch_rows):
            block = rows[start : start + self.max_batch_rows]
            states = self.statevector_batch_slots(spec, block)
            out[start : start + block.shape[0]] = expectation_from_statevectors(
                obs, states, spec.n_qubits
            )
        return out

    def expectation_over(
        self,
        spec: CircuitSpec,
        thetas: npt.NDArray[Any],
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[Any]:
        """``<O>`` for one circuit at many logical parameter vectors."""
        values = np.atleast_2d(np.asarray(thetas, dtype=float))
        return self.expectation_over_slots(
            spec, spec.bind_slots_batch(values), obs, shots, seed
        )

    # ------------------------------------------------------------- semantics --
    def expectation(
        self,
        spec: CircuitSpec,
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """``<O>``. ``shots=None`` means exact, where the backend supports it.

        When sampling, terms are partitioned into qubit-wise-commuting groups and each
        group costs **one** circuit rather than one per term. On a simulator that is a
        modest saving; on a device, where circuit count is the binding constraint, it
        is the difference between ``Z0 + Z1 + Z2 + Z0Z2`` costing four circuits and
        costing one.
        """
        self._check_bound(spec)
        if shots is None:
            if not self.supports_exact:
                raise ValueError(
                    f"the {self.name!r} backend has no exact mode; pass shots=N to sample"
                )
            return expectation_from_statevector(obs, self.statevector(spec), spec.n_qubits)
        return sum(
            self._sampled_group(spec, group, shots, seed)
            for group in group_qubit_wise_commuting(obs)
        )

    def _sampled_group(
        self, spec: CircuitSpec, group: list[PauliString], shots: int, seed: int | None
    ) -> float:
        """One circuit for a whole group of mutually qubit-wise-commuting terms."""
        # identity terms carry no measurement at all
        value = sum(float(t.coeff.real) for t in group if not t.paulis)
        measured = [t for t in group if t.paulis]
        if not measured:
            return value

        # Qubit-wise commuting means every term in the group agrees on the Pauli it
        # wants on any qubit they share, so one rotation diagonalises all of them.
        wanted: dict[int, str] = {}
        for term in measured:
            for qubit, pauli in term.paulis:
                if pauli != "I":
                    wanted[qubit] = pauli
        rotation: list[tuple[str, tuple[int, ...]]] = []
        for qubit, pauli in sorted(wanted.items()):
            rotation.extend(basis_rotation(PauliString(((qubit, pauli),), 1.0)))

        rotated = CircuitSpec(
            n_qubits=spec.n_qubits,
            ops=spec.ops + tuple(Op(g, q) for g, q in rotation),
            n_params=0,
        )
        counts = self.counts(rotated, shots, seed)
        return value + sum(
            expectation_from_counts(term, counts, spec.n_qubits) for term in measured
        )

    # ------------------------------------------------------------- utilities --
    @staticmethod
    def _check_bound(spec: CircuitSpec) -> None:
        if not spec.is_bound:
            raise ValueError(
                "circuit still has free parameters; call spec.bind(theta) before running it"
            )

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"
