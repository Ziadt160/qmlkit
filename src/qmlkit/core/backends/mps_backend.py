"""Aer's matrix-product-state simulator — width, at the price of entanglement.

Every other backend here stores ``2**n`` amplitudes, which is why 30 qubits is out of
reach: the array alone is 16 GiB. An MPS stores the state as a chain of small tensors
and only grows when the circuit entangles, so a *lightly entangled* circuit stays
cheap at widths a statevector cannot reach at all.

Measured, a linear chain of `ry` + nearest-neighbour `cx`, asking for one expectation:

| qubits | statevector | MPS |
|---|---|---|
| 20 | 0.013 s | 0.001 s |
| 24 | 0.279 s | 0.002 s |
| 28 | 4.97 s | 0.002 s |
| 32 | **fails — needs 65 GiB** | 0.002 s |

**This backend has no statevector, and that is the point.** Asking an MPS for all
``2**n`` amplitudes contracts the whole network and throws away everything it was
doing for you — measured 13.6 s against 0.26 s at 24 qubits, fifty times *slower* than
the statevector simulator it was meant to beat. So ``supports_statevector`` is false:
``adjoint`` and ``backprop`` refuse rather than quietly costing more than they save,
and ``parameter-shift`` works, because a shift rule never inspects a state.

**It is exact only while the entanglement stays inside the bond dimension.** Beyond
that Aer truncates, and the answer is an approximation that does not announce itself.
A hardware-efficient ansatz with a full entangler ring is the worst case for this and
a chain-structured or shallow circuit is the best; :meth:`bond_dimension` reports what
the run actually needed, so the question can be answered rather than assumed.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import BackendNotAvailable
from qmlkit.core.backends.qiskit_backend import QiskitBackend
from qmlkit.core.observables import Observable, as_sum

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qmlkit.core.ir import CircuitSpec

__all__ = ["MPSBackend"]


class MPSBackend(QiskitBackend):
    """Runs qmlkit circuits on ``AerSimulator(method="matrix_product_state")``.

    >>> import qmlkit as qk                                       # doctest: +SKIP
    >>> qk.expectation(spec, qk.Z(0), backend="mps")              # doctest: +SKIP
    """

    name = "mps"

    #: There is no pure state to hand back at a price worth paying, so the gradient
    #: methods that read amplitudes refuse instead of silently costing more than they
    #: save. This is the same flag the density-matrix backends set, for the same
    #: reason: the caller named a backend, and it cannot answer that question.
    supports_statevector = False

    #: It *is* shot-free — exact, given that the bond dimension held.
    supports_exact = True

    def __init__(self, seed: int | None = None, max_bond_dimension: int | None = None) -> None:
        super().__init__(seed)
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "Qiskit Aer is not installed. It is a separate package from Qiskit:\n"
                "    pip install 'qmlkit[aer]'"
            ) from exc
        options: dict[str, Any] = {"method": "matrix_product_state"}
        if max_bond_dimension is not None:
            # Capping the bond is the one setting here that turns an exact simulator
            # into an approximate one, and it does it silently: measured on a 12-qubit
            # hardware-efficient ansatz, capping at 4 moved the expectation from
            # 1.4154454301 to 1.4512716722 with nothing raised and the answer still in
            # range. That is the failure this library exists to refuse, so it is said
            # out loud at the moment the caller asks for it.
            warnings.warn(
                f"max_bond_dimension={max_bond_dimension} makes this backend "
                "APPROXIMATE: once a circuit entangles past that bond, Aer truncates "
                "and the expectation is wrong without anything raising (measured 3.6e-2 "
                "off on a 12-qubit hardware-efficient ansatz at bond 4). Check what a "
                "run actually needed with MPSBackend.bond_dimension(spec), or leave it "
                "unset and let the bond grow.",
                RuntimeWarning,
                stacklevel=2,
            )
            options["matrix_product_state_max_bond_dimension"] = max_bond_dimension
        self._simulator = AerSimulator(**options)
        self.max_bond_dimension = max_bond_dimension

    #: The only gates this library emits that Aer's MPS method cannot take directly.
    #: Everything else in the table goes straight through, including the ``unitary``
    #: a registered gate arrives as — so a custom gate needs no special handling here.
    _NEEDS_DECOMPOSING = frozenset({"crx", "cry", "crz"})

    def _prepared(self, spec: CircuitSpec) -> Any:
        """One circuit, decomposed only if it has to be.

        Transpiling costs ~57 ms against a simulation of ~2 ms, so doing it
        unconditionally would cost thirty times what it saves. Controlled rotations
        are the only gates in qmlkit's table outside the MPS basis, so the check is a
        set membership per op and the transpile happens only for circuits that
        genuinely contain one.
        """
        circuit = self.to_qiskit(spec)
        if any(op.gate in self._NEEDS_DECOMPOSING for op in spec.ops):
            from qiskit import transpile

            circuit = transpile(circuit, self._simulator, optimization_level=0)
        return circuit

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        """Always refuses, and the refusal is the feature.

        Contracting the network to ``2**n`` amplitudes discards the representation
        that makes this backend worth having: measured 13.6 s against 0.26 s at 24
        qubits, and impossible at 32. A backend that answered here would look correct
        and be pointless, which is the harder failure to notice.
        """
        raise NotImplementedError(
            f"the {self.name!r} backend has no statevector - materialising 2**"
            f"{spec.n_qubits} amplitudes is exactly what a matrix product state avoids, "
            "and doing it is slower than using the 'aer' backend in the first place. "
            "Ask for an expectation instead, or use backend='aer'."
        )

    def _pauli_string(self, obs_term: Any, n_qubits: int) -> str:
        """One qmlkit Pauli term as a Qiskit label.

        The one place in this library that hands an observable to an SDK. Everywhere
        else expectation is derived once in the base class from a statevector, which
        is what makes agreement between backends a property; here the simulator
        computes it internally, so the translation has to be right and is asserted
        against the NumPy reference rather than argued for.

        Qiskit labels are most-significant-qubit first and qmlkit is big-endian, so
        qmlkit qubit ``i`` is position ``i`` from the left.
        """
        letters = ["I"] * n_qubits
        for qubit, pauli in obs_term.paulis:
            letters[qubit] = pauli.upper()
        return "".join(letters)

    def expectation(
        self,
        spec: CircuitSpec,
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """``<O>`` computed inside the MPS, without ever forming a statevector."""
        self._check_bound(spec)
        if shots is not None:
            return super().expectation(spec, obs, shots, seed)

        from qiskit.quantum_info import SparsePauliOp

        total = as_sum(obs)
        terms = [(self._pauli_string(t, spec.n_qubits), complex(t.coeff)) for t in total.terms]
        operator = SparsePauliOp.from_list(terms)

        circuit = self._prepared(spec)
        circuit.save_expectation_value(operator, range(spec.n_qubits))
        result = self._simulator.run(circuit).result()
        value = float(np.real(result.data(0)["expectation_value"]))
        return value

    def bond_dimension(self, spec: CircuitSpec) -> int:
        """The largest bond the circuit actually needed.

        The number that says whether this run was exact. An MPS is exact while the
        bond dimension is allowed to grow to what the entanglement demands; once Aer
        truncates, the answer is an approximation and nothing about it looks different.
        """
        circuit = self._prepared(spec)
        circuit.save_matrix_product_state()
        result = self._simulator.run(circuit).result()
        state = result.data(0)["matrix_product_state"]
        return max((len(lam) for lam in state[1]), default=1)
