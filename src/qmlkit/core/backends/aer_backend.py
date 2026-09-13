"""Aer's statevector simulator — the fast Qiskit path.

Qiskit ships two ways to get a statevector and they are not close in cost.
``qiskit.quantum_info.Statevector`` is the reference implementation: exact, pure
Python and NumPy, and what the ``qiskit`` backend uses. ``AerSimulator`` is the C++
one. Same answer, very different wall clock, and nothing in the library was reaching
the second.

So there are three Qiskit-flavoured backends and it is worth being blunt about which
is which:

| name | simulator | for |
|---|---|---|
| ``qiskit`` | ``quantum_info.Statevector`` | the reference; exact, slow, no extra install |
| ``aer`` | ``AerSimulator(method="statevector")`` | the same answer, faster |
| ``qiskit-aer`` | ``AerSimulator(method="density_matrix")`` | noise; a *different* question |

This class changes only where the amplitudes come from. Circuit translation, qubit
order, the matrix fallback that lets a registered gate run here, and every
measurement semantic are inherited unchanged from :class:`QiskitBackend` and the base
class — which is what makes "same answer" a property rather than a hope, and is
asserted by the cross-backend suite like any other backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import BackendNotAvailable
from qmlkit.core.backends.qiskit_backend import QiskitBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qmlkit.core.ir import CircuitSpec

__all__ = ["AerBackend"]


class AerBackend(QiskitBackend):
    """Runs qmlkit circuits on ``AerSimulator(method="statevector")``.

    >>> import qmlkit as qk                                    # doctest: +SKIP
    >>> qk.expectation(spec, qk.Z(0), backend="aer")           # doctest: +SKIP
    """

    name = "aer"

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "Qiskit Aer is not installed. It is a separate package from Qiskit:\n"
                "    pip install 'qmlkit[aer]'"
            ) from exc
        self._simulator = AerSimulator(method="statevector")

    def _prepared(self, spec: CircuitSpec) -> Any:
        """One qmlkit circuit, ready for Aer.

        **Deliberately not transpiled.** Aer simulates this library's gate set
        directly, including the ``UnitaryGate`` a registered gate arrives as, so the
        transpiler has nothing to contribute here — and it is not cheap: measured at
        **57 ms against 1.1 ms** for the simulation it was preparing, fifty times the
        work it saved. Transpiling per circuit made this backend thirty times slower
        than the pure-Python ``quantum_info.Statevector`` it exists to beat.

        Nothing relabels the qubits either, which is the other reason the density
        backend transpiles and this one must not.
        """
        circuit = self.to_qiskit(spec)
        circuit.save_statevector()
        return circuit

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        result = self._simulator.run(self._prepared(spec)).result()
        return np.asarray(result.get_statevector(), dtype=complex)

    def statevector_batch_slots(
        self, spec: CircuitSpec, slot_angles: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Every row in **one** Aer job.

        This override is the whole reason this backend exists. Aer's simulation kernel
        is C++ and fast; its *per-job* cost — transpiling, building a job, unpacking a
        result — is Python and is not. On the circuits this library runs, one job per
        row makes Aer slower than the pure-Python ``quantum_info.Statevector`` it was
        supposed to beat. Submitting the whole batch as one job is what turns that
        around, and it is also exactly the shape a real provider wants.
        """
        from qmlkit.progress import task as progress_task

        rows = np.atleast_2d(np.asarray(slot_angles, dtype=float))
        with progress_task(f"{self.name} circuits", len(rows)) as tracked:
            circuits = []
            for row in rows:
                circuits.append(self._prepared(spec.with_slot_angles(row)))
                tracked.advance()
            result = self._simulator.run(circuits).result()
        return np.stack(
            [np.asarray(result.get_statevector(i), dtype=complex) for i in range(len(rows))]
        )
