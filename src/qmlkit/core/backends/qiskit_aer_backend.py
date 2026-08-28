"""Qiskit Aer's density-matrix simulator, with a noise model.

The circuit translation is
:class:`~qmlkit.core.backends.qiskit_backend.QiskitBackend`'s, inherited unchanged -
including the qubit remap that makes Qiskit's little-endian index agree with qmlkit's,
which is the part an independent second copy would get wrong.

``qiskit-aer`` is a separate distribution from ``qiskit``: installing the Qiskit
backend does not give you this one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import BackendNotAvailable
from qmlkit.core.backends.noisy import NoisyBackend
from qmlkit.core.backends.qiskit_backend import QiskitBackend
from qmlkit.core.ir import CircuitSpec

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit


class QiskitAerBackend(NoisyBackend, QiskitBackend):
    """Runs qmlkit circuits on ``AerSimulator(method="density_matrix")``.

    ``noise`` takes a ``qiskit_aer.noise.NoiseModel`` - built by hand, or lifted off
    a real device with ``NoiseModel.from_backend(...)`` - or ``None`` for an exact
    mixed-state simulation.

    ``None`` is the case that keeps the rest honest: it must reproduce the pure-state
    backends to machine precision, and ``tests/test_noisy_backends.py`` asserts it.

    >>> from qiskit_aer.noise import NoiseModel, depolarizing_error   # doctest: +SKIP
    >>> nm = NoiseModel()                                             # doctest: +SKIP
    >>> nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 1), ["rx", "ry", "rz"])
    ...                                                               # doctest: +SKIP
    >>> be = qk.get_backend("qiskit-aer", noise=nm)                   # doctest: +SKIP
    """

    name = "qiskit-aer"

    def __init__(self, seed: int | None = None, noise: Any = None) -> None:
        super().__init__(seed)
        try:
            from qiskit_aer import AerSimulator
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "Qiskit Aer is not installed. It is a separate package from Qiskit:\n"
                "    pip install 'qmlkit[aer]'"
            ) from exc
        self._simulator = AerSimulator(method="density_matrix", noise_model=noise)
        self.noise = noise

    def density_matrix(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        from qiskit import transpile

        circuit = self.to_qiskit(spec)
        circuit.save_density_matrix()
        # optimization_level=0 and no coupling map means a trivial layout, so the qubit
        # indices the translation chose survive transpilation. The noise model's basis
        # gates are what the circuit is rewritten *into*, which is the point: noise is
        # attached to the gates a device would actually run.
        compiled = transpile(circuit, self._simulator, optimization_level=0)
        result = self._simulator.run(compiled).result()
        return np.asarray(result.data(0)["density_matrix"], dtype=complex)

    def to_qiskit(self, spec: CircuitSpec) -> QuantumCircuit:
        """The same translation the pure-state backend uses; noise is attached by the
        simulator, not written into the circuit."""
        return super().to_qiskit(spec)

    def __repr__(self) -> str:
        noise = "None" if self.noise is None else type(self.noise).__name__
        return f"<QiskitAerBackend name={self.name!r} noise={noise}>"
