"""Cirq's density-matrix simulator, with a noise model.

The circuit translation is :class:`~qmlkit.core.backends.cirq_backend.CirqBackend`'s,
unchanged and inherited - a second copy of the gate table is exactly the kind of
divergence the cross-backend suite exists to catch. Only the simulator differs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import BackendNotAvailable
from qmlkit.core.backends.cirq_backend import CirqBackend
from qmlkit.core.backends.noisy import NoisyBackend
from qmlkit.core.ir import CircuitSpec

if TYPE_CHECKING:  # pragma: no cover
    import cirq


class CirqDensityBackend(NoisyBackend, CirqBackend):
    """Runs qmlkit circuits on ``cirq.DensityMatrixSimulator``.

    ``noise`` is anything Cirq accepts as a ``NOISE_MODEL_LIKE``: a single-qubit
    channel (applied after every moment, on every qubit), a full ``cirq.NoiseModel``,
    or ``None``.

    ``None`` is not a pointless case - it is an exact mixed-state simulator that must
    agree with the pure-state backends to machine precision, which is what makes the
    noisy results trustworthy. ``tests/test_noisy_backends.py`` asserts exactly that.

    >>> import cirq, qmlkit as qk                                   # doctest: +SKIP
    >>> be = qk.get_backend("cirq-density", noise=cirq.depolarize(0.01))  # doctest: +SKIP
    >>> qk.expectation(spec, qk.Z(0), backend=be)                   # doctest: +SKIP
    """

    name = "cirq-density"

    #: a DensityMatrixSimulator, not the base class's state-vector one. Cirq brands
    #: its result types by simulator, so this is declared rather than inferred.
    _simulator: Any

    def __init__(self, seed: int | None = None, noise: Any = None) -> None:
        super().__init__(seed)
        try:
            self._simulator = self._cirq.DensityMatrixSimulator(dtype=np.complex128, noise=noise)
        except (TypeError, ValueError) as exc:
            raise BackendNotAvailable(
                f"cirq could not build a noise model from {noise!r}.\n"
                "Pass a cirq channel (cirq.depolarize(0.01), cirq.amplitude_damp(0.05)), "
                "a cirq.NoiseModel, or None for an exact mixed-state simulation."
            ) from exc
        self.noise = noise

    def density_matrix(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        circuit = self.to_cirq(spec)
        # the explicit qubit_order is load-bearing here for the same reason it is on
        # the pure-state backend: Cirq drops qubits a circuit never touches
        result = self._simulator.simulate(circuit, qubit_order=self.qubits(spec.n_qubits))
        return np.asarray(result.final_density_matrix, dtype=complex)

    def to_cirq(self, spec: CircuitSpec) -> cirq.Circuit:
        """The same translation the pure-state backend uses; noise is applied by the
        simulator, not written into the circuit."""
        return super().to_cirq(spec)

    def __repr__(self) -> str:
        return f"<CirqDensityBackend name={self.name!r} noise={self.noise!r}>"
