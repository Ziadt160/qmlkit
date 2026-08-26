"""Qiskit backend.

**Endianness.** Qiskit is little-endian: its statevector index puts qubit 0 in the
*least* significant bit, while qmlkit puts it in the most significant. Rather than
reversing vectors after the fact — easy to get right once and wrong forever after —
we map qmlkit qubit ``i`` onto Qiskit qubit ``n-1-i`` when building the circuit.
The two index conventions then coincide exactly, and Qiskit's own bitstring keys
come out in qmlkit order with no post-processing.

Every gate convention here is verified against the NumPy backend by
``tests/test_cross_backend.py`` rather than assumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.ir import CircuitSpec, ParamRef

if TYPE_CHECKING:  # pragma: no cover
    from qiskit import QuantumCircuit

#: qmlkit gate name -> QuantumCircuit method name
_GATE_METHODS = {
    "i": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "phase": "p",
    "cx": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
}


class QiskitBackend(Backend):
    """Runs qmlkit circuits on Qiskit's reference statevector simulator."""

    name = "qiskit"
    supports_statevector = True
    supports_exact = True

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        try:
            import qiskit  # noqa: F401
            from qiskit.quantum_info import Statevector
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "Qiskit is not installed. Install it with:\n    pip install 'qmlkit[qiskit]'"
            ) from exc
        self._Statevector = Statevector

    # ---------------------------------------------------------------- build --
    def to_qiskit(self, spec: CircuitSpec) -> QuantumCircuit:
        """Translate a bound :class:`CircuitSpec` into a Qiskit circuit.

        Public because an idiomatic Qiskit circuit is useful in its own right —
        draw it, transpile it, or send it to a provider.
        """
        from qiskit import QuantumCircuit

        self._check_bound(spec)
        n = spec.n_qubits
        qc = QuantumCircuit(n)

        for op in spec.ops:
            try:
                method = _GATE_METHODS[op.gate]
            except KeyError:
                raise NotImplementedError(
                    f"gate {op.gate!r} has no Qiskit mapping; add it to _GATE_METHODS"
                ) from None
            angles = [self._angle(p, op.gate) for p in op.params]
            wires = [n - 1 - q for q in op.qubits]  # qmlkit qubit -> qiskit qubit
            getattr(qc, method)(*angles, *wires)
        return qc

    @staticmethod
    def _angle(p: Any, gate: str) -> float:
        if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
            raise ValueError(f"unbound parameter reached the backend in {gate!r}")
        return float(p)

    # ----------------------------------------------------------------- run --
    def statevector(self, spec: CircuitSpec) -> np.ndarray:
        data = self._Statevector.from_instruction(self.to_qiskit(spec)).data
        return np.asarray(data, dtype=complex)
