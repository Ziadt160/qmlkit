"""Cirq backend.

**Endianness.** Cirq orders its statevector by the given ``qubit_order``, first
qubit most significant — which is already qmlkit's convention. Mapping qmlkit
qubit ``i`` to ``LineQubit(i)`` and passing an explicit ascending order needs no
index gymnastics, but the explicit order *is* required: without it Cirq drops
qubits the circuit never touches and silently returns a shorter vector.

Every gate convention here is verified against the NumPy backend by
``tests/test_cross_backend.py`` rather than assumed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.ir import CircuitSpec, ParamRef

if TYPE_CHECKING:  # pragma: no cover
    import cirq


def _gate_factories() -> dict[str, Callable[..., Any]]:
    """qmlkit gate name -> a callable returning a Cirq gate (imported lazily)."""
    import cirq

    return {
        "i": lambda: cirq.I,
        "x": lambda: cirq.X,
        "y": lambda: cirq.Y,
        "z": lambda: cirq.Z,
        "h": lambda: cirq.H,
        "s": lambda: cirq.S,
        "sdg": lambda: cirq.S**-1,
        "t": lambda: cirq.T,
        "tdg": lambda: cirq.T**-1,
        "rx": cirq.rx,
        "ry": cirq.ry,
        "rz": cirq.rz,
        # Qiskit-style phase gate: diag(1, e^{i*theta}) == Z**(theta/pi)
        "phase": lambda t: cirq.Z ** (t / np.pi),
        "cx": lambda: cirq.CNOT,
        "cy": lambda: cirq.Y.controlled(),
        "cz": lambda: cirq.CZ,
        "swap": lambda: cirq.SWAP,
        "crx": lambda t: cirq.rx(t).controlled(),
        "cry": lambda t: cirq.ry(t).controlled(),
        "crz": lambda t: cirq.rz(t).controlled(),
    }


class CirqBackend(Backend):
    """Runs qmlkit circuits on ``cirq.Simulator``."""

    name = "cirq"
    supports_statevector = True
    supports_exact = True

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        try:
            import cirq
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "Cirq is not installed. Install it with:\n    pip install 'qmlkit[cirq]'"
            ) from exc
        self._cirq = cirq
        self._factories = _gate_factories()
        # complex128 keeps the reference comparison honest; Cirq defaults to complex64
        self._simulator = cirq.Simulator(dtype=np.complex128)

    # ---------------------------------------------------------------- build --
    def to_cirq(self, spec: CircuitSpec) -> cirq.Circuit:
        """Translate a bound :class:`CircuitSpec` into a Cirq circuit."""
        self._check_bound(spec)
        cirq = self._cirq
        qubits = self.qubits(spec.n_qubits)
        moments = []
        for op in spec.ops:
            try:
                factory = self._factories[op.gate]
            except KeyError:
                raise NotImplementedError(
                    f"gate {op.gate!r} has no Cirq mapping; add it to _gate_factories()"
                ) from None
            angles = [self._angle(p, op.gate) for p in op.params]
            gate = factory(*angles)
            moments.append(gate.on(*[qubits[q] for q in op.qubits]))
        return cirq.Circuit(moments)

    def qubits(self, n: int) -> list[cirq.LineQubit]:
        return self._cirq.LineQubit.range(n)

    @staticmethod
    def _angle(p: Any, gate: str) -> float:
        if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
            raise ValueError(f"unbound parameter reached the backend in {gate!r}")
        return float(p)

    # ----------------------------------------------------------------- run --
    def statevector(self, spec: CircuitSpec) -> np.ndarray:
        circuit = self.to_cirq(spec)
        # the explicit qubit_order is load-bearing: it keeps idle qubits in the register
        result = self._simulator.simulate(circuit, qubit_order=self.qubits(spec.n_qubits))
        return np.asarray(result.final_state_vector, dtype=complex)
