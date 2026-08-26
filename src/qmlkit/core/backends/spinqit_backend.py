"""SpinQit backend.

Verified against a live SpinQit install (Python 3.10). The conventions that matter,
all confirmed empirically rather than assumed:

* **Bit order matches qmlkit.** ``X`` on qubit 0 of a two-qubit register yields
  ``{'10': shots}`` — qubit 0 is the most significant bit. No remapping needed,
  unlike Qiskit.
* **``result.states`` is an exact complex128 statevector**, so SpinQit supports an
  exact mode; it is not sampling-only.
* **``result.probabilities`` is a dict** keyed by bitstring, not an array.
* **``ControlledGate(Rz)``** reproduces qmlkit's ``crz`` exactly, and ``P``, ``Sd``
  and ``Td`` match ``phase``, ``sdg`` and ``tdg``.
* **Shots are configured with ``config.configure_shots(n)``.** There is no
  ``with_shots``; ``Circuit.measure`` takes ``(qubits, clbits)``, not one argument.

By default sampling goes through qmlkit's shared seeded sampler so results are
reproducible and comparable with the other backends. Pass ``native_sampling=True``
to let SpinQit draw the shots itself, which is what you want when the point is to
model the device rather than to compare backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.ir import CircuitSpec, ParamRef

__all__ = ["SpinQitBackend"]


def _gate_table() -> dict[str, Any]:
    """qmlkit gate name -> SpinQit gate object (imported lazily)."""
    import spinqit as sq

    return {
        "i": sq.I,
        "x": sq.X,
        "y": sq.Y,
        "z": sq.Z,
        "h": sq.H,
        "s": sq.S,
        "sdg": sq.Sd,
        "t": sq.T,
        "tdg": sq.Td,
        "rx": sq.Rx,
        "ry": sq.Ry,
        "rz": sq.Rz,
        "phase": sq.P,
        "cx": sq.CX,
        "cz": sq.CZ,
        "swap": sq.SWAP,
        "crx": sq.ControlledGate(sq.Rx),
        "cry": sq.ControlledGate(sq.Ry),
        "crz": sq.ControlledGate(sq.Rz),
    }


#: Gates whose native SpinQit definition disagrees with qmlkit's, emitted as an
#: equivalent decomposition instead.
#:
#: ``CY``: SpinQit's controlled-Y applies ``[[0, -1], [1, 0]]`` (= -iY) to the
#: control-1 subspace instead of ``Y = [[0, -i], [i, 0]]``. That is a *relative*
#: phase between the control branches, so it is physically observable, not a
#: harmless global phase - a Bell-type superposition of the control qubit gives
#: different measurement statistics. SpinQit's single-qubit ``Y`` is correct;
#: only the controlled form is affected. ``Sd . CX . S`` on the target reproduces
#: the standard gate exactly, verified against the NumPy reference.
_DECOMPOSITIONS: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    "cy": (("sdg", (1,)), ("cx", (0, 1)), ("s", (1,))),
}


class SpinQitBackend(Backend):
    """Runs qmlkit circuits on SpinQit's simulator."""

    name = "spinqit"
    supports_statevector = True
    supports_exact = True

    def __init__(
        self,
        seed: int | None = None,
        compiler: str = "native",
        native_sampling: bool = False,
    ) -> None:
        super().__init__(seed)
        try:
            import spinqit  # noqa: F401
            from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "SpinQit is not installed or not importable in this interpreter.\n"
                "It ships wheels for Python 3.8-3.10 only and pins numpy<2:\n"
                "    pip install 'qmlkit[spinqit]'   # on a 3.8-3.10 interpreter\n"
                "The 'numpy' backend is exact and works on every supported Python."
            ) from exc
        self._compiler_name = compiler
        self._compiler = get_compiler(compiler)
        self._engine = get_basic_simulator()
        self._config_cls = BasicSimulatorConfig
        self._gates = _gate_table()
        self.native_sampling = native_sampling

    # ---------------------------------------------------------------- build --
    def to_spinqit(self, spec: CircuitSpec) -> Any:
        """Translate a bound :class:`CircuitSpec` into a SpinQit ``Circuit``.

        Public because an idiomatic SpinQit circuit is useful on its own — draw it,
        or hand it to SpinQit's own algorithms.
        """
        from spinqit import Circuit

        self._check_bound(spec)
        circ = Circuit()
        q = circ.allocateQubits(spec.n_qubits)

        for op in spec.ops:
            if op.gate in _DECOMPOSITIONS:
                for sub_gate, sub_pos in _DECOMPOSITIONS[op.gate]:
                    sub_qubits = tuple(op.qubits[i] for i in sub_pos)
                    self._emit(circ, q, sub_gate, sub_qubits, ())
                continue
            angles = tuple(self._angle(p, op.gate) for p in op.params)
            self._emit(circ, q, op.gate, op.qubits, angles)
        return circ

    def _emit(
        self,
        circ: Any,
        q: Any,
        gate_name: str,
        qubits: tuple[int, ...],
        angles: tuple[float, ...],
    ) -> None:
        try:
            gate = self._gates[gate_name]
        except KeyError:
            raise NotImplementedError(
                f"gate {gate_name!r} has no SpinQit mapping; add it to _gate_table()"
            ) from None
        wires = q[qubits[0]] if len(qubits) == 1 else tuple(q[i] for i in qubits)
        if angles:
            circ << (gate, wires, *angles)
        else:
            circ << (gate, wires)

    @staticmethod
    def _angle(p: Any, gate: str) -> float:
        if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
            raise ValueError(f"unbound parameter reached the backend in {gate!r}")
        return float(p)

    # ------------------------------------------------------------------ run --
    def _execute(self, spec: CircuitSpec, shots: int | None = None) -> Any:
        exe = self._compiler.compile(self.to_spinqit(spec), 0)
        config = self._config_cls()
        if shots is not None:
            config.configure_shots(shots)
        return self._engine.execute(exe, config)

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        return np.asarray(self._execute(spec).states, dtype=complex).ravel()

    def counts(self, spec: CircuitSpec, shots: int, seed: int | None = None) -> dict[str, int]:
        """Sample the computational basis.

        Uses qmlkit's shared seeded sampler by default so runs are reproducible and
        directly comparable with the other backends. ``native_sampling=True`` hands
        the job to SpinQit instead, which is unseeded but models the device.
        """
        if not self.native_sampling:
            return super().counts(spec, shots, seed)
        self._check_bound(spec)
        if shots <= 0:
            raise ValueError("shots must be positive")
        raw = self._execute(spec, shots).counts
        width = spec.n_qubits
        return {str(k).zfill(width): int(v) for k, v in raw.items()}

    # -------------------------------------------------------------- self-test --
    def verify_conventions(self) -> dict[str, bool]:
        """Check this install against the conventions the backend assumes.

        Cheap insurance against a SpinQit release quietly changing bit order or a
        gate definition. Returns a name -> passed mapping; all values should be True.
        """
        from qmlkit.core.builder import QCircuit

        results: dict[str, bool] = {}

        qc = QCircuit(2)
        qc.x(0)
        probs = np.abs(self.statevector(qc.to_spec())) ** 2
        results["qubit0_is_most_significant"] = bool(np.argmax(probs) == 0b10)

        qc = QCircuit(1)
        qc.ry(0, 0.7)
        psi = self.statevector(qc.to_spec())
        results["ry_is_exp_minus_i_theta_y_over_2"] = bool(
            np.allclose(psi, [np.cos(0.35), np.sin(0.35)], atol=1e-9)
        )

        qc = QCircuit(2)
        qc.x(0).crz(0, 1, 0.7)
        psi = self.statevector(qc.to_spec())
        results["controlled_rz_matches"] = bool(
            np.isclose(psi[0b10], np.exp(-1j * 0.35), atol=1e-9)
        )

        qc = QCircuit(1)
        qc.x(0).phase(0, 0.7)
        psi = self.statevector(qc.to_spec())
        results["phase_is_diag_1_expi_theta"] = bool(
            np.isclose(psi[1], np.exp(1j * 0.7), atol=1e-9)
        )
        qc = QCircuit(2)
        qc.h(0).cy(0, 1)
        psi = self.statevector(qc.to_spec())
        expected = np.array([1, 0, 0, 1j]) / np.sqrt(2)
        results["controlled_y_matches"] = bool(np.allclose(psi, expected, atol=1e-8))

        return results

    def __repr__(self) -> str:
        return (
            f"<SpinQitBackend compiler={self._compiler_name!r} "
            f"native_sampling={self.native_sampling}>"
        )
