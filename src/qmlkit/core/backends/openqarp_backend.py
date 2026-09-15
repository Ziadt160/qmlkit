"""OpenQARP backend.

Fujitsu's Open Quantum Application Research Package: a Python framework over a
compiled C++ core, in which *blocks* describe circuits, *primitives* say what to read
off them, and *engines* say how they run. This backend takes the engine directly --
``qarpx.QarpSimulator`` -- and with it the two entry points no other backend here
offers: an expectation contracted inside the simulator without ever materialising the
statevector, and that same expectation swept over a list of parameter sets without
coming back to Python between rows.

**Endianness.** qarp indexes amplitudes LSB-first: qubit ``q`` carries bit ``q``,
which is the opposite of qmlkit. The handling is the Qiskit backend's, for the Qiskit
backend's reason -- qmlkit qubit ``i`` is mapped onto qarp qubit ``n-1-i`` at build
time, so the two index conventions coincide and nothing downstream reverses anything.
Reversing the vector afterwards would also have been the slower of the two:
``qarp.endianness.lsb_to_msb_statevector`` builds its permutation from a Python loop
over ``2**n`` formatted strings, which costs 40 ms at 16 qubits -- four times the
11 ms simulation it would be decorating.

**What is overridden, and why.** ``statevector`` is the primitive, as everywhere
else. Two derived quantities are overridden as well, and both hand this library's
observable to an SDK instead of deriving it in :mod:`qmlkit.core.backends.base` --
the same exception the MPS backend makes, and the reason both are pinned against the
NumPy reference in ``tests/test_cross_backend.py`` rather than argued for here.

:meth:`~OpenQARPBackend.expectation`
    contracted in C++ from a state that is never returned, and no ``2**n`` array to
    hold: 0.39 ms against 0.54 at ten qubits, 0.91 against 1.09 at twelve, 2.77
    against 3.45 at fourteen, and the two converge by sixteen.

:meth:`~OpenQARPBackend.expectation_over_slots`
    the reason to reach for this SDK at all. One sweep instead of a translation per
    row: 256 rows of a three-layer ``ry``/``rz`` ring at ten qubits take 37 ms here,
    against 237 ms on the NumPy reference and 666 ms on Aer. A batched
    parameter-shift gradient and every forward pass of ``QuantumLayer`` are made of
    exactly this call -- an adjoint gradient and a Gram matrix are not, and ask for
    statevectors instead, where this backend is ordinary. Against its own row-by-row
    path the sweep is 6.5x at eight qubits and 1.2x at fourteen, bit-identical at
    both.

``native_expectations=False`` takes both away and derives expectations from the
statevector like every other backend. That is not a fallback for when something
breaks; it is how the suite asks one backend the same question by two routes that
share no arithmetic, and compares the answers.

(Every number here comes out of ``scripts/probe_openqarp.py``, on one machine --
Windows, Python 3.14, OpenQARP 0.1.0. Run it on yours; the ratios travel, the
milliseconds do not. ``docs/guides/openqarp.md`` is the long form.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.ir import CircuitSpec, Op, ParamRef
from qmlkit.core.observables import Observable, as_sum

if TYPE_CHECKING:  # pragma: no cover
    import qarpx

#: qmlkit gate name -> the qarp ``Block`` method that emits it. Wires come first and
#: the angle last, which is the opposite of Qiskit's order. Every convention in this
#: table -- including the ``rz`` global phase and the controlled rotations, where SDKs
#: disagree most -- is checked against the NumPy reference rather than assumed.
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

#: One qarp symbol per qmlkit *slot*. Slot space, not logical-parameter space, is what
#: a sweep binds: a weight-tied parameter fills several slots and a shift rule moves
#: one of them at a time, so a symbol per slot is the only naming that can express
#: what a gradient actually asks for.
_SLOT_SYMBOL = "qmlkit_slot_{}"


class OpenQARPBackend(Backend):
    """Runs qmlkit circuits on OpenQARP's ``QarpSimulator``."""

    name = "openqarp"
    supports_statevector = True
    supports_exact = True

    #: Rows per call to the C++ sweep. Measured flat from 1 to 256 rows per call --
    #: 256 rows of a twelve-qubit ansatz cost 155 ms either way -- so this is a
    #: progress-reporting granularity, not a throughput knob. A Gram matrix that would
    #: otherwise sit silent for minutes reports itself every chunk.
    sweep_rows: int = 256

    def __init__(self, seed: int | None = None, native_expectations: bool = True) -> None:
        super().__init__(seed)
        try:
            import qarpx
            from qarp.blocks import CompositeBlock, SimpleBlock, SynthesizedUnitaryBlock
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "OpenQARP is not installed. Install it with:\n"
                "    pip install 'qmlkit[openqarp]'\n"
                "(the distribution is 'openqarp'; the module it installs is 'qarp')"
            ) from exc
        self._qx = qarpx
        self._SimpleBlock = SimpleBlock
        self._CompositeBlock = CompositeBlock
        self._SynthesizedUnitaryBlock = SynthesizedUnitaryBlock
        self._native = bool(native_expectations)

    # ---------------------------------------------------------------- build --
    def to_openqarp(self, spec: CircuitSpec) -> qarpx.Block:
        """Translate a bound :class:`CircuitSpec` into a qarp block."""
        self._check_bound(spec)
        return self._emit(spec, symbolic=False)

    def _emit(self, spec: CircuitSpec, *, symbolic: bool) -> qarpx.Block:
        """The one translation. ``symbolic`` leaves each slot as a qarp symbol.

        Consecutive named gates accumulate in a single ``SimpleBlock``; a registered
        gate -- which has no name here and arrives as a matrix -- closes that run and
        becomes a child of its own. Circuits with no such gate, which is nearly all of
        them, never pay for the composite.
        """
        n = spec.n_qubits
        children: list[Any] = []
        segment = self._SimpleBlock(n, name="qmlkit")
        emitted = False
        slot = 0

        for op in spec.ops:
            angles: list[Any] = []
            for p in op.params:
                if isinstance(p, ParamRef):
                    if not symbolic:  # pragma: no cover - _check_bound rules this out
                        raise ValueError(f"unbound parameter reached the backend in {op.gate!r}")
                    angles.append(self._qx.Param.symbol(_SLOT_SYMBOL.format(slot)))
                    slot += 1
                else:
                    angles.append(float(p))

            method = _GATE_METHODS.get(op.gate)
            if method is None:
                if emitted:
                    children.append(segment.build())
                    segment = self._SimpleBlock(n, name="qmlkit")
                    emitted = False
                children.append(self._synthesised(op, angles, n))
                continue
            getattr(segment, method)(*(n - 1 - q for q in op.qubits), *angles)
            emitted = True

        if not children:
            return segment.build()
        if emitted:
            children.append(segment.build())
        return self._CompositeBlock(children, n_qubits=n, name="qmlkit").build()

    def _synthesised(self, op: Op, angles: list[Any], n_qubits: int) -> qarpx.Block:
        """A gate qarp has no name for, emitted as its matrix.

        What makes ``register_gate`` mean the same thing here as on the NumPy
        reference; without it, the registry the documentation advertises as an
        extension point would stop at this backend. qarp decomposes the matrix
        exactly, global phase included, by Quantum Shannon decomposition.

        The wire order is the subtle part, and it is subtle in the same way it is on
        Qiskit. A matrix carries its qubit order in its *basis* rather than in a wire
        list: qmlkit writes the op's first wire as the most significant bit, and a
        qarp sub-block reads its own local qubit 0 as the least significant. Reversing
        the wire list is what lines the two up, and the global ``n-1-i`` map goes on
        top of that. Getting it wrong is silent -- the circuit still runs, with the
        operands swapped -- so it is asserted against the reference over several wire
        orders rather than reasoned about here.
        """
        from qmlkit.core.gates import gate_matrix

        if any(isinstance(a, self._qx.Param) for a in angles):  # pragma: no cover
            raise ValueError(
                f"the gate {op.gate!r} reaches OpenQARP as a matrix, which cannot carry "
                "a free parameter; bind the circuit first"
            )
        matrix = np.asfortranarray(np.asarray(gate_matrix(op.gate, angles), dtype=complex))
        wires = [n_qubits - 1 - q for q in reversed(op.qubits)]
        block: qarpx.Block = self._SynthesizedUnitaryBlock(
            matrix, target_qubits=wires, name=op.gate
        ).build()
        return block

    # ------------------------------------------------------------ observable --
    def _terms(self, obs: Observable, n_qubits: int) -> list[tuple[list[tuple[int, str]], complex]]:
        """One qmlkit observable in qarp's ``[(paulis, coeff), ...]`` form.

        The second place a qubit index crosses the boundary, so it carries the same
        ``n-1-i`` map the gates do. Identity factors are dropped rather than sent:
        qarp accepts them, but the base class drops them too, and two spellings of one
        term is how a difference hides.
        """
        return [
            (
                [(n_qubits - 1 - q, p.upper()) for q, p in term.paulis if p.upper() != "I"],
                complex(term.coeff),
            )
            for term in as_sum(obs).terms
        ]

    def _contracts(self, obs: Observable, shots: int | None) -> bool:
        """Whether the simulator may contract this observable itself.

        Not when shots were asked for -- sampling semantics belong to the base class,
        which is why a seed reproduces the same counts on every backend -- and not
        when any coefficient is complex. ``QarpSimulator.expectation`` returns
        ``Re <psi|H|psi>`` whatever it is handed, and a non-Hermitian observable is
        meant to *raise* here rather than come back as a plausible real number. The
        base class owns that refusal, so this defers to it.
        """
        if shots is not None or not self._native:
            return False
        return not any(term.coeff.imag for term in as_sum(obs).terms)

    @staticmethod
    def _sweepable(spec: CircuitSpec) -> bool:
        """Whether one symbolic block can stand in for every row of a sweep.

        A registered gate reaches qarp as a matrix and a matrix cannot carry a symbol,
        so a custom gate whose angle is a *parameter* has to be rebuilt row by row and
        the sweep gives way to the base class's loop.
        """
        return not any(
            op.gate not in _GATE_METHODS and any(isinstance(p, ParamRef) for p in op.params)
            for op in spec.ops
        )

    # ------------------------------------------------------------------ run --
    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        block = self.to_openqarp(spec)
        state = self._qx.QarpSimulator().statevector(block.flatten(), spec.n_qubits)
        return np.asarray(state, dtype=complex)

    def expectation(
        self,
        spec: CircuitSpec,
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> float:
        """``<O>``, contracted inside the simulator when it may be."""
        self._check_bound(spec)
        if not self._contracts(obs, shots):
            return super().expectation(spec, obs, shots, seed)
        commands = self._emit(spec, symbolic=False).flatten()
        value = self._qx.QarpSimulator().expectation(
            commands, spec.n_qubits, self._terms(obs, spec.n_qubits)
        )
        return float(value)

    def expectation_over_slots(
        self,
        spec: CircuitSpec,
        slot_angles: npt.NDArray[Any],
        obs: Observable,
        shots: int | None = None,
        seed: int | None = None,
    ) -> npt.NDArray[Any]:
        """``<O>`` at many slot-angle vectors, as one C++ sweep per chunk.

        The circuit is translated **once**, with a symbol at every slot, and the rows
        are bound inside the kernel. That is the difference between paying for the
        translation once and paying for it per row, and at the widths quantum machine
        learning actually runs at the translation is most of the bill: 7.4x at eight
        qubits, 1.7x at twelve, 1.2x at fourteen -- and the same numbers to the last
        bit, because it is the same circuit either way.
        """
        rows = np.atleast_2d(np.asarray(slot_angles, dtype=float))
        if not self._contracts(obs, shots) or not self._sweepable(spec):
            return super().expectation_over_slots(spec, rows, obs, shots, seed)

        n_slots = len(spec.slots())
        if rows.shape[1] != n_slots:
            raise ValueError(f"expected {n_slots} slot angles, got {rows.shape[1]}")

        from qmlkit.progress import task as progress_task

        commands = self._emit(spec, symbolic=True).flatten()
        terms = self._terms(obs, spec.n_qubits)
        names = [_SLOT_SYMBOL.format(k) for k in range(n_slots)]
        simulator = self._qx.QarpSimulator()

        out = np.empty(rows.shape[0], dtype=float)
        with progress_task(f"{self.name} circuits", rows.shape[0]) as tracked:
            for start in range(0, rows.shape[0], self.sweep_rows):
                chunk = rows[start : start + self.sweep_rows]
                bindings = [dict(zip(names, map(float, row), strict=True)) for row in chunk]
                out[start : start + chunk.shape[0]] = simulator.batch_expectation(
                    commands, spec.n_qubits, terms, bindings
                )
                tracked.advance(chunk.shape[0])
        return out

    def __repr__(self) -> str:
        native = "" if self._native else " native_expectations=False"
        return f"<{type(self).__name__} name={self.name!r}{native}>"
