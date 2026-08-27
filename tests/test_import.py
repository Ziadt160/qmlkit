"""Reading circuits in, checked where importers actually break: qubit order.

A gate-name table is easy and its mistakes are loud. Endianness is neither — an
importer that maps qubits the wrong way round produces a circuit that runs, returns
numbers in the right range, and is wrong. So almost everything here is asserted
against a **statevector**, not against an op list: the round trip
``from_qiskit(to_qiskit(spec))`` has to reproduce the amplitudes, and the PennyLane
importer has to agree with PennyLane's own simulator.

The circuit zoo is deliberately asymmetric — controlled gates whose control and
target differ, gates on the first and last wire, and idle qubits — because a
symmetric circuit cannot tell a correct mapping from a reversed one.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.interop import UnsupportedGate, from_qasm


# --------------------------------------------------------------------------- #
# a zoo that can tell a reversed register from a correct one
# --------------------------------------------------------------------------- #
def _asymmetric(n_qubits: int = 3) -> qk.CircuitSpec:
    qc = qk.QCircuit(n_qubits)
    qc.h(0)
    qc.ry(1, 0.7)
    qc.cx(0, 2)          # control and target on different ends
    qc.crz(2, 1, 1.1)    # control > target, so a flip would be visible
    qc.rz(0, -0.4)
    qc.swap(0, 1)
    return qc.to_spec()


def _state(spec: qk.CircuitSpec) -> np.ndarray:
    return np.asarray(qk.statevector(spec))


QASM_BELL = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0],q[1];
"""


# --------------------------------------------------------------------------- #
# QASM: the stdlib-only path
# --------------------------------------------------------------------------- #
def test_qasm_needs_no_optional_dependency():
    """The widest import path must work in a bare `pip install qmlkit`."""
    spec = from_qasm(QASM_BELL)
    assert spec.n_qubits == 2
    amplitudes = _state(spec)
    assert abs(amplitudes[0]) == pytest.approx(1 / np.sqrt(2))
    assert abs(amplitudes[3]) == pytest.approx(1 / np.sqrt(2))
    assert np.allclose(amplitudes[[1, 2]], 0)


def test_qasm_matches_qiskit_on_the_same_source():
    """The parser is only right if it agrees with the tool that wrote the file."""
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
h q[0];
ry(0.7) q[1];
cx q[0],q[2];
crz(-1.1) q[2],q[1];
rz(0.3) q[0];
"""
    ours = _state(from_qasm(qasm))
    theirs = np.asarray(Statevector(qiskit.QuantumCircuit.from_qasm_str(qasm)))
    # qmlkit is big-endian, Qiskit little-endian: the importer already flipped the
    # wires, so comparing amplitudes means undoing that on Qiskit's index instead
    n = 3
    reordered = np.array(
        [theirs[int(format(i, f"0{n}b")[::-1], 2)] for i in range(2**n)]
    )
    assert np.allclose(ours, reordered, atol=1e-12)


def test_qasm_angle_expressions_are_evaluated():
    spec = from_qasm("qreg q[1];\nrx(pi/2) q[0];\nrz(-2*pi/3) q[0];\nry(0.25) q[0];")
    angles = [op.params[0] for op in spec.ops]
    assert angles == pytest.approx([np.pi / 2, -2 * np.pi / 3, 0.25])


def test_qasm_angle_expressions_cannot_execute_code():
    """`eval` on a file's contents would be a straightforward code-execution hole."""
    with pytest.raises(ValueError, match="not plain arithmetic|only "):
        from_qasm('qreg q[1];\nrx(__import__("os").getcwd()) q[0];')
    with pytest.raises(ValueError, match="only "):
        from_qasm("qreg q[1];\nrx(theta) q[0];")


def test_qasm_comments_headers_and_barriers_are_ignored():
    spec = from_qasm(
        "// leading comment\n"
        "OPENQASM 2.0;\n"
        'include "qelib1.inc";\n'
        "/* a block\n   comment */\n"
        "qreg q[2];\n"
        "creg c[2];\n"
        "x q[0];  // trailing\n"
        "barrier q;\n"
    )
    assert len(spec.ops) == 1


def test_qasm_big_endian_flag_passes_indices_through():
    little = from_qasm("qreg q[3];\nx q[0];")
    big = from_qasm("qreg q[3];\nx q[0];", little_endian=False)
    assert little.ops[0].qubits == (2,)  # qubit 0 little-endian is qmlkit's last wire
    assert big.ops[0].qubits == (0,)


def test_qasm_refuses_what_it_cannot_represent():
    with pytest.raises(UnsupportedGate, match="mid-circuit measurement"):
        from_qasm("qreg q[1];\ncreg c[1];\nmeasure q[0] -> c[0];")
    with pytest.raises(UnsupportedGate, match="custom gate"):
        from_qasm("qreg q[1];\ngate mygate a { x a; }\n")
    with pytest.raises(UnsupportedGate, match="no mapping"):
        from_qasm("qreg q[2];\nccx q[0],q[1],q[0];")


def test_qasm_missing_or_multiple_registers_are_named_errors():
    with pytest.raises(ValueError, match="no quantum register"):
        from_qasm('OPENQASM 2.0;\ninclude "qelib1.inc";\n')
    with pytest.raises(ValueError, match="2 quantum registers"):
        from_qasm("qreg a[1];\nqreg b[1];\n")


def test_qasm_out_of_range_qubit_is_caught():
    with pytest.raises(ValueError, match="outside the 2-qubit register"):
        from_qasm("qreg q[2];\nx q[5];")


def test_u3_decomposes_and_warns_about_the_dropped_phase():
    with pytest.warns(UserWarning, match="overall phase"):
        spec = from_qasm("qreg q[1];\nu3(0.3,0.4,0.5) q[0];")
    assert [op.gate for op in spec.ops] == ["rz", "ry", "rz"]


def test_u3_matches_qiskit_up_to_a_global_phase():
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    qasm = "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[1];\nu3(0.3,0.4,0.5) q[0];\n"
    with pytest.warns(UserWarning):
        ours = _state(from_qasm(qasm))
    theirs = np.asarray(Statevector(qiskit.QuantumCircuit.from_qasm_str(qasm)))
    overlap = abs(np.vdot(ours, theirs))
    assert overlap == pytest.approx(1.0, abs=1e-12)  # equal up to a global phase
    assert not np.allclose(ours, theirs)  # and the phase really is dropped


# --------------------------------------------------------------------------- #
# Qiskit: the round trip that pins endianness
# --------------------------------------------------------------------------- #
@pytest.mark.qiskit
def test_qiskit_round_trip_reproduces_the_statevector():
    pytest.importorskip("qiskit")
    spec = _asymmetric()
    restored = qk.from_qiskit(qk.get_backend("qiskit").to_qiskit(spec))
    np.testing.assert_allclose(_state(restored), _state(spec), atol=1e-12)


@pytest.mark.qiskit
@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4])
def test_qiskit_round_trip_over_random_circuits(n_qubits):
    """Randomised, because a hand-picked circuit confirms what the author believed."""
    pytest.importorskip("qiskit")
    rng = np.random.default_rng(n_qubits)
    for _ in range(6):
        qc = qk.QCircuit(n_qubits)
        for _ in range(10):
            gate = rng.choice(["h", "x", "ry", "rz", "cx", "cz", "crz", "swap"])
            if gate in ("h", "x"):
                getattr(qc, gate)(int(rng.integers(n_qubits)))
            elif gate in ("ry", "rz"):
                getattr(qc, gate)(int(rng.integers(n_qubits)), float(rng.uniform(-np.pi, np.pi)))
            elif n_qubits > 1:
                a, b = rng.choice(n_qubits, size=2, replace=False)
                if gate == "crz":
                    qc.crz(int(a), int(b), float(rng.uniform(-np.pi, np.pi)))
                else:
                    getattr(qc, gate)(int(a), int(b))
        spec = qc.to_spec()
        restored = qk.from_qiskit(qk.get_backend("qiskit").to_qiskit(spec))
        np.testing.assert_allclose(_state(restored), _state(spec), atol=1e-12)


@pytest.mark.qiskit
def test_qiskit_unbound_parameters_become_paramrefs_in_qiskits_own_order():
    """The thing QASM cannot carry, and the reason from_qiskit exists next to it."""
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    alpha, beta = Parameter("alpha"), Parameter("beta")
    qc = QuantumCircuit(2)
    qc.ry(alpha, 0)
    qc.rz(beta, 1)
    spec = qk.from_qiskit(qc)

    assert spec.n_params == 2
    order = list(qc.parameters)  # Qiskit sorts by name
    index = {p.name: order.index(p) for p in order}
    refs = {op.gate: op.params[0].index for op in spec.ops}
    assert refs["ry"] == index["alpha"]
    assert refs["rz"] == index["beta"]

    # and the parameters actually drive the circuit
    value = qk.expectation(spec, qk.Z(1), [0.0, 0.0])   # qiskit qubit 0 -> qmlkit 1
    assert value == pytest.approx(1.0)
    assert qk.expectation(spec, qk.Z(1), [np.pi, 0.0]) == pytest.approx(-1.0)


@pytest.mark.qiskit
def test_qiskit_compound_parameter_expressions_are_refused_clearly():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.ry(2 * theta + 1, 0)
    with pytest.raises(UnsupportedGate, match="compound parameter expression"):
        qk.from_qiskit(qc)


@pytest.mark.qiskit
def test_qiskit_unknown_gate_names_what_is_supported():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.ccx(0, 1, 2)
    with pytest.raises(UnsupportedGate, match="no mapping"):
        qk.from_qiskit(qc)


# --------------------------------------------------------------------------- #
# PennyLane: the migration path, and the claim that it needs no flip
# --------------------------------------------------------------------------- #
@pytest.mark.pennylane
def test_pennylane_wire_order_matches_without_flipping():
    """Asserted against their simulator rather than taken on trust."""
    qml = pytest.importorskip("pennylane")

    def circuit():
        qml.Hadamard(wires=0)
        qml.RY(0.7, wires=1)
        qml.CNOT(wires=[0, 2])
        qml.CRZ(-1.1, wires=[2, 1])
        qml.RZ(0.3, wires=0)

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def node():
        circuit()
        return qml.state()

    ours = _state(qk.from_pennylane(circuit))
    np.testing.assert_allclose(ours, np.asarray(node()), atol=1e-12)


@pytest.mark.pennylane
def test_pennylane_qnode_is_constructed_with_its_arguments():
    qml = pytest.importorskip("pennylane")
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def node(t):
        qml.RY(t, wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.state()

    spec = qk.from_pennylane(node, 0.9)
    np.testing.assert_allclose(_state(spec), np.asarray(node(0.9)), atol=1e-12)


@pytest.mark.pennylane
def test_pennylane_templates_flatten_through_the_tape():
    """A template is many ops on the tape, which is exactly what should import."""
    qml = pytest.importorskip("pennylane")

    weights = np.reshape(np.linspace(0.1, 1.2, 6), (2, 3))

    def circuit():
        qml.BasicEntanglerLayers(weights, wires=range(3))

    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def node():
        circuit()
        return qml.state()

    spec = qk.from_pennylane(circuit)
    assert len(spec.ops) > 6
    np.testing.assert_allclose(_state(spec), np.asarray(node()), atol=1e-12)


@pytest.mark.pennylane
def test_pennylane_bad_input_says_what_it_takes():
    pytest.importorskip("pennylane")
    with pytest.raises(TypeError, match="QuantumTape, a QNode or a quantum function"):
        qk.from_pennylane(42)


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
def test_importers_are_registered_and_reachable_by_name():
    assert {"qasm", "qiskit", "pennylane"} <= set(qk.list_importers())
    assert qk.get_importer("qasm") is from_qasm


def test_registering_an_importer_makes_it_reachable():
    qk.register_importer("toy", lambda text: from_qasm(text))
    try:
        assert "toy" in qk.list_importers()
        assert qk.get_importer("toy")(QASM_BELL).n_qubits == 2
    finally:
        from qmlkit.interop import _IMPORTERS

        _IMPORTERS.pop("toy", None)


def test_unknown_importer_suggests_a_real_one():
    with pytest.raises(KeyError, match="qasm"):
        qk.get_importer("qsam")
