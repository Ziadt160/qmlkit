"""The fluent builder, entanglement patterns, and shot arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.utils.shots import runtime_estimate


# ---------------------------------------------------------- entangler patterns
@pytest.mark.parametrize(
    ("pattern", "n", "expected"),
    [
        ("chain", 4, ((0, 1), (1, 2), (2, 3))),
        ("linear", 3, ((0, 1), (1, 2))),
        ("ring", 3, ((0, 1), (1, 2), (2, 0))),
        ("ring", 2, ((0, 1),)),  # a 2-qubit ring is one pair, not two
        ("full", 3, ((0, 1), (0, 2), (1, 2))),
        ("alternating", 4, ((0, 1), (2, 3), (1, 2))),
    ],
)
def test_entangler_patterns(pattern, n, expected):
    assert qk.entangler_pairs(n, pattern) == expected


def test_single_qubit_has_no_entanglers():
    assert qk.entangler_pairs(1, "ring") == ()


def test_unknown_pattern_is_refused():
    with pytest.raises(ValueError, match="unknown entanglement pattern"):
        qk.entangler_pairs(3, "spiral")


# ------------------------------------------------------------------- builder
def test_every_single_qubit_gate_builds_and_runs():
    qc = qk.QCircuit(1)
    qc.h(0).x(0).y(0).z(0).s(0).t(0).rx(0, 0.1).ry(0, 0.2).rz(0, 0.3).phase(0, 0.4)
    psi = qk.statevector(qc.to_spec())
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_every_two_qubit_gate_builds_and_runs():
    qc = qk.QCircuit(2)
    qc.h(0).cx(0, 1).cy(0, 1).cz(0, 1).swap(0, 1)
    qc.crx(0, 1, 0.1).cry(0, 1, 0.2).crz(0, 1, 0.3)
    psi = qk.statevector(qc.to_spec())
    assert np.isclose(np.linalg.norm(psi), 1.0)


def test_params_allocates_a_block():
    qc = qk.QCircuit(2)
    refs = qc.params(3)
    assert [r.index for r in refs] == [0, 1, 2]
    assert qc.n_params == 3


def test_builder_rejects_a_non_positive_width():
    with pytest.raises(ValueError, match="n_qubits must be positive"):
        qk.QCircuit(0)


def test_entangle_refuses_a_parametric_gate():
    qc = qk.QCircuit(2)
    with pytest.raises(ValueError, match="parameterised"):
        qc.entangle(gate="crz")


def test_parametric_entangle_allocates_one_angle_per_pair():
    qc = qk.QCircuit(3)
    qc.parametric_entangle("crz", "ring")
    spec = qc.to_spec()
    assert spec.n_params == 3
    assert qk.grad_circuit_cost(spec) == 3 * 4  # four-term rule per CRZ


def test_rotation_layer_on_a_subset_of_wires():
    qc = qk.QCircuit(4)
    qc.rotation_layer(("ry",), wires=[0, 2])
    spec = qc.to_spec()
    assert spec.n_params == 2
    assert {op.qubits[0] for op in spec.ops} == {0, 2}


def test_repr_reports_shape():
    qc = qk.QCircuit(2)
    qc.rotation_layer(("ry",))
    assert "n_qubits=2" in repr(qc)
    assert len(qc) == 2
    assert "n_qubits=2" in repr(qc.to_spec())


def test_spec_iteration_and_length():
    qc = qk.QCircuit(2)
    qc.h(0).cx(0, 1)
    spec = qc.to_spec()
    assert len(spec) == 2
    assert [op.gate for op in spec] == ["h", "cx"]


# --------------------------------------------------------------------- gates
def test_gate_registry_rejects_a_duplicate_name():
    with pytest.raises(ValueError, match="already registered"):
        qk.register_gate(qk.GateDef("ry", 1, 1, lambda t: np.eye(2, dtype=complex)))


def test_unknown_gate_names_the_alternatives():
    with pytest.raises(KeyError, match="unknown gate"):
        qk.get_gate("frobnicate")


def test_gate_aliases_resolve():
    assert qk.get_gate("cnot").name == "cx"
    assert "cx" in qk.list_gates()


def test_wrong_qubit_count_is_rejected():
    with pytest.raises(ValueError, match="acts on 2 qubit"):
        qk.Op("cx", (0,))


def test_repeated_qubit_in_a_two_qubit_gate_is_rejected():
    with pytest.raises(ValueError, match="repeated qubit"):
        qk.Op("cx", (1, 1))


# ------------------------------------------------------------ shot arithmetic
def test_variance_and_standard_error():
    assert qk.variance(0.0) == pytest.approx(1.0)
    assert qk.variance(1.0) == pytest.approx(0.0)
    assert qk.standard_error(0.0, 100) == pytest.approx(0.1)


def test_shot_helpers_reject_nonsense():
    with pytest.raises(ValueError, match="shots must be positive"):
        qk.standard_error(0.0, 0)
    with pytest.raises(ValueError, match="eps must be positive"):
        qk.shots_for_precision(0.0)
    with pytest.raises(ValueError, match="rate_hz must be positive"):
        runtime_estimate(100, 0.0)


def test_runtime_estimate():
    assert runtime_estimate(10_000, 1_000.0) == pytest.approx(10.0)


def test_counts_require_positive_shots():
    with pytest.raises(ValueError, match="shots must be positive"):
        qk.run_counts(qk.angle_encode([0.3]), shots=0)


# ---------------------------------------------------------------- execution
def test_expectation_batch_over_several_parameter_vectors():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    spec = qc.to_spec()
    thetas = [[0.0], [np.pi / 2], [np.pi]]
    got = qk.expectation_batch([spec] * 3, qk.Z(0), thetas)
    assert got == pytest.approx([1.0, 0.0, -1.0], abs=1e-12)


def test_expectation_batch_rejects_a_length_mismatch():
    spec = qk.angle_encode([0.3])
    with pytest.raises(ValueError, match="but 2 parameter vectors"):
        qk.expectation_batch([spec], qk.Z(0), [[0.1], [0.2]])


def test_exact_expectation_reports_zero_uncertainty():
    value, err = qk.expectation(qk.angle_encode([0.7]), qk.Z(0), return_std=True)
    assert err == 0.0
    assert value == pytest.approx(np.cos(0.7))


def test_default_observable_is_z0():
    spec = qk.angle_encode([0.7])
    assert qk.expectation(spec) == pytest.approx(qk.expectation(spec, qk.Z(0)))


def test_set_default_backend_round_trip():
    original = qk.default_backend()
    try:
        qk.set_default_backend("numpy", seed=1)
        assert qk.expectation(qk.angle_encode([0.4]), qk.Z(0)) == pytest.approx(np.cos(0.4))
    finally:
        qk.set_default_backend(original)


def test_backend_instance_passes_through():
    be = qk.NumpyBackend(seed=2)
    assert qk.get_backend(be) is be


def test_too_many_qubits_is_refused_with_a_clear_message():
    be = qk.NumpyBackend(max_qubits=3)
    qc = qk.QCircuit(4)
    qc.h(0)
    with pytest.raises(ValueError, match="max_qubits"):
        be.statevector(qc.to_spec())
