"""Core tests: IR, backend, observables, execution."""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk


# ------------------------------------------------------------------ acceptance
def test_phase0_acceptance_criterion():
    """The Phase 0 goal: expectation(angle_encode([0.7]), Z(0)) == cos(0.7)."""
    assert qk.expectation(qk.angle_encode([0.7]), qk.Z(0)) == pytest.approx(np.cos(0.7), abs=1e-12)


# ------------------------------------------------------------------------- IR
def test_depth_and_resources():
    qc = qk.QCircuit(3)
    qc.rotation_layer(("ry",)).entangle("chain")
    spec = qc.to_spec()
    res = spec.resources()
    assert res["n_qubits"] == 3
    assert res["n_params"] == 3
    assert res["n_2q"] == 2
    assert res["n_1q"] == 3
    assert spec.depth() == 3  # ry, then cx(0,1), then cx(1,2)


def test_adjoint_inverts_the_circuit():
    qc = qk.QCircuit(2)
    qc.ry(0, 0.6).cx(0, 1).rz(1, -1.2).s(0)
    spec = qc.to_spec()
    combined = spec.compose(spec.adjoint())
    psi = qk.statevector(combined)
    expected = np.zeros(4, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(psi, expected, atol=1e-12)


def test_compose_offsets_parameter_indices():
    a = qk.QCircuit(1)
    a.ry(0, qk.ParamRef(0))
    b = qk.QCircuit(1)
    b.rz(0, qk.ParamRef(0))
    joined = a.to_spec().compose(b.to_spec())
    assert joined.n_params == 2
    assert [s.ref.index for s in joined.slots()] == [0, 1]


def test_unbound_circuit_refuses_to_run():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    with pytest.raises(ValueError, match="free parameters"):
        qk.get_backend("numpy").statevector(qc.to_spec())


def test_wrong_parameter_count_is_rejected():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    with pytest.raises(ValueError, match="expected 1 parameters"):
        qk.expectation(qc.to_spec(), qk.Z(0), theta=[0.1, 0.2])


def test_out_of_range_qubit_is_rejected():
    with pytest.raises(ValueError, match="out of range"):
        qk.CircuitSpec(2, (qk.Op("x", (5,)),))


# ------------------------------------------------------------------ encoding
def test_basis_encode_puts_all_probability_on_one_string():
    spec = qk.basis_encode([1, 0, 1])
    counts = qk.run_counts(spec, shots=512, seed=0)
    assert counts == {"101": 512}
    assert qk.basis_index([1, 0, 1]) == 5


def test_qubit_zero_is_the_most_significant_bit():
    spec = qk.basis_encode([1, 0])
    assert list(qk.run_counts(spec, shots=64, seed=1)) == ["10"]


def test_angle_encoding_gives_one_cosine_per_feature():
    xs = [0.3, 1.4, 2.9]
    spec = qk.angle_encode(xs)
    for i, x in enumerate(xs):
        assert qk.expectation(spec, qk.Z(i)) == pytest.approx(np.cos(x), abs=1e-12)


def test_to_angle_range_handles_a_constant_column():
    x = np.array([[1.0, 5.0], [1.0, 7.0]])
    out = qk.to_angle_range(x, 0.0, 2 * np.pi)
    assert out[0, 0] == pytest.approx(np.pi)
    assert out[1, 0] == pytest.approx(np.pi)
    assert out[0, 1] == pytest.approx(0.0)
    assert out[1, 1] == pytest.approx(2 * np.pi)


def test_n_qubits_for():
    assert qk.n_qubits_for(4) == 2
    assert qk.n_qubits_for(5) == 3
    assert qk.n_qubits_for(1) == 1


# --------------------------------------------------------------- observables
def test_multi_qubit_expectation_is_correct():
    """The bug the per-lecture expz() had: dividing by n0 + n1 past one qubit."""
    qc = qk.QCircuit(2)
    qc.h(0).cx(0, 1)  # Bell state
    spec = qc.to_spec()
    assert qk.expectation(spec, qk.Z(0)) == pytest.approx(0.0, abs=1e-12)
    assert qk.expectation(spec, qk.Z(1)) == pytest.approx(0.0, abs=1e-12)
    assert qk.expectation(spec, qk.ZZ(0, 1)) == pytest.approx(1.0, abs=1e-12)


def test_x_and_y_expectations_use_a_basis_rotation():
    qc = qk.QCircuit(1)
    qc.ry(0, np.pi / 2)  # |+>
    spec = qc.to_spec()
    assert qk.expectation(spec, qk.X(0)) == pytest.approx(1.0, abs=1e-12)
    assert qk.expectation(spec, qk.Z(0)) == pytest.approx(0.0, abs=1e-12)

    qc2 = qk.QCircuit(1)
    qc2.rx(0, -np.pi / 2)  # +1 eigenstate of Y
    assert qk.expectation(qc2.to_spec(), qk.Y(0)) == pytest.approx(1.0, abs=1e-12)


def test_pauli_sum_is_linear():
    spec = qk.angle_encode([0.5, 1.0])
    obs = 2.0 * qk.Z(0) + qk.Z(1)
    expected = 2 * np.cos(0.5) + np.cos(1.0)
    assert qk.expectation(spec, obs) == pytest.approx(expected, abs=1e-12)


def test_sampled_expectation_converges_to_the_exact_one():
    spec = qk.angle_encode([1.0])
    exact = qk.expectation(spec, qk.Z(0))
    sampled, err = qk.expectation(spec, qk.Z(0), shots=200_000, seed=7, return_std=True)
    assert abs(sampled - exact) < 5 * err
    assert err == pytest.approx(qk.standard_error(sampled, 200_000))


def test_shot_noise_shrinks_as_one_over_sqrt_n():
    spec = qk.angle_encode([np.arccos(0.5)])
    exact = 0.5
    errors = []
    for shots in (100, 10_000):
        vals = [qk.expectation(spec, qk.Z(0), shots=shots, seed=s) for s in range(40)]
        errors.append(np.sqrt(np.mean((np.array(vals) - exact) ** 2)))
    # 100x more shots => ~10x smaller error
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.5)


# --------------------------------------------------------------------- shots
def test_shots_for_precision_inverts_standard_error():
    n = qk.shots_for_precision(0.01, z=0.0)
    assert qk.standard_error(0.0, n) == pytest.approx(0.01, rel=1e-3)


def test_p0_and_z_round_trip():
    for z in (-1.0, -0.3, 0.0, 0.42, 1.0):
        assert qk.z_from_p0(qk.p0_from_z(z)) == pytest.approx(z)


# ------------------------------------------------------------------ backends
def test_backend_registry():
    assert "numpy" in qk.list_backends()
    assert isinstance(qk.get_backend("numpy"), qk.NumpyBackend)
    with pytest.raises(KeyError, match="unknown backend"):
        qk.get_backend("nope")


def test_missing_backend_explains_itself():
    """A missing SDK must produce an install hint, not an ImportError traceback."""
    missing = [b for b in qk.list_backends() if not qk.is_available(b)]
    if not missing:  # pragma: no cover - every SDK happens to be installed
        pytest.skip("all backends available in this interpreter")
    name = missing[0]
    with pytest.raises(qk.BackendNotAvailable) as exc:
        qk.get_backend(name)
    message = str(exc.value)
    assert "pip install" in message
    assert "Available now:" in message


def test_availability_is_reported_without_importing_the_sdk():
    assert "numpy" in qk.available_backends()
    assert qk.is_available("numpy")
    assert not qk.is_available("no-such-backend")
    report = qk.backend_report()
    assert "numpy" in report
    for name in qk.list_backends():
        assert name in report


def test_default_backend_can_be_set_from_the_environment(monkeypatch):
    """QMLKIT_BACKEND lets an existing script target another SDK unchanged."""
    import qmlkit.core.backends.registry as reg

    monkeypatch.setattr(reg, "_default", None)
    monkeypatch.setenv("QMLKIT_BACKEND", "numpy")
    assert reg.default_backend().name == "numpy"
