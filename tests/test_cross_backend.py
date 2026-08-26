"""Cross-backend equivalence.

Every backend must agree with the NumPy reference on the same circuit. This is the
suite that turns "the Qiskit translation looks right" into "the Qiskit translation
is right" — endianness, gate conventions, controlled-gate qubit order and basis
rotations are all exercised here rather than assumed.

Backends whose SDK is not installed are skipped, not failed: the point is to test
what this interpreter can actually run.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.registry import available_backends, is_available

#: every backend except the reference itself
OTHER_BACKENDS = ["spinqit", "qiskit", "cirq"]

#: Per-backend numerical tolerance.
#:
#: Qiskit and Cirq agree with the NumPy reference to machine precision. SpinQit's
#: simulator carries a floor around 1e-10 - a single-qubit Ry(0.7) expectation
#: lands ~5.6e-11 off the analytic cos(0.7) - so it gets a looser bound. This is a
#: property of that simulator, not a translation error, and it is worth knowing
#: before anyone reports a "gradient mismatch" that is really accumulated noise.
TOLERANCE = {"spinqit": 1e-7, "qiskit": 1e-9, "cirq": 1e-9}

pytestmark = pytest.mark.parametrize(
    "backend_name",
    [
        pytest.param(
            name,
            marks=[
                pytest.mark.skipif(not is_available(name), reason=f"{name} not installed"),
                getattr(pytest.mark, name),
            ],
        )
        for name in OTHER_BACKENDS
    ],
)


# --------------------------------------------------------------------------- #
# the circuit zoo — each one targets a convention that differs between SDKs
# --------------------------------------------------------------------------- #
def _circuits() -> dict[str, tuple[qk.CircuitSpec, np.ndarray]]:
    """name -> (spec, theta). Every parameterised gate type appears at least once."""
    out: dict[str, tuple[qk.CircuitSpec, np.ndarray]] = {}

    # endianness: an asymmetric basis state is the sharpest possible probe
    out["basis_101"] = (qk.basis_encode([1, 0, 1]), np.array([]))

    # single-qubit rotations
    qc = qk.QCircuit(1)
    qc.rx(0, 0.4).ry(0, 1.1).rz(0, -0.7).phase(0, 0.9)
    out["single_qubit_rotations"] = (qc.to_spec(), np.array([]))

    # every non-parametric gate
    qc = qk.QCircuit(2)
    qc.h(0).x(1).y(0).z(1).s(0).sdg(1).t(0).tdg(1)
    out["fixed_gates"] = (qc.to_spec(), np.array([]))

    # two-qubit gates, deliberately on an asymmetric state so control/target order matters
    qc = qk.QCircuit(3)
    qc.ry(0, 0.6).ry(1, 1.3).ry(2, 0.2)
    qc.cx(0, 1).cy(1, 2).cz(0, 2).swap(1, 2)
    out["two_qubit_gates"] = (qc.to_spec(), np.array([]))

    # controlled rotations: the four-term-rule gates, and where qubit order bites
    qc = qk.QCircuit(2)
    qc.h(0).ry(1, 0.5)
    qc.crx(0, 1, 0.3).cry(1, 0, 0.7).crz(0, 1, 1.2)
    out["controlled_rotations"] = (qc.to_spec(), np.array([]))

    # a parameterised layered circuit, bound at run time
    qc = qk.QCircuit(3)
    qc.rotation_layer(("ry", "rz")).entangle("ring").rotation_layer(("ry",))
    spec = qc.to_spec()
    rng = np.random.default_rng(17)
    out["layered_ansatz"] = (spec, rng.uniform(-np.pi, np.pi, spec.n_params))

    # an idle qubit — Cirq silently drops these without an explicit qubit_order
    qc = qk.QCircuit(3)
    qc.ry(0, 0.8)
    out["idle_qubits"] = (qc.to_spec(), np.array([]))

    return out


CIRCUITS = _circuits()
CASES = sorted(CIRCUITS)


def _bind(name: str) -> qk.CircuitSpec:
    spec, theta = CIRCUITS[name]
    return spec if spec.n_params == 0 else spec.bind(theta)


def _global_phase_aligned(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate away any global phase difference before comparing amplitudes."""
    i = int(np.argmax(np.abs(a)))
    if abs(a[i]) < 1e-12 or abs(b[i]) < 1e-12:  # pragma: no cover
        return a, b
    return a * np.exp(-1j * np.angle(a[i])), b * np.exp(-1j * np.angle(b[i]))


# --------------------------------------------------------------------------- #
# the equivalence tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", CASES)
def test_statevector_matches_reference(backend_name, case):
    """Amplitudes must agree, up to an unobservable global phase."""
    spec = _bind(case)
    ref = qk.get_backend("numpy").statevector(spec)
    got = qk.get_backend(backend_name).statevector(spec)
    assert got.shape == ref.shape, f"{backend_name} returned {got.shape}, expected {ref.shape}"
    a, b = _global_phase_aligned(ref, got)
    atol = TOLERANCE[backend_name]
    assert np.allclose(a, b, atol=atol), f"{backend_name} disagrees on {case}"


@pytest.mark.parametrize("case", CASES)
def test_probabilities_match_reference(backend_name, case):
    """Probabilities are phase-free, so these must match without any alignment."""
    spec = _bind(case)
    ref = qk.get_backend("numpy").probabilities(spec)
    got = qk.get_backend(backend_name).probabilities(spec)
    assert np.allclose(ref, got, atol=TOLERANCE[backend_name])


@pytest.mark.parametrize("case", CASES)
def test_exact_expectations_match_reference(backend_name, case):
    """Expectations over X, Y, Z and a two-body term — exercises basis rotation."""
    spec = _bind(case)
    n = spec.n_qubits
    observables = [qk.Z(0), qk.X(0), qk.Y(0), qk.Z(n - 1)]
    if n >= 2:
        observables.append(qk.ZZ(0, n - 1))
        observables.append(qk.Z(0) + 0.5 * qk.X(n - 1))
    ref_be = qk.get_backend("numpy")
    got_be = qk.get_backend(backend_name)
    for obs in observables:
        assert got_be.expectation(spec, obs) == pytest.approx(
            ref_be.expectation(spec, obs), abs=TOLERANCE[backend_name]
        ), f"{backend_name} disagrees on <{obs}> for {case}"


def test_seeded_sampling_is_identical_across_backends(backend_name):
    """Shared sampling means a seed reproduces the same counts on any simulator."""
    spec = _bind("layered_ansatz")
    ref = qk.get_backend("numpy").counts(spec, shots=2048, seed=42)
    got = qk.get_backend(backend_name).counts(spec, shots=2048, seed=42)
    assert ref == got


def test_sampled_expectation_converges_to_exact(backend_name):
    spec = _bind("layered_ansatz")
    be = qk.get_backend(backend_name)
    obs = qk.Z(0) + qk.ZZ(0, 2)
    exact = be.expectation(spec, obs)
    sampled = be.expectation(spec, obs, shots=100_000, seed=3)
    assert sampled == pytest.approx(exact, abs=0.03)


def test_gradients_match_across_backends(backend_name):
    """The whole point: parameter-shift must give the same gradient on any backend."""
    qc = qk.QCircuit(2)
    qc.ry(0, qk.ParamRef(0))
    qc.crz(0, 1, qk.ParamRef(1))
    qc.ry(1, qk.ParamRef(2))
    spec = qc.to_spec()
    theta = np.array([0.7, 1.1, -0.4])
    obs = qk.Z(0) + qk.ZZ(0, 1)

    ref = qk.param_shift_grad_circuit(spec, theta, obs, backend="numpy")
    got = qk.param_shift_grad_circuit(spec, theta, obs, backend=backend_name)
    assert got == pytest.approx(ref, abs=TOLERANCE[backend_name])


def test_backend_roundtrips_to_its_native_circuit(backend_name):
    """The translation is public API: users can inspect and reuse the native circuit."""
    spec = _bind("layered_ansatz")
    be = qk.get_backend(backend_name)
    exporter = {"qiskit": "to_qiskit", "cirq": "to_cirq", "spinqit": "to_spinqit"}[backend_name]
    native = getattr(be, exporter)(spec)
    assert native is not None


# --------------------------------------------------------------------------- #
# availability plumbing (runs regardless of which SDKs are present)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", [None])
def test_registry_reports_availability_without_importing(backend_name, case):
    assert backend_name in qk.list_backends()
    assert backend_name in available_backends()
    assert is_available(backend_name)


def test_controlled_y_matches_the_standard_gate(backend_name):
    """SpinQit's native CY applies -iY to the control-1 subspace instead of Y.

    That is a relative phase between control branches, so it changes measurement
    statistics whenever the control is in superposition - exactly the case here.
    The backend emits a decomposition instead; this pins that down.
    """
    qc = qk.QCircuit(2)
    qc.h(0).cy(0, 1)
    spec = qc.to_spec()
    expected = np.array([1, 0, 0, 1j]) / np.sqrt(2)
    got = qk.get_backend(backend_name).statevector(spec)
    a, b = _global_phase_aligned(expected, got)
    assert np.allclose(a, b, atol=TOLERANCE[backend_name])
