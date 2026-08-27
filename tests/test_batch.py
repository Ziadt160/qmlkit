"""Batched execution, checked against the one-at-a-time path it replaces.

A batched simulator is a performance change that must not be a semantic one, so
almost every test here is an equality against the loop it is meant to replace. The
loop is the reference implementation; batching is an optimisation, and an
optimisation that changes an answer is a bug regardless of how fast it is.

The riskiest part is not the contraction — it is ``_VECTORISED``, the closed-form
batched gate matrices. A vectorised ``ry`` with a sign error produces a circuit that
runs and returns numbers in range, which is precisely the plausible-wrong-number
failure this library exists to catch. So every entry in that table is asserted equal
to the scalar ``gate_matrix`` it stands in for, over a range of angles including the
degenerate ones.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import _VECTORISED, NumpyBackend
from qmlkit.core.gates import gate_matrix
from qmlkit.core.observables import (
    expectation_from_statevector,
    expectation_from_statevectors,
)

ANGLES = [-np.pi, -1.3, -0.0, 0.0, 0.4, np.pi / 2, np.pi, 2 * np.pi, 7.9]


def _thetas(n_params: int, batch: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (batch, n_params))


# --------------------------------------------------------------------------- #
# the vectorised gate matrices — the part that could be silently wrong
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate", sorted(_VECTORISED))
def test_vectorised_matrices_equal_the_scalar_reference(gate):
    """Every closed-form batched builder against the gate it stands in for."""
    angles = np.array(ANGLES)
    batched = _VECTORISED[gate](angles)
    assert batched.shape[0] == angles.size
    for i, angle in enumerate(angles):
        np.testing.assert_allclose(
            batched[i], gate_matrix(gate, (float(angle),)), atol=1e-15, rtol=0
        )


@pytest.mark.parametrize("gate", sorted(_VECTORISED))
def test_vectorised_matrices_are_unitary(gate):
    for matrix in _VECTORISED[gate](np.array(ANGLES)):
        identity = np.eye(matrix.shape[0])
        np.testing.assert_allclose(matrix.conj().T @ matrix, identity, atol=1e-14)


def test_the_vectorised_table_covers_the_rotations_that_dominate():
    """A regression guard: dropping one of these silently costs the speedup."""
    assert {"rx", "ry", "rz", "phase", "crx", "cry", "crz"} <= set(_VECTORISED)


# --------------------------------------------------------------------------- #
# batched statevectors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits", [1, 2, 3, 5, 8])
def test_statevector_batch_equals_the_loop(n_qubits):
    backend = qk.get_backend("numpy")
    ansatz = qk.hardware_efficient(n_qubits, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, seed=n_qubits)
    np.testing.assert_allclose(
        backend.statevector_batch(spec, thetas),
        np.stack([qk.statevector(spec, t) for t in thetas]),
        atol=1e-14,
    )


def test_the_wide_fallback_path_is_still_exact():
    """Above the crossover the loop runs instead, and must give the same answer."""
    backend = NumpyBackend()
    backend.batch_max_qubits = 2  # force the fallback on a circuit wider than that
    ansatz = qk.hardware_efficient(4, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params)
    np.testing.assert_allclose(
        backend.statevector_batch(spec, thetas),
        np.stack([qk.statevector(spec, t) for t in thetas]),
        atol=1e-14,
    )


def test_batching_honours_paramref_scale_and_offset():
    """One logical parameter can drive an angle of `scale*theta + offset`."""
    qc = qk.QCircuit(2)
    qc.ry(0, qk.ParamRef(0, scale=2.0, offset=0.5))
    qc.rz(1, qk.ParamRef(0, scale=-1.0))
    spec = qc.to_spec()
    thetas = _thetas(1, batch=5)
    np.testing.assert_allclose(
        qk.get_backend("numpy").statevector_batch(spec, thetas),
        np.stack([qk.statevector(spec, t) for t in thetas]),
        atol=1e-14,
    )


def test_batching_handles_a_shared_parameter():
    """Weight tying: one logical parameter driving several gates."""
    qc = qk.QCircuit(3)
    shared = qc.param()
    qc.rotation_layer(("ry",), shared=shared)
    spec = qc.to_spec()
    assert len(spec.occurrences_of(0)) == 3
    thetas = _thetas(spec.n_params, batch=4)
    np.testing.assert_allclose(
        qk.get_backend("numpy").statevector_batch(spec, thetas),
        np.stack([qk.statevector(spec, t) for t in thetas]),
        atol=1e-14,
    )


def test_a_custom_gate_falls_back_and_stays_correct():
    """A gate registered at run time has no vectorised builder; it must still work."""
    qk.register_gate(
        qk.GateDef(
            "batch_probe",
            n_qubits=1,
            n_params=1,
            matrix=lambda t: np.array(
                [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=complex
            ),
            frequencies=(1.0,),
        )
    )
    assert "batch_probe" not in _VECTORISED
    qc = qk.QCircuit(2)
    qc.apply("batch_probe", 0, qc.param())
    qc.cx(0, 1)
    spec = qc.to_spec()
    thetas = _thetas(1, batch=4)
    np.testing.assert_allclose(
        qk.get_backend("numpy").statevector_batch(spec, thetas),
        np.stack([qk.statevector(spec, t) for t in thetas]),
        atol=1e-14,
    )


def test_a_mismatched_parameter_width_is_an_error():
    ansatz = qk.hardware_efficient(3, 2)
    with pytest.raises(ValueError, match=f"{ansatz.n_params} parameters"):
        qk.get_backend("numpy").statevector_batch(ansatz.build(), _thetas(ansatz.n_params - 1))


# --------------------------------------------------------------------------- #
# batched expectation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "obs",
    [
        qk.Z(0),
        qk.X(1),
        qk.Z(0) + 0.5 * qk.ZZ(0, 2),
        qk.X(0) + qk.Y(1) + 2.0 * qk.Z(2),
        qk.Z(0) * 0.0 + qk.I(),
    ],
    ids=["z", "x", "sum", "mixed-basis", "identity"],
)
def test_expectation_from_statevectors_matches_the_scalar_version(obs):
    rng = np.random.default_rng(0)
    n = 3
    states = rng.normal(size=(5, 2**n)) + 1j * rng.normal(size=(5, 2**n))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    np.testing.assert_allclose(
        expectation_from_statevectors(obs, states, n),
        [expectation_from_statevector(obs, s, n) for s in states],
        atol=1e-13,
    )


def test_expectation_over_equals_the_loop():
    ansatz = qk.hardware_efficient(4, 3)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=9)
    obs = qk.Z(0) + 0.5 * qk.ZZ(1, 3) + (-0.25) * qk.X(2)
    np.testing.assert_allclose(
        qk.expectation_over(spec, thetas, obs),
        [qk.expectation(spec, obs, t) for t in thetas],
        atol=1e-13,
    )


def test_expectation_over_accepts_a_single_vector():
    ansatz = qk.hardware_efficient(2, 1)
    spec = ansatz.build()
    theta = np.full(ansatz.n_params, 0.3)
    got = qk.expectation_over(spec, theta, qk.Z(0))
    assert got.shape == (1,)
    assert got[0] == pytest.approx(qk.expectation(spec, qk.Z(0), theta))


def test_expectation_over_samples_when_asked():
    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=4)
    sampled = qk.expectation_over(spec, thetas, qk.Z(0), shots=4096, seed=0)
    exact = qk.expectation_over(spec, thetas, qk.Z(0))
    assert sampled.shape == exact.shape
    assert np.max(np.abs(sampled - exact)) < 0.1  # shot noise, not a different quantity


def test_a_sampling_only_backend_refuses_exact_batching():
    """The same refusal the scalar path gives, not a silent zero."""

    class Device(qk.Backend):
        name = "batch_probe_device"
        supports_statevector = False
        supports_exact = False

        def counts(self, spec, shots, seed=None):  # pragma: no cover - not reached
            return {"0" * spec.n_qubits: shots}

    ansatz = qk.hardware_efficient(2, 1)
    with pytest.raises(ValueError, match="no exact mode"):
        Device().expectation_over(ansatz.build(), _thetas(ansatz.n_params), qk.Z(0))


def test_every_backend_inherits_a_working_batch_path():
    """The base class default must work for a backend that only has a statevector."""

    class Minimal(qk.Backend):
        name = "batch_probe_minimal"
        supports_statevector = True
        supports_exact = True

        def statevector(self, spec):
            return qk.get_backend("numpy").statevector(spec)

    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=4)
    np.testing.assert_allclose(
        Minimal().expectation_over(spec, thetas, qk.Z(0)),
        qk.expectation_over(spec, thetas, qk.Z(0)),
        atol=1e-13,
    )


# --------------------------------------------------------------------------- #
# the layer that actually benefits
# --------------------------------------------------------------------------- #
def test_the_torch_layer_is_unchanged_by_batching():
    """The speedup must be invisible in the numbers."""
    torch = pytest.importorskip("torch")
    from qmlkit.core.backends.numpy_backend import NumpyBackend as Backend

    X, _ = qk.datasets.make_moons(n_samples=24, seed=0)
    xt = torch.as_tensor(X, dtype=torch.get_default_dtype())
    torch.manual_seed(0)
    model = qk.VQC(n_features=2, n_classes=2, n_qubits=4, n_layers=2, seed=0)

    original = Backend.batch_max_qubits
    try:
        Backend.batch_max_qubits = 0  # force one-at-a-time
        loop = model(xt).detach().numpy()
        Backend.batch_max_qubits = original
        batched = model(xt).detach().numpy()
    finally:
        Backend.batch_max_qubits = original
    np.testing.assert_allclose(loop, batched, atol=1e-12)


def test_a_single_row_still_returns_a_single_row():
    """Batching must not turn a 1-D input into a 2-D output."""
    torch = pytest.importorskip("torch")

    layer = qk.QuantumLayer(
        qk.AngleFeatureMap(3), qk.hardware_efficient(3, 1), [qk.Z(0), qk.Z(1)]
    )
    one = layer(torch.zeros(3))
    many = layer(torch.zeros((4, 3)))
    # a 1-D input is treated as a batch of one, which is what the layer has always
    # done -- batching must not change that either way
    assert one.shape == (1, 2)
    assert many.shape == (4, 2)
    np.testing.assert_allclose(one.detach().numpy()[0], many[0].detach().numpy(), atol=1e-13)
