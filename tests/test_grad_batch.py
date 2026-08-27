"""Batched gradients, against the per-sample gradients they replace.

Two claims are under test, and they are different in kind.

The first is *correctness*: a batched gradient must equal the loop exactly, including
for the cases where batching could plausibly get it wrong — weight-tied parameters
whose occurrences must be shifted one at a time, rescaled parameter references that
carry a chain-rule factor, and multi-frequency gates whose rule has four terms rather
than two.

The second is *reach*: :func:`param_shift_grad_batch` never inspects a state, so it
must work on every backend, not only the NumPy one. That is asserted by running it
through each installed backend and comparing, because "should work everywhere" is a
claim about code that has not been run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import _VECTORISED, _VECTORISED_D
from qmlkit.core.gates import gate_derivative
from qmlkit.gradients.batch import (
    adjoint_grad_batch,
    grad_batch,
    param_shift_grad_batch,
)

ANGLES = np.array([-np.pi, -1.3, 0.0, 0.4, np.pi / 2, np.pi, 2 * np.pi, 7.9])


def _thetas(n_params: int, batch: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (batch, n_params))


def _loop(spec, thetas, obs, **kwargs) -> np.ndarray:
    return np.stack([qk.grad(spec, t, obs, **kwargs) for t in thetas])


# --------------------------------------------------------------------------- #
# the vectorised derivatives — the part that could be silently wrong
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("gate", sorted(_VECTORISED_D))
def test_vectorised_derivatives_equal_the_scalar_reference(gate):
    batched = _VECTORISED_D[gate](ANGLES)
    assert batched.shape[0] == ANGLES.size
    for i, angle in enumerate(ANGLES):
        np.testing.assert_allclose(
            batched[i], gate_derivative(gate, (float(angle),)), atol=1e-15, rtol=0
        )


def test_the_derivative_table_covers_the_same_gates_as_the_matrix_table():
    """A gate with a batched matrix but no batched derivative silently loses the win."""
    assert set(_VECTORISED_D) == set(_VECTORISED)


# --------------------------------------------------------------------------- #
# equality with the loop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits,n_layers", [(1, 1), (2, 1), (3, 2), (4, 3), (5, 2)])
def test_adjoint_batch_equals_the_loop(n_qubits, n_layers):
    ansatz = qk.hardware_efficient(n_qubits, n_layers)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, seed=n_qubits)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, n_qubits - 1) if n_qubits > 1 else qk.Z(0)
    np.testing.assert_allclose(
        adjoint_grad_batch(spec, thetas, obs),
        _loop(spec, thetas, obs, method="adjoint"),
        atol=1e-12,
    )


@pytest.mark.parametrize("n_qubits,n_layers", [(1, 1), (2, 1), (3, 2), (4, 2)])
def test_param_shift_batch_equals_the_loop(n_qubits, n_layers):
    ansatz = qk.hardware_efficient(n_qubits, n_layers)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=5, seed=n_qubits)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, n_qubits - 1) if n_qubits > 1 else qk.Z(0)
    np.testing.assert_allclose(
        param_shift_grad_batch(spec, thetas, obs),
        _loop(spec, thetas, obs, method="parameter-shift"),
        atol=1e-11,
    )


def test_the_two_batched_routes_agree_with_each_other():
    """Independent derivations of the same quantity — the library's own standard."""
    ansatz = qk.strongly_entangling(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=4)
    obs = qk.Z(0) + qk.X(1)
    np.testing.assert_allclose(
        adjoint_grad_batch(spec, thetas, obs),
        param_shift_grad_batch(spec, thetas, obs),
        atol=1e-11,
    )


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_a_multi_frequency_gate_uses_its_real_rule(method):
    """`crz` has two frequencies and a four-term rule, not the familiar two-term one."""
    qc = qk.QCircuit(2)
    qc.ry(0, qk.ParamRef(0))
    qc.crz(0, 1, qk.ParamRef(1))
    spec = qc.to_spec()
    assert qk.grad_circuit_cost(spec) == 6  # 2 + 4, not 2 * 2
    thetas = _thetas(2, batch=4)
    np.testing.assert_allclose(
        grad_batch(spec, thetas, qk.Z(1), method=method),
        _loop(spec, thetas, qk.Z(1), method=method),
        atol=1e-11,
    )


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_weight_tying_sums_over_occurrences(method):
    """The case that slot-space batching exists for: one parameter, three gates."""
    qc = qk.QCircuit(3)
    shared = qc.param()
    qc.rotation_layer(("ry",), shared=shared)
    spec = qc.to_spec()
    assert len(spec.occurrences_of(0)) == 3
    thetas = _thetas(1, batch=4)
    obs = qk.Z(0) + qk.Z(1) + qk.Z(2)
    np.testing.assert_allclose(
        grad_batch(spec, thetas, obs, method=method),
        _loop(spec, thetas, obs, method=method),
        atol=1e-11,
    )


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_a_rescaled_reference_carries_its_chain_rule(method):
    qc = qk.QCircuit(2)
    qc.ry(0, qk.ParamRef(0, scale=2.0, offset=0.5))
    qc.rz(1, qk.ParamRef(0, scale=-1.0))
    spec = qc.to_spec()
    thetas = _thetas(1, batch=4)
    np.testing.assert_allclose(
        grad_batch(spec, thetas, qk.Z(0) + qk.Z(1), method=method),
        _loop(spec, thetas, qk.Z(0) + qk.Z(1), method=method),
        atol=1e-11,
    )


def test_gradients_flow_through_the_encoding():
    """`df/dx`, which is what lets a classical pre-net train."""
    spec = qk.angle_encode([0.4, 1.1], trainable=True)
    thetas = np.array([[0.4, 1.1], [0.9, -0.2]])
    got = grad_batch(spec, thetas, qk.Z(0), method="parameter-shift")
    np.testing.assert_allclose(got[:, 0], -np.sin(thetas[:, 0]), atol=1e-11)
    np.testing.assert_allclose(got[:, 1], 0.0, atol=1e-11)


def test_a_parameterless_circuit_returns_an_empty_gradient():
    spec = qk.angle_encode([0.3, 0.8])
    got = param_shift_grad_batch(spec, np.zeros((3, 0)), qk.Z(0))
    assert got.shape == (3, 0)


# --------------------------------------------------------------------------- #
# reach: the parameter-shift route must work on every backend
# --------------------------------------------------------------------------- #
def test_param_shift_batch_agrees_across_every_installed_backend():
    """The claim that makes this a QML layer over *any* simulator, not just NumPy."""
    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=4)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
    reference = param_shift_grad_batch(spec, thetas, obs, backend="numpy")
    names = [n for n in qk.available_backends() if n != "numpy"]
    if not names:
        pytest.skip("no second backend installed")
    for name in names:
        np.testing.assert_allclose(
            param_shift_grad_batch(spec, thetas, obs, backend=name),
            reference,
            atol=1e-8,  # SpinQit carries a 1e-10 precision floor
            err_msg=f"backend {name!r} disagrees",
        )


def test_a_sampling_only_device_can_still_take_a_batched_gradient():
    """Parameter-shift never inspects a state, so a device is not excluded."""

    class Device(qk.Backend):
        name = "grad_batch_probe_device"
        supports_statevector = False
        supports_exact = False

        def __init__(self, seed=None):
            super().__init__(seed)
            self.circuits = 0

        def counts(self, spec, shots, seed=None):
            self.circuits += 1
            probs = np.abs(qk.get_backend("numpy").statevector(spec)) ** 2
            rng = np.random.default_rng(seed if seed is not None else 0)
            return {
                format(i, f"0{spec.n_qubits}b"): int(c)
                for i, c in enumerate(rng.multinomial(shots, probs / probs.sum()))
                if c
            }

    ansatz = qk.hardware_efficient(2, 1)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=3)
    device = Device()
    got = param_shift_grad_batch(spec, thetas, qk.Z(0), shots=8192, backend=device)
    assert got.shape == (3, ansatz.n_params)
    assert device.circuits > 0
    exact = param_shift_grad_batch(spec, thetas, qk.Z(0), backend="numpy")
    assert np.max(np.abs(got - exact)) < 0.2  # shot noise, not a different quantity


def test_adjoint_is_refused_on_a_backend_with_no_statevector():
    class Device(qk.Backend):
        name = "grad_batch_probe_nostate"
        supports_statevector = False
        supports_exact = False

        def counts(self, spec, shots, seed=None):  # pragma: no cover - not reached
            return {"0" * spec.n_qubits: shots}

    ansatz = qk.hardware_efficient(2, 1)
    with pytest.raises(ValueError, match="parameter-shift"):
        adjoint_grad_batch(ansatz.build(), _thetas(ansatz.n_params), qk.Z(0), backend=Device())


def test_chunking_does_not_change_the_answer():
    """A large fan-out is split; the split must be invisible."""
    from qmlkit.core.backends.numpy_backend import NumpyBackend

    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=8)
    whole = NumpyBackend()
    chunked = NumpyBackend()
    chunked.max_batch_rows = 7  # deliberately not a divisor of the fan-out
    np.testing.assert_allclose(
        param_shift_grad_batch(spec, thetas, qk.Z(0), backend=chunked),
        param_shift_grad_batch(spec, thetas, qk.Z(0), backend=whole),
        atol=1e-13,
    )


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_auto_picks_adjoint_when_exact_and_shift_when_sampling():
    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    thetas = _thetas(ansatz.n_params, batch=3)
    np.testing.assert_allclose(
        grad_batch(spec, thetas, qk.Z(0)),
        adjoint_grad_batch(spec, thetas, qk.Z(0)),
        atol=1e-14,
    )
    sampled = grad_batch(spec, thetas, qk.Z(0), shots=4096, seed=0)
    assert sampled.shape == (3, ansatz.n_params)


def test_adjoint_with_shots_is_refused_rather_than_ignored():
    ansatz = qk.hardware_efficient(2, 1)
    with pytest.raises(ValueError, match="exact"):
        grad_batch(ansatz.build(), _thetas(ansatz.n_params), qk.Z(0), "adjoint", shots=100)


def test_an_unbatchable_method_says_what_to_do_instead():
    ansatz = qk.hardware_efficient(2, 1)
    with pytest.raises(ValueError, match="qk.grad"):
        grad_batch(ansatz.build(), _thetas(ansatz.n_params), qk.Z(0), method="spsa")


# --------------------------------------------------------------------------- #
# the layer that benefits
# --------------------------------------------------------------------------- #
def test_training_gradients_are_unchanged_by_batching():
    """The speedup must be invisible in the numbers, to float32 round-trip."""
    torch = pytest.importorskip("torch")
    from qmlkit.nn.layer import _Runner

    X, y = qk.datasets.make_moons(n_samples=24, seed=0)
    xt = torch.as_tensor(X, dtype=torch.get_default_dtype())
    yt = torch.as_tensor(y, dtype=torch.long)
    loss_fn = torch.nn.CrossEntropyLoss()

    def grads_with(batchable):
        torch.manual_seed(0)
        model = qk.VQC(n_features=2, n_classes=2, n_qubits=3, n_layers=2, seed=0)
        original = _Runner._BATCHABLE
        try:
            _Runner._BATCHABLE = batchable
            model.zero_grad()
            loss_fn(model(xt), yt).backward()
            return [p.grad.clone() for p in model.parameters()]
        finally:
            _Runner._BATCHABLE = original

    default = _Runner._BATCHABLE
    for looped, batched in zip(
        grads_with(frozenset()), grads_with(default), strict=True
    ):
        torch.testing.assert_close(looped, batched, atol=1e-6, rtol=1e-5)


def test_an_unbatchable_grad_method_still_trains():
    """SPSA has no batched form; the layer must fall back, not fail."""
    torch = pytest.importorskip("torch")

    X, y = qk.datasets.make_moons(n_samples=8, seed=0)
    model = qk.VQC(n_features=2, n_classes=2, n_qubits=2, n_layers=1, grad_method="spsa", seed=0)
    xt = torch.as_tensor(X, dtype=torch.get_default_dtype())
    yt = torch.as_tensor(y, dtype=torch.long)
    model.zero_grad()
    torch.nn.CrossEntropyLoss()(model(xt), yt).backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in model.parameters())
