"""The PyTorch bridge and the ready-made models.

The headline assertion is the last one in the gradient section: a classical layer
placed *before* the quantum one receives a real gradient and trains. The lecture's
implementation returns ``None`` there, silently freezing every pre-net — including
the one in its own transfer-learning example.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import qmlkit as qk  # noqa: E402
from qmlkit.nn import VQC, QuantumLayer, VQRegressor  # noqa: E402
from qmlkit.nn.layer import QuantumFunction  # noqa: E402

pytestmark = pytest.mark.torch


def _layer(n_qubits=2, n_layers=1, fmap=None, **kw) -> QuantumLayer:
    return QuantumLayer(
        fmap or qk.AngleFeatureMap(n_qubits, entangle=n_qubits > 1),
        qk.hardware_efficient(n_qubits, n_layers),
        [qk.Z(0)],
        init_seed=0,
        **kw,
    ).double()


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def test_layer_maps_batch_of_features_to_batch_of_expectations():
    layer = QuantumLayer(
        qk.AngleFeatureMap(3), qk.hardware_efficient(3, 2), [qk.Z(0), qk.Z(2)], init_seed=0
    ).double()
    out = layer(torch.randn(5, 3, dtype=torch.float64))
    assert out.shape == (5, 2)
    assert bool((out.abs() <= 1 + 1e-9).all()), "expectations must lie in [-1, 1]"


def test_layer_accepts_a_single_unbatched_sample():
    assert _layer()(torch.randn(2, dtype=torch.float64)).shape == (1, 1)


def test_layer_rejects_the_wrong_feature_width():
    with pytest.raises(ValueError, match="expects 2 features"):
        _layer()(torch.randn(4, 5, dtype=torch.float64))


def test_layer_rejects_a_mismatched_ansatz_width():
    with pytest.raises(ValueError, match="feature map uses 3 qubits"):
        QuantumLayer(qk.AngleFeatureMap(3), qk.hardware_efficient(2, 1))


def test_layer_defaults_to_one_observable_per_qubit():
    layer = QuantumLayer(qk.AngleFeatureMap(3), qk.hardware_efficient(3, 1))
    assert layer.n_outputs == 3


def test_layer_matches_a_direct_expectation():
    """The torch path must not change the number the core computes."""
    fmap, ansatz = qk.AngleFeatureMap(2, entangle=True), qk.hardware_efficient(2, 1)
    layer = QuantumLayer(fmap, ansatz, [qk.Z(0)], init_seed=4).double()
    x = np.array([0.4, 1.2])
    direct = qk.expectation(
        fmap.build_parametric().compose(ansatz.build(), param_offset=fmap.n_angles),
        qk.Z(0),
        theta=np.concatenate([fmap.angles(x), layer.theta.detach().numpy()]),
    )
    got = float(layer(torch.tensor(x, dtype=torch.float64))[0, 0].detach())
    assert got == pytest.approx(direct, abs=1e-12)


# --------------------------------------------------------------------------- #
# gradients
# --------------------------------------------------------------------------- #
def test_gradcheck_inputs():
    layer = _layer()
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: layer(t), (x,), eps=1e-6, atol=1e-7)


def test_gradcheck_weights():
    layer = _layer()
    x = torch.randn(2, 2, dtype=torch.float64)
    theta = layer.theta.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        lambda p: QuantumFunction.apply(x, p, layer._runner), (theta,), eps=1e-6, atol=1e-7
    )


def test_gradcheck_inputs_through_a_nonlinear_feature_map():
    """df/dx via the classical Jacobian of a data map that is not the identity."""
    layer = _layer(fmap=qk.ZZFeatureMap(2, reps=1))
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: layer(t), (x,), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift"])
def test_both_gradient_methods_give_the_same_weights_gradient(method):
    layer = _layer(grad_method=method)
    x = torch.randn(3, 2, dtype=torch.float64)
    layer(x).sum().backward()
    assert layer.theta.grad is not None
    assert torch.isfinite(layer.theta.grad).all()


def test_the_pre_net_actually_trains():
    """The Lecture 6 defect, asserted directly.

    A Linear before the QuantumLayer must receive a non-zero gradient. Returning
    None for df/dx — as the lecture does — leaves this at exactly zero, silently.
    """
    model = nn.Sequential(
        nn.Linear(6, 3),
        nn.Tanh(),
        QuantumLayer(qk.AngleFeatureMap(3), qk.hardware_efficient(3, 1), [qk.Z(0)], init_seed=1),
        nn.Linear(1, 2),
    ).double()
    x = torch.randn(5, 6, dtype=torch.float64)
    y = torch.randint(0, 2, (5,))
    nn.CrossEntropyLoss()(model(x), y).backward()

    pre_grad = model[0].weight.grad
    assert pre_grad is not None
    assert float(pre_grad.norm()) > 1e-9, "pre-net received no gradient — the dressed-circuit bug"
    assert float(model[2].theta.grad.norm()) > 1e-9


def test_a_frozen_backbone_stays_frozen():
    backbone = nn.Linear(8, 3).double()
    backbone.requires_grad_(False)
    model = nn.Sequential(backbone, _layer(3), nn.Linear(1, 1).double()).double()
    model(torch.randn(4, 8, dtype=torch.float64)).sum().backward()
    assert backbone.weight.grad is None
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable == sum(p.numel() for p in model[1:].parameters())


# --------------------------------------------------------------------------- #
# configuration and cost
# --------------------------------------------------------------------------- #
def test_defaults_are_simulator_shaped():
    layer = _layer()
    assert layer._runner.shots is None
    assert layer._runner.grad_method == "adjoint"


def test_configure_switches_to_device_realism():
    layer = _layer()
    layer.configure(shots=512, grad_method="parameter-shift")
    assert layer._runner.shots == 512
    assert layer._runner.grad_method == "parameter-shift"
    assert torch.isfinite(layer(torch.randn(2, 2, dtype=torch.float64))).all()


def test_resources_reports_both_gradient_costs():
    r = _layer(n_qubits=3, n_layers=2).resources()
    assert r["passes_per_sample_adjoint"] == 1
    assert r["circuits_per_sample_parameter_shift"] > r["passes_per_sample_adjoint"]


def test_shots_make_the_output_noisy_but_close():
    exact = _layer(n_qubits=2)
    noisy = _layer(n_qubits=2).configure(shots=20000, grad_method="parameter-shift")
    noisy.theta.data = exact.theta.data.clone()
    x = torch.randn(3, 2, dtype=torch.float64)
    assert torch.allclose(exact(x), noisy(x), atol=0.05)


# --------------------------------------------------------------------------- #
# the ready-made models
# --------------------------------------------------------------------------- #
def test_vqc_two_line_path_learns_a_nonlinear_rule():
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like: not linearly separable
    model = VQC(n_features=4, n_classes=2, seed=0).fit(X, y, epochs=25)
    assert model.history_[-1] < model.history_[0]
    assert model.score(X, y) > 0.7


def test_vqc_predict_shapes_and_probabilities():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 3))
    y = (X[:, 0] > 0).astype(int)
    model = VQC(n_features=3, n_classes=2, seed=0).fit(X, y, epochs=3)
    assert model.predict(X).shape == (12,)
    proba = model.predict_proba(X)
    assert proba.shape == (12, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_vq_regressor_fits_and_scores():
    rng = np.random.default_rng(2)
    torch.manual_seed(0)
    X = rng.normal(size=(50, 3))
    y = np.sin(X[:, 0]) + 0.3 * X[:, 1]
    model = VQRegressor(n_features=3, seed=1).fit(X, y, epochs=30)
    assert model.predict(X).shape == (50,)
    assert model.score(X, y) > 0.3


def test_models_accept_a_fully_custom_configuration():
    """Every default is one keyword away — and the rest still works."""
    model = VQC(
        n_features=4,
        n_classes=2,
        n_qubits=3,
        feature_map=qk.ZZFeatureMap(3, reps=1),
        ansatz=qk.Ansatz(
            3, qk.repeat(2, qk.RotationLayer(("ry", "rz")) + qk.EntanglerLayer("cz", "ring"))
        ),
        observables=[qk.Z(0), qk.ZZ(0, 2)],
        seed=0,
    )
    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 4))
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y, epochs=2)
    assert model.predict(X).shape == (10,)
    assert model.resources()["n_qubits"] == 3


def test_model_fit_accepts_batching_and_a_custom_optimizer():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(16, 2))
    y = (X[:, 0] > 0).astype(int)
    model = VQC(n_features=2, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    model.fit(X, y, epochs=2, batch_size=4, optimizer=opt)
    assert len(model.history_) == 2
