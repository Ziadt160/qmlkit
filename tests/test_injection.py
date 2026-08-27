"""Every algorithm must let you supply your own circuit.

The rule this file enforces: **an algorithm owns its loop, not its ansatz.** QCNN,
QLSTM, QCBM, VQE, QAOA — each of them is a *structure* that a paper then fills with
a particular circuit, and there are always more papers than classes. So a variant is
an argument, never a subclass.

Accepting the argument is not enough — a constructor that takes ``ansatz`` and then
quietly ignores it looks identical from the outside and would pass any test that
only checked the call succeeded. So every case below injects two ansätze of
*different sizes* and asserts the model's own parameter count follows.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk

torch = pytest.importorskip("torch")

from qmlkit.algorithms import QAOA, VQE, ising_hamiltonian  # noqa: E402
from qmlkit.generative import QCBM  # noqa: E402
from qmlkit.nn.advanced import QLSTM, DressedQuantumNet, MPSLayer, QCNNLayer  # noqa: E402
from qmlkit.nn.layer import QuantumLayer  # noqa: E402
from qmlkit.nn.models import VQC, VQRegressor  # noqa: E402

N_QUBITS = 4


def tiny() -> qk.Ansatz:
    """4 parameters."""
    return qk.Ansatz(N_QUBITS, qk.RotationLayer("rx"), "tiny")


def big() -> qk.Ansatz:
    """36 parameters — deliberately a different number from `tiny`."""
    return qk.Ansatz(N_QUBITS, qk.repeat(3, qk.RotationLayer(("rx", "ry", "rz"))), "big")


def _quantum_weights(model: object) -> int:
    """However a model stores its circuit parameters, count them."""
    if isinstance(model, torch.nn.Module):
        return sum(int(p.numel()) for name, p in model.named_parameters() if "theta" in name)
    ansatz = getattr(model, "ansatz", None)
    if ansatz is not None:
        return int(ansatz.n_weights)
    raise AssertionError(f"cannot find the parameters of {type(model).__name__}")


#: name -> a constructor taking the ansatz to inject. Everything the library calls an
#: algorithm belongs here; adding one without a row is what this file exists to catch.
BUILDERS = {
    "QuantumLayer": lambda a: QuantumLayer(qk.AngleFeatureMap(N_QUBITS), a, [qk.Z(0)]),
    "VQC": lambda a: VQC(n_features=N_QUBITS, n_classes=2, ansatz=a),
    "VQRegressor": lambda a: VQRegressor(n_features=N_QUBITS, ansatz=a),
    "QLSTM": lambda a: QLSTM(n_inputs=2, hidden_size=2, n_qubits=N_QUBITS, ansatz=a),
    "QCNNLayer": lambda a: QCNNLayer(N_QUBITS, ansatz=a),
    "MPSLayer": lambda a: MPSLayer(N_QUBITS, ansatz=a),
    "DressedQuantumNet": lambda a: DressedQuantumNet(
        None, in_features=N_QUBITS, n_qubits=N_QUBITS, n_outputs=2, ansatz=a
    ),
    "QCBM": lambda a: QCBM(N_QUBITS, ansatz=a),
    "VQE": lambda a: VQE(ising_hamiltonian(N_QUBITS), ansatz=a),
    "QAOA": lambda a: QAOA([(0, 1), (1, 2), (2, 3)], ansatz=a),
}


@pytest.mark.torch
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_algorithm_accepts_a_custom_ansatz(name):
    """It must take one at all."""
    model = BUILDERS[name](tiny())
    assert model is not None


@pytest.mark.torch
@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_the_injected_ansatz_is_actually_used(name):
    """And it must *use* it — silently ignoring the argument is the real failure."""
    small = _quantum_weights(BUILDERS[name](tiny()))
    large = _quantum_weights(BUILDERS[name](big()))
    assert small == pytest.approx(tiny().n_weights * (small // max(tiny().n_weights, 1)))
    assert large > small, (
        f"{name} accepted a {big().n_weights}-parameter ansatz but kept "
        f"{large} weights — the argument is being ignored"
    )


@pytest.mark.torch
def test_a_hand_written_ansatz_works_everywhere():
    """Not just the built-ins: an ansatz invented on the spot has to be accepted."""
    invented = qk.Ansatz(
        N_QUBITS,
        qk.repeat(2, qk.RotationLayer(("ry",)) + qk.EntanglerLayer("cz", "alternating")),
        "invented_here",
    )
    for name, build in BUILDERS.items():
        model = build(invented)
        assert _quantum_weights(model) > 0, f"{name} lost the invented ansatz"


# --------------------------------------------------------------------------- #
# feature maps are the other half of the same rule
# --------------------------------------------------------------------------- #
FEATURE_MAP_BUILDERS = {
    "QuantumKernel": lambda f: qk.QuantumKernel(f),
    "QSVC": lambda f: qk.QSVC(f),
    "QSVR": lambda f: qk.QSVR(f),
    "NearestFidelityClassifier": lambda f: qk.NearestFidelityClassifier(f),
}


@pytest.mark.parametrize("name", sorted(FEATURE_MAP_BUILDERS))
def test_kernel_models_take_any_feature_map(name):
    """A kernel method is defined by its embedding, so that has to be the argument."""
    for fmap in (qk.AngleFeatureMap(2), qk.ZZFeatureMap(2, reps=2), qk.ZFeatureMap(2)):
        model = FEATURE_MAP_BUILDERS[name](fmap)
        assert model.feature_map is fmap


# --------------------------------------------------------------------------- #
# structural choices that are not an ansatz
# --------------------------------------------------------------------------- #
def test_qcnn_filter_and_pooling_are_arguments():
    counts = {name: qk.qcnn_ansatz(8, filter=name).n_params for name in qk.list_conv_filters()}
    assert len(set(counts.values())) == len(counts), f"filters are not distinct: {counts}"
    assert qk.qcnn_ansatz(8, pool="controlled").n_params > qk.qcnn_ansatz(8).n_params


def test_mps_and_ttn_share_the_qcnn_filter_registry():
    """The same two-qubit block, so the same extension point — not three of them."""
    for factory in (qk.mps_ansatz, qk.tree_tensor_network):
        cheap = factory(6, filter="ry_cx").n_params
        general = factory(6, filter="su4").n_params
        assert general > cheap, f"{factory.__name__} ignored the filter"


def test_qaoa_mixer_is_an_argument():
    x_only = qk.qaoa_ansatz(4, p=2, mixer="x").resources()
    both = qk.qaoa_ansatz(4, p=2, mixer="xy").resources()
    assert both["n_1q"] > x_only["n_1q"]
    assert both["n_params"] == x_only["n_params"]  # still 2p angles, whatever the mixer


def test_boltzmann_machine_connectivity_is_an_argument():
    """The coupling graph is this model's structure, so it is not hard-coded."""
    from qmlkit.generative import QuantumBoltzmannMachine

    chain = QuantumBoltzmannMachine(2, 2, seed=0)
    full = QuantumBoltzmannMachine(2, 2, seed=0, pattern="full")
    restricted = QuantumBoltzmannMachine(2, 2, seed=0, edges=[(0, 2), (0, 3), (1, 2), (1, 3)])
    assert len(chain.couplings) == 3
    assert len(full.couplings) == 6
    assert sorted(restricted.couplings) == [(0, 2), (0, 3), (1, 2), (1, 3)]
    # a genuine RBM: no visible-visible or hidden-hidden edge
    assert all(a < 2 <= b for a, b in restricted.couplings)


def test_vqe_and_qaoa_take_any_optimizer():
    """The optimiser is a function, so a custom one needs no registration."""
    calls = {"n": 0}

    def my_optimizer(loss, theta0, **kw):
        calls["n"] += 1
        theta = np.asarray(theta0, dtype=float)
        return theta, [float(loss(theta))]

    result = VQE(ising_hamiltonian(3), optimizer=my_optimizer).run(seed=0)
    assert calls["n"] == 1
    assert np.isfinite(result.energy)
