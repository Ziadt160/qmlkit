"""Generalized re-uploading, the structured layers, and the generative models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.ansatz.blocks import BuildContext
from qmlkit.core.builder import QCircuit
from qmlkit.core.execute import expectation
from qmlkit.core.ir import ParamRef
from qmlkit.fourier import spectrum
from qmlkit.generative import (
    QCBM,
    QGAN,
    QuantumBoltzmannMachine,
    QuantumHopfield,
    boltzmann,
    gaussian_kernel,
    ising_energy,
    kl_divergence,
    mmd_squared,
    partition_function,
    total_variation,
)

torch = pytest.importorskip("torch")

# Without this marker the file runs nowhere in CI: the core jobs have no torch and
# skip it, and the torch job selects by marker and never sees it.
pytestmark = pytest.mark.torch
nn = torch.nn


# --------------------------------------------------------------------------- #
# re-uploading is a pattern, not a structure
# --------------------------------------------------------------------------- #
def test_reupload_accepts_any_feature_map():
    for fmap in (
        qk.AngleFeatureMap(2),
        qk.ZZFeatureMap(2, reps=1),
        qk.PauliFeatureMap(2, paulis=("Z", "X", "ZZ"), reps=1),
    ):
        model = qk.reupload(fmap, n_layers=2)
        assert model.n_inputs == fmap.n_angles
        assert model.n_weights > 0
        assert model.bind(np.zeros(fmap.n_features), model.init(seed=0)) is not None


def test_reupload_accepts_any_trainable_block():
    custom = qk.RotationLayer(("rx", "rz")) + qk.EntanglerLayer("cz", "ring")
    model = qk.reupload(qk.AngleFeatureMap(3), n_layers=2, block=custom)
    assert model.n_weights == 2 * 3 * 2  # two layers, three wires, two rotations


def test_order_changes_the_circuit():
    """SW encodes first; WS transforms the state before the first upload."""
    sw = qk.reupload(qk.AngleFeatureMap(2), n_layers=2, order="SW").build()
    ws = qk.reupload(qk.AngleFeatureMap(2), n_layers=2, order="WS").build()
    assert [op.gate for op in sw.ops] != [op.gate for op in ws.ops]


def test_share_weights_ties_across_layers():
    free = qk.reupload(qk.AngleFeatureMap(2), n_layers=3)
    tied = qk.reupload(qk.AngleFeatureMap(2), n_layers=3, share_weights=True)
    assert tied.n_weights < free.n_weights
    occurrences = [len(tied.build().occurrences_of(i)) for i in range(tied.n_inputs, tied.n_params)]
    assert max(occurrences) > 1


def test_every_upload_references_the_same_inputs():
    """Re-uploading feeds the same data in again — it does not consume new features."""
    model = qk.reupload(qk.AngleFeatureMap(2), n_layers=3)
    spec = model.build()
    for i in range(model.n_inputs):
        assert len(spec.occurrences_of(i)) == 3


def test_two_different_feature_maps_in_one_model():
    """Not expressible with a single fixed re-uploading class."""
    zz, angle = qk.ZZFeatureMap(2, reps=1), qk.AngleFeatureMap(2, entangle=False)
    model = qk.Ansatz(
        2,
        qk.EncodingLayer(zz) + qk.RotationLayer(("rz", "ry")) + qk.EncodingLayer(angle),
        "mixed",
    )
    assert model.n_weights == 4
    assert model.n_inputs == zz.n_angles + angle.n_angles  # a sum, never a max
    assert model.build(np.zeros(model.n_params)) is not None  # full vector


def test_each_feature_map_owns_its_own_input_slots():
    """Two maps in one model must not land on each other's angles.

    The maps disagree about what angle ``i`` means - ``ZZFeatureMap`` emits ``2*x_i``
    for a Z term, ``AngleFeatureMap`` emits ``x_i`` - so a shared slot would feed the
    first map's *transformed* angles into the second and encode the wrong number
    without raising.
    """
    zz, angle = qk.ZZFeatureMap(2, reps=1), qk.AngleFeatureMap(2, rotation="ry", entangle=False)
    model = qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle))

    spec = model.build()
    referenced = {p.index for op in spec.ops for p in op.params if isinstance(p, ParamRef)}
    assert referenced == set(range(model.n_params))  # nothing reserved is left dead

    # the trailing ry layer reads the angle map's own slots, which follow the ZZ map's
    x, weights = np.array([0.3, 0.7]), np.zeros(model.n_weights)
    encoded = [op.params[0] for op in model.bind(x, weights).ops if op.gate == "ry"]
    assert encoded == pytest.approx([0.0, 0.0, 0.3, 0.7])  # x itself, not 2*x


def test_composed_encodings_equal_the_circuit_written_by_hand():
    """The whole point: the composition means what a reader thinks it means."""
    x, weights = np.array([0.3, 0.7]), np.array([0.25, -0.4])
    zz = qk.ZZFeatureMap(2, reps=1)
    angle = qk.AngleFeatureMap(2, rotation="ry", entangle=False)
    model = qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle))

    ref = QCircuit(2)
    for op in zz.build(x).ops:  # both maps see the raw features
        ref.apply(op.gate, op.qubits, *op.params)
    ref.ry(0, weights[0])
    ref.ry(1, weights[1])
    for op in angle.build(x).ops:
        ref.apply(op.gate, op.qubits, *op.params)

    got = expectation(model.bind(x, weights), qk.Z(0))
    assert got == pytest.approx(expectation(ref.to_spec(), qk.Z(0)), abs=1e-12)


def test_composed_angles_and_jacobian_stack_in_slot_order():
    zz = qk.ZZFeatureMap(2, reps=1)
    angle = qk.AngleFeatureMap(2, rotation="ry", entangle=False)
    model = qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle))
    x = np.array([0.3, 0.7])

    assert model.angles(x) == pytest.approx(np.concatenate([zz.angles(x), angle.angles(x)]))
    jac = model.angle_jacobian(x)
    assert jac.shape == (model.n_inputs, 2)
    assert jac == pytest.approx(np.vstack([zz.angle_jacobian(x), angle.angle_jacobian(x)]))


def test_the_same_feature_map_twice_shares_one_set_of_slots():
    """Re-uploading is the same map again, so it must not consume new features."""
    fmap = qk.AngleFeatureMap(2, rotation="ry", entangle=False)
    model = qk.Ansatz(2, qk.EncodingLayer(fmap) + qk.RotationLayer("rz") + qk.EncodingLayer(fmap))
    assert model.n_inputs == fmap.n_angles
    assert model.feature_maps == (fmap,)
    for i in range(model.n_inputs):
        assert len(model.build().occurrences_of(i)) == 2


def test_n_inputs_is_inferred_and_a_wrong_one_is_refused():
    zz, angle = qk.ZZFeatureMap(2, reps=1), qk.AngleFeatureMap(2, entangle=False)
    block = qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle)
    assert qk.Ansatz(2, block).n_inputs == 5

    # the count that used to be demanded - one map's angles - is the collision
    with pytest.raises(ValueError, match="reserve 5 slot"):
        qk.Ansatz(2, block, "mixed", n_inputs=3)
    with pytest.raises(ValueError, match="reserve 5 slot"):
        qk.Ansatz(2, block, "mixed", n_inputs=0)


def test_a_hand_built_context_still_refuses_to_overrun_its_slots():
    """``Ansatz`` infers the total; a context built directly is told what it has."""
    ctx = BuildContext(2, n_inputs=2)
    with pytest.raises(ValueError, match="need at least 3"):
        qk.EncodingLayer(qk.ZZFeatureMap(2)).emit(QCircuit(2), ctx)


def test_feature_maps_that_disagree_about_the_data_width_are_refused():
    """Every map is handed the same ``x``, so a model whose maps disagree cannot bind."""
    with pytest.raises(ValueError, match="different numbers of features"):
        qk.Ansatz(
            3,
            qk.EncodingLayer(qk.AngleFeatureMap(2)) + qk.EncodingLayer(qk.AngleFeatureMap(3)),
        )


def test_a_composed_model_is_a_drop_in_feature_map_for_quantum_layer():
    """The README promised this of any composition; only ``reupload()`` ever did it."""
    torch = pytest.importorskip("torch")
    zz = qk.ZZFeatureMap(2, reps=1)
    angle = qk.AngleFeatureMap(2, rotation="ry", entangle=False)
    model = qk.Ansatz(
        2, qk.EncodingLayer(zz) + qk.RotationLayer(("rz", "ry")) + qk.EncodingLayer(angle)
    )
    assert model.n_features == 2  # five input slots, but the data is still two wide

    layer = qk.QuantumLayer(model, observables=[qk.Z(0)])
    x = torch.tensor([[0.3, 0.7]], dtype=torch.get_default_dtype(), requires_grad=True)
    layer(x).sum().backward()

    # the chain rule runs through the *composite* jacobian, so check it against x
    weights = layer.theta.detach().numpy()
    point, eps = np.array([0.3, 0.7]), 1e-6

    def f(values):
        return expectation(model.bind(values, weights), qk.Z(0))

    numeric = [
        (f(point + step) - f(point - step)) / (2 * eps)
        for step in (np.array([eps, 0.0]), np.array([0.0, eps]))
    ]
    assert x.grad.numpy().ravel() == pytest.approx(numeric, abs=1e-5)


def test_an_ansatz_with_no_encoding_passes_its_slot_values_through():
    """No feature map means the slots *are* the angles, which is what bind() assumed."""
    plain = qk.Ansatz(2, qk.RotationLayer("ry"))
    assert plain.n_inputs == 0
    assert plain.angles([0.3, 0.7]) == pytest.approx([0.3, 0.7])
    assert plain.angle_jacobian([0.3, 0.7]) == pytest.approx(np.eye(2))


def test_reupload_validates_arguments():
    with pytest.raises(ValueError, match="n_layers must be at least 1"):
        qk.reupload(qk.AngleFeatureMap(2), n_layers=0)
    with pytest.raises(ValueError, match="unknown order 'sideways'"):
        qk.reupload(qk.AngleFeatureMap(2), order="sideways")


def test_reupload_warns_when_the_block_commutes_with_the_encoding():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qk.reupload(qk.AngleFeatureMap(1, entangle=False), n_layers=3, rotations=("ry",))
    assert caught and "commutes" in str(caught[0].message)


def test_reupload_reaches_the_frequencies_it_claims():
    for layers in (2, 3):
        model = qk.reupload(qk.AngleFeatureMap(1, entangle=False), n_layers=layers, entangler=None)
        theta = np.random.default_rng(0).uniform(-np.pi, np.pi, model.n_weights)
        spec = model.build()

        def f(x, spec=spec, theta=theta, model=model):
            return qk.expval(spec, qk.Z(0), theta=np.concatenate([model.angles([x]), theta]))

        present = set(spectrum(f, layers + 3))
        assert present <= set(range(layers + 1))
        assert layers in present
        assert model.n_frequencies == layers + 1


def test_reupload_model_drops_into_a_torch_layer():
    model = qk.reupload(qk.AngleFeatureMap(2), n_layers=2)
    layer = qk.QuantumLayer(model, observables=[qk.Z(0)], init_seed=0).double()
    x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
    assert layer(x).shape == (2, 1)
    assert torch.autograd.gradcheck(lambda t: layer(t), (x,), eps=1e-6, atol=1e-6)


def test_layer_refuses_a_redundant_ansatz():
    model = qk.reupload(qk.AngleFeatureMap(2), n_layers=2)
    with pytest.raises(ValueError, match="already contains its trainable block"):
        qk.QuantumLayer(model, qk.hardware_efficient(2, 1))


# --------------------------------------------------------------------------- #
# structured layers
# --------------------------------------------------------------------------- #
def test_qcnn_layer_runs_and_ties_its_filter():
    layer = qk.QCNNLayer(4, init_seed=0).double()
    assert layer(torch.randn(3, 4, dtype=torch.float64)).shape == (3, 1)
    assert (
        qk.QCNNLayer(8, tie_weights=True).n_weights < qk.QCNNLayer(8, tie_weights=False).n_weights
    )


def test_qcnn_is_shallower_than_a_deep_hardware_efficient_ansatz():
    assert qk.qcnn_ansatz(8).resources()["depth"] < qk.hardware_efficient(8, 8).resources()["depth"]


def test_mps_layer_runs():
    layer = qk.MPSLayer(4, init_seed=0).double()
    assert layer(torch.randn(2, 4, dtype=torch.float64)).shape == (2, 1)


def test_qlstm_cell_has_four_quantum_gates_and_classical_recurrence():
    cell = qk.QLSTMCell(3, hidden_size=2, n_qubits=3).double()
    assert set(cell.gates) == {"forget", "input", "candidate", "output"}
    h, c = cell(torch.randn(2, 3, dtype=torch.float64))
    assert h.shape == (2, 2) and c.shape == (2, 2)
    h2, _ = cell(torch.randn(2, 3, dtype=torch.float64), (h, c))
    assert h2.shape == (2, 2)


def test_qlstm_processes_a_sequence_and_trains():
    model = qk.QLSTM(2, hidden_size=2, n_qubits=2).double()
    x = torch.randn(2, 3, 2, dtype=torch.float64)
    outputs, (h, c) = model(x)
    assert outputs.shape == (2, 3, 2)
    outputs.sum().backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.parameters())
    with pytest.raises(ValueError, match=r"expected \(batch, time, features\)"):
        model(torch.randn(2, 3, dtype=torch.float64))


def test_dressed_net_freezes_the_backbone_but_trains_the_pre_net():
    backbone = nn.Linear(8, 5).double()
    net = qk.DressedQuantumNet(backbone, 5, 3, 2, init_seed=0).double()
    net(torch.randn(3, 8, dtype=torch.float64)).sum().backward()
    assert backbone.weight.grad is None
    assert net.frozen_parameters == sum(p.numel() for p in backbone.parameters())
    assert float(net.pre.weight.grad.norm()) > 1e-9


# --------------------------------------------------------------------------- #
# generative losses
# --------------------------------------------------------------------------- #
def test_mmd_is_zero_for_identical_samples_and_positive_otherwise():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(30, 2))
    b = rng.normal(loc=3.0, size=(30, 2))
    assert mmd_squared(a, a) == pytest.approx(0.0, abs=1e-12)
    assert mmd_squared(a, b) > 0.1


def test_mmd_accepts_several_bandwidths():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(20, 2)), rng.normal(size=(20, 2))
    assert np.isfinite(mmd_squared(a, b, gamma=(0.25, 1.0, 4.0)))


def test_gaussian_kernel_shape_and_validation():
    assert gaussian_kernel(np.zeros((3, 2)), np.zeros((4, 2))).shape == (3, 4)
    with pytest.raises(ValueError, match="sample widths differ"):
        gaussian_kernel(np.zeros((3, 2)), np.zeros((3, 5)))


def test_distribution_distances():
    p = np.array([0.5, 0.5])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert total_variation(p, p) == pytest.approx(0.0, abs=1e-12)
    assert total_variation(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Born machines
# --------------------------------------------------------------------------- #
def test_qcbm_samples_have_the_right_shape_and_probabilities_sum_to_one():
    model = QCBM(3, n_layers=2, seed=0)
    assert model.probabilities().sum() == pytest.approx(1.0)
    samples = model.sample(64, seed=0)
    assert samples.shape == (64, 3)
    assert set(np.unique(samples)) <= {0, 1}


def test_qcbm_training_moves_its_distribution_toward_the_target():
    """The circuit is the distribution; MMD is the only loss it admits.

    Progress is measured on the **exact** distribution, not a sampled MMD. A sampled
    before/after comparison is two noisy estimates, and can go the wrong way by chance
    even when training worked — which made an earlier version of this test flaky.
    """
    target = qk.datasets.bars_and_stripes(2)
    model = QCBM(4, n_layers=3, seed=0)
    before = model.exact_distance(target)
    model.fit(target, n_iterations=60, n_samples=256, seed=0)
    assert model.exact_distance(target) < before


def test_qcbm_score_is_reproducible_when_seeded():
    target = qk.datasets.bars_and_stripes(2)
    model = QCBM(4, n_layers=2, seed=0)
    assert model.score(target, seed=7) == model.score(target, seed=7)


def test_qcbm_exact_distance_validates_its_metric():
    model = QCBM(2, n_layers=1, seed=0)
    data = np.array([[0, 0], [1, 1]])
    assert 0.0 <= model.exact_distance(data, "tv") <= 1.0
    assert model.exact_distance(data, "kl") >= 0.0
    with pytest.raises(ValueError, match="unknown metric"):
        model.exact_distance(data, "vibes")


def test_qgan_generator_loss_and_equilibrium():
    gen = QCBM(3, n_layers=2, seed=0)

    def discriminator(batch):
        return np.full(len(batch), 0.5)  # always guessing

    gan = QGAN(gen, discriminator, seed=0)
    assert np.isfinite(gan.generator_loss(gen.params_, n_samples=64))
    gap = gan.equilibrium_gap(qk.datasets.bars_and_stripes(2), n_samples=64)
    assert gap == pytest.approx(0.5, abs=0.51)  # a coin-flip discriminator


def test_qgan_trains_the_generator():
    gen = QCBM(3, n_layers=2, seed=0)
    gan = QGAN(gen, lambda b: np.full(len(b), 0.6), seed=0)
    gan.fit_generator(n_iterations=10, n_samples=64, seed=0)
    assert len(gan.history_) == 11


# --------------------------------------------------------------------------- #
# energy-based models
# --------------------------------------------------------------------------- #
def test_boltzmann_normalises_and_favours_low_energy():
    p, z = boltzmann(np.array([0.0, 1.0, 2.0]))
    assert p.sum() == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]
    assert z > 0
    assert partition_function(np.zeros(4)) == pytest.approx(4.0)


def test_temperature_sharpens_or_flattens():
    e = np.array([0.0, 1.0, 2.0])
    cold, _ = boltzmann(e, beta=5.0)
    hot, _ = boltzmann(e, beta=0.1)
    assert cold.max() > hot.max()


def test_ising_energy_matches_the_formula():
    fields = np.array([0.5, -0.3])
    couplings = {(0, 1): 0.8}
    assert ising_energy([1, -1], fields, couplings) == pytest.approx(
        -(0.5 * 1 + (-0.3) * (-1)) - 0.8 * 1 * (-1)
    )


def test_qbm_marginal_is_a_distribution():
    qbm = QuantumBoltzmannMachine(2, 1, seed=0)
    marginal = qbm.visible_marginal()
    assert marginal.shape == (4,)
    assert marginal.sum() == pytest.approx(1.0)
    assert np.all(marginal >= 0)


def test_qbm_gradient_is_data_minus_model():
    g = QuantumBoltzmannMachine.grad(np.array([0.8, 0.2]), np.array([0.5, 0.5]))
    assert np.allclose(g, [0.3, -0.3])


def test_hopfield_recalls_the_nearest_pattern():
    memory = QuantumHopfield().store({"A": [1, 0, 0, 1], "B": [1, 1, 0, 0], "C": [0, 1, 1, 0]})
    assert memory.recall([1.0, 0.2, 0.0, 0.9]) == "A"
    assert memory.recall([0.9, 1.0, 0.1, 0.0]) == "B"
    overlaps = memory.overlaps([1, 0, 0, 1])
    assert overlaps["A"] == pytest.approx(1.0, abs=1e-9)


def test_hopfield_swap_readout_round_trip():
    for overlap in (0.0, 0.5, 1.0):
        p0 = QuantumHopfield.swap_probability(overlap)
        assert QuantumHopfield.overlap_from_probability(p0) == pytest.approx(overlap)


def test_hopfield_rejects_degenerate_input():
    with pytest.raises(ValueError, match="zero vector"):
        QuantumHopfield().store({"A": [0, 0]})
    with pytest.raises(ValueError, match="no patterns stored"):
        QuantumHopfield().overlaps([1, 0])
    with pytest.raises(ValueError, match="cue is the zero vector"):
        QuantumHopfield().store({"A": [1, 0]}).overlaps([0, 0])
