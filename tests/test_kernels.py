"""Quantum kernels: estimators, Gram matrices, PSD repair, and kernel models."""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.info import state_fidelity
from qmlkit.kernels import (
    QSVC,
    QSVR,
    NearestFidelityClassifier,
    QuantumKernel,
    TrainableKernel,
    center_kernel,
    closest_psd_matrix,
    concentration_report,
    displace_matrix,
    fidelity_kernel,
    flip_matrix,
    geometric_difference,
    hadamard_test,
    is_psd,
    kernel_matrix,
    kernel_shot_cost,
    min_eigenvalue,
    normalize_kernel,
    projected_kernel_matrix,
    rkhs_model,
    square_kernel_matrix,
    swap_probability,
    swap_readout,
    swap_test_kernel,
    target_alignment,
    threshold_matrix,
)

sklearn = pytest.importorskip("sklearn")


def _blobs(seed: int = 1, n: int = 16):
    rng = np.random.default_rng(seed)
    a = rng.normal([0.9, 0.9], 0.25, size=(n // 2, 2))
    b = rng.normal([2.2, 2.2], 0.25, size=(n // 2, 2))
    X = np.clip(np.vstack([a, b]), 0, np.pi)
    return X, np.array([0] * (n // 2) + [1] * (n // 2))


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fmap",
    [
        qk.AngleFeatureMap(1, entangle=False),
        qk.AngleFeatureMap(2),
        qk.ZZFeatureMap(2, reps=1),
        qk.PauliFeatureMap(2, paulis=("Z", "X", "ZZ"), reps=1),
    ],
)
def test_inversion_and_swap_estimators_match_the_exact_overlap(fmap):
    rng = np.random.default_rng(0)
    x = rng.uniform(0, np.pi, fmap.n_features)
    xp = rng.uniform(0, np.pi, fmap.n_features)
    exact = state_fidelity(fmap.build(x), fmap.build(xp))
    assert fidelity_kernel(fmap, x, xp) == pytest.approx(exact, abs=1e-10)
    assert swap_test_kernel(fmap, x, xp) == pytest.approx(exact, abs=1e-9)


def test_one_qubit_angle_kernel_is_cos_squared():
    """The analytic result the whole of Lecture 4 is built on."""
    fm = qk.AngleFeatureMap(1, entangle=False)
    for dx in (0.0, np.pi / 2, np.pi, 1.3):
        assert fidelity_kernel(fm, [0.0], [dx]) == pytest.approx(np.cos(dx / 2) ** 2, abs=1e-10)


def test_self_kernel_is_one():
    fm = qk.ZZFeatureMap(2, reps=2)
    assert fidelity_kernel(fm, [0.4, 1.2], [0.4, 1.2]) == pytest.approx(1.0, abs=1e-10)


def test_hadamard_test_keeps_the_sign_magnitude_estimators_lose():
    """+1/2 and -1/2 map to the same magnitude; only the Hadamard test tells them apart."""
    fm = qk.AngleFeatureMap(1, entangle=False)
    plus = hadamard_test(fm, [0.0], [np.pi / 2])
    minus = hadamard_test(fm, [0.0], [3 * np.pi / 2])
    assert plus == pytest.approx(np.cos(np.pi / 4), abs=1e-9)
    assert minus == pytest.approx(np.cos(3 * np.pi / 4), abs=1e-9)
    assert plus > 0 > minus
    # ...whereas the magnitude estimator collapses them
    assert fidelity_kernel(fm, [0.0], [np.pi / 2]) == pytest.approx(
        fidelity_kernel(fm, [0.0], [3 * np.pi / 2]), abs=1e-9
    )


def test_hadamard_test_rejects_a_bad_part():
    with pytest.raises(ValueError, match="unknown part 'sideways'"):
        hadamard_test(qk.AngleFeatureMap(1, entangle=False), [0.1], [0.2], part="sideways")


def test_swap_readout_round_trip():
    for k in (-1.0, 0.0, 0.5, 1.0):
        assert swap_readout(swap_probability(k)) == pytest.approx(k)


def test_sampled_kernel_converges_to_exact():
    fm = qk.AngleFeatureMap(2)
    x, xp = np.array([0.4, 1.2]), np.array([1.9, 0.6])
    exact = fidelity_kernel(fm, x, xp)
    assert fidelity_kernel(fm, x, xp, shots=200_000, seed=0) == pytest.approx(exact, abs=0.01)


# --------------------------------------------------------------------------- #
# Gram matrices
# --------------------------------------------------------------------------- #
def test_gram_matrix_is_symmetric_unit_diagonal_and_psd():
    X, _ = _blobs()
    K = QuantumKernel(qk.ZZFeatureMap(2, reps=1))(X)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert is_psd(K)


def test_square_matrix_uses_only_the_upper_triangle():
    X, _ = _blobs(n=10)
    kernel = QuantumKernel(qk.AngleFeatureMap(2), cache=False)
    square_kernel_matrix(X, kernel.evaluate)
    assert kernel.n_evaluations == 10 * 9 // 2  # diagonal is free


def test_symmetric_cache_reuses_reversed_pairs():
    X, _ = _blobs(n=8)
    kernel = QuantumKernel(qk.AngleFeatureMap(2))
    kernel(X)
    before = kernel.n_evaluations
    kernel(X, X)  # every pair already seen, in one order or the other
    assert kernel.n_evaluations - before <= len(X)


def test_rectangular_matrix_shape():
    X, _ = _blobs(n=8)
    Y, _ = _blobs(seed=2, n=6)
    K = QuantumKernel(qk.AngleFeatureMap(2))(Y, X)
    assert K.shape == (6, 8)


def test_kernel_matrix_requires_a_kernel():
    with pytest.raises(ValueError, match="needs a kernel"):
        kernel_matrix(np.zeros((2, 2)))


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="unknown estimator"):
        QuantumKernel(qk.AngleFeatureMap(1, entangle=False), estimator="telepathy")(
            np.array([[0.1], [0.2]])
        )


def test_bandwidth_changes_the_kernel_spread():
    X, _ = _blobs(n=10)
    tight = QuantumKernel(qk.AngleFeatureMap(2), bandwidth=0.2)(X)
    wide = QuantumKernel(qk.AngleFeatureMap(2), bandwidth=2.0)(X)
    off = lambda M: M[~np.eye(len(M), dtype=bool)]  # noqa: E731
    assert off(tight).mean() > off(wide).mean()


def test_kernel_shot_cost():
    assert kernel_shot_cost(10, 1000) == 45 * 1000
    assert kernel_shot_cost(10, 1000, include_diagonal=True) == 55 * 1000


# --------------------------------------------------------------------------- #
# PSD repair
# --------------------------------------------------------------------------- #
def test_shot_noise_breaks_psd_and_repair_restores_it():
    """Every entry is an estimate, so the estimated Gram matrix can leave the cone."""
    X, _ = _blobs(n=14)
    noisy = QuantumKernel(qk.ZZFeatureMap(2, reps=1), shots=100, seed=0)(X)
    assert not is_psd(noisy), "expected shot noise to push an eigenvalue negative"
    for method in ("threshold", "displace", "flip"):
        assert is_psd(closest_psd_matrix(noisy, method))


def test_repair_methods_do_what_they_say():
    K = np.array([[1.0, 0.9], [0.9, -0.5]])
    assert min_eigenvalue(threshold_matrix(K)) >= -1e-12
    assert min_eigenvalue(displace_matrix(K)) >= -1e-12
    assert np.allclose(np.abs(np.linalg.eigvalsh(K)), np.sort(np.linalg.eigvalsh(flip_matrix(K))))
    assert np.allclose(displace_matrix(np.eye(2)), np.eye(2))  # already PSD: unchanged


def test_unknown_repair_method_is_rejected():
    with pytest.raises(ValueError, match="unknown method"):
        closest_psd_matrix(np.eye(2), "magic")


def test_center_and_normalize():
    X, _ = _blobs(n=8)
    K = QuantumKernel(qk.AngleFeatureMap(2))(X)
    assert abs(center_kernel(K).sum()) < 1e-8
    assert np.allclose(np.diag(normalize_kernel(K * 4.0)), 1.0)


# --------------------------------------------------------------------------- #
# is the kernel any good?
# --------------------------------------------------------------------------- #
def test_target_alignment_is_maximal_for_a_perfect_kernel():
    y = np.array([0, 0, 1, 1])
    perfect = np.outer(2 * y - 1, 2 * y - 1).astype(float)
    assert target_alignment(perfect, y) == pytest.approx(1.0)
    assert abs(target_alignment(np.eye(4), y)) < 1.0


def test_alignment_prefers_the_kernel_that_separates():
    X, y = _blobs(n=16)
    good = target_alignment(QuantumKernel(qk.ZZFeatureMap(2, reps=1))(X), y)
    scrambled = target_alignment(
        QuantumKernel(qk.ZZFeatureMap(2, reps=1))(X), np.random.default_rng(0).permutation(y)
    )
    assert good > scrambled


def test_concentration_report_flags_a_wide_feature_map():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (8, 6))
    K = QuantumKernel(qk.ZZFeatureMap(6, reps=2))(X)
    rep = concentration_report(K, n_qubits=6, shots=1000)
    assert rep["off_diagonal_std"] < 0.2
    assert rep["shots_to_resolve"] == 4**6
    assert set(rep) >= {"off_diagonal_mean", "is_psd", "resolvable", "min_eigenvalue"}


def test_projected_kernel_resists_concentration():
    """The whole point: local reduced states stay informative where global fidelity dies."""
    rng = np.random.default_rng(0)
    n = 8
    X = rng.uniform(0, np.pi, (8, n))
    fm = qk.ZZFeatureMap(n, reps=2)
    off = lambda M: M[~np.eye(len(M), dtype=bool)].std()  # noqa: E731
    assert off(projected_kernel_matrix(fm, X)) > off(QuantumKernel(fm)(X))


def test_geometric_difference_is_small_for_identical_kernels():
    X, _ = _blobs(n=8)
    K = QuantumKernel(qk.AngleFeatureMap(2))(X)
    same = geometric_difference(K, K)
    other = geometric_difference(K, np.eye(len(X)))
    assert same < other


def test_geometric_difference_rejects_a_shape_mismatch():
    with pytest.raises(ValueError, match="different shapes"):
        geometric_difference(np.eye(3), np.eye(4))


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #
def test_qsvc_separates_a_dataset_built_to_be_quantum_separable():
    X, y = qk.datasets.ad_hoc_data(n_samples=30, n_features=2, gap=0.4, seed=0)
    clf = QSVC(qk.ZZFeatureMap(2, reps=2)).fit(X, y)
    assert clf.score(X, y) > 0.9


def test_qsvc_fills_only_the_upper_triangle():
    X, y = _blobs(n=12)
    clf = QSVC(qk.AngleFeatureMap(2)).fit(X, y)
    assert clf.n_circuit_evaluations == 12 * 11 // 2


def test_qsvr_fits_a_smooth_target():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (24, 1))
    y = np.sin(X[:, 0])
    reg = QSVR(qk.AngleFeatureMap(1, entangle=False), C=10.0).fit(X, y)
    assert reg.score(X, y) > 0.5


def test_estimators_refuse_to_predict_before_fitting():
    clf = QSVC(qk.AngleFeatureMap(2))
    with pytest.raises(ValueError, match="must be fitted"):
        clf.predict(np.zeros((1, 2)))


def test_decision_function_is_available():
    X, y = _blobs(n=10)
    clf = QSVC(qk.AngleFeatureMap(2)).fit(X, y)
    assert clf.decision_function(X).shape == (10,)


def test_nearest_fidelity_classifier_needs_no_solver():
    X, y = _blobs(n=16)
    clf = NearestFidelityClassifier(qk.ZZFeatureMap(2, reps=1)).fit(X, y)
    assert clf.score(X, y) > 0.7
    with pytest.raises(ValueError, match="must be fitted"):
        NearestFidelityClassifier(qk.AngleFeatureMap(2)).predict(X)


def test_trainable_kernel_improves_alignment():
    """Train the embedding itself, before fitting any classifier."""
    X, y = _blobs(n=12)

    def factory(params):
        return qk.AngleFeatureMap(2, reps=1, entangle=True)

    def scaled_factory(params):
        class Scaled(qk.AngleFeatureMap):
            def angles(self, x):
                return np.asarray(params, dtype=float) * super().angles(x)

        return Scaled(2, entangle=True)

    trainer = TrainableKernel(scaled_factory, n_params=2)
    start = trainer.alignment(np.ones(2), X, y)
    trainer.fit(X, y, n_iterations=15, theta0=np.ones(2), seed=0)
    assert max(trainer.history_) >= start - 1e-9
    assert trainer.kernel() is not None


def test_trainable_kernel_refuses_before_fitting():
    with pytest.raises(ValueError, match="must be fitted"):
        TrainableKernel(lambda p: qk.AngleFeatureMap(2), 2).kernel()


def test_rkhs_model_is_a_weighted_similarity_sum():
    fm = qk.AngleFeatureMap(1, entangle=False)
    kernel = QuantumKernel(fm)
    anchors = np.array([[0.0], [np.pi]])
    value = rkhs_model([1.0, -1.0], anchors, [np.pi / 3], kernel.evaluate)
    expected = np.cos(np.pi / 6) ** 2 - np.cos((np.pi - np.pi / 3) / 2) ** 2
    assert value == pytest.approx(expected, abs=1e-9)
