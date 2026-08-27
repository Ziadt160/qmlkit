"""Metrics, cross-validated against scikit-learn.

The library's own test suite can only catch the mistakes its author thought of, so
:mod:`qmlkit.evaluate` is checked the way the rest of qmlkit is checked: against a
second, independently written implementation. Every metric with a scikit-learn
equivalent is asserted equal to it, on **randomly generated** label sets rather
than hand-picked ones — hand-picked cases confirm what the author already believed.

The scikit-learn comparisons skip when it is not installed; the behaviour tests
(what the notes say, which metric is primary) always run, because those are the
part scikit-learn has no opinion about.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmlkit import evaluate

TOL = 1e-12


def _sk():
    """scikit-learn's metrics, or skip — it is a test-time second opinion, not a dependency."""
    return pytest.importorskip("sklearn.metrics", reason="parity needs scikit-learn")


# --------------------------------------------------------------------------- #
# classification parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_classification_matches_sklearn(seed, n_classes):
    m = _sk()
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 120))
    y_true = rng.integers(0, n_classes, size=n)
    y_pred = np.where(rng.random(n) < 0.7, y_true, rng.integers(0, n_classes, size=n))
    labels = list(range(n_classes))

    got = evaluate.classification(y_true, y_pred, labels=labels)

    assert got["accuracy"] == pytest.approx(m.accuracy_score(y_true, y_pred), abs=TOL)
    assert got["balanced_accuracy"] == pytest.approx(
        m.balanced_accuracy_score(y_true, y_pred), abs=1e-10
    )
    assert got["mcc"] == pytest.approx(m.matthews_corrcoef(y_true, y_pred), abs=1e-10)
    assert got["cohen_kappa"] == pytest.approx(m.cohen_kappa_score(y_true, y_pred), abs=1e-10)
    for average in ("macro", "weighted"):
        for name, fn in (
            ("precision", m.precision_score),
            ("recall", m.recall_score),
            ("f1", m.f1_score),
        ):
            assert got[f"{name}_{average}"] == pytest.approx(
                fn(y_true, y_pred, average=average, labels=labels, zero_division=0), abs=1e-10
            )
    np.testing.assert_array_equal(
        got.extras["confusion_matrix"], m.confusion_matrix(y_true, y_pred, labels=labels)
    )


@pytest.mark.parametrize("seed", range(8))
def test_binary_threshold_metrics_match_sklearn(seed):
    m = _sk()
    rng = np.random.default_rng(seed)
    n = int(rng.integers(40, 150))
    y_true = rng.integers(0, 2, size=n)
    # continuous scores, correlated with the truth and without ties
    score = np.clip(0.5 + 0.25 * (2 * y_true - 1) + 0.3 * rng.normal(size=n), 1e-6, 1 - 1e-6)
    proba = np.column_stack([1 - score, score])

    got = evaluate.classification(y_true, (score > 0.5).astype(int), proba, labels=[0, 1])
    assert got["roc_auc"] == pytest.approx(m.roc_auc_score(y_true, score), abs=1e-10)
    assert got["average_precision"] == pytest.approx(
        m.average_precision_score(y_true, score), abs=1e-10
    )
    assert got["log_loss"] == pytest.approx(m.log_loss(y_true, proba, labels=[0, 1]), abs=1e-10)
    assert got["brier"] == pytest.approx(m.brier_score_loss(y_true, score), abs=1e-10)


def test_roc_auc_handles_ties_exactly():
    """Every score identical means the ranking is pure chance: 0.5, not 0 or 1."""
    m = _sk()
    y = np.array([0, 0, 1, 1, 0, 1])
    tied = np.ones(6)
    assert evaluate.roc_auc(y, tied) == pytest.approx(0.5)
    assert evaluate.roc_auc(y, tied) == pytest.approx(m.roc_auc_score(y, tied))

    half = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert evaluate.roc_auc(y, half) == pytest.approx(m.roc_auc_score(y, half), abs=1e-12)


def test_multiclass_ovr_auc_matches_sklearn():
    m = _sk()
    rng = np.random.default_rng(3)
    y = rng.integers(0, 3, size=90)
    logits = rng.normal(size=(90, 3)) + np.eye(3)[y]
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    got = evaluate.classification(y, proba.argmax(axis=1), proba, labels=[0, 1, 2])
    assert got["roc_auc_ovr"] == pytest.approx(
        m.roc_auc_score(y, proba, multi_class="ovr", average="macro"), abs=1e-10
    )


# --------------------------------------------------------------------------- #
# regression parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(8))
def test_regression_matches_sklearn(seed):
    m = _sk()
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 120))
    y_true = rng.normal(loc=2.0, scale=3.0, size=n)
    y_pred = y_true + rng.normal(scale=0.8, size=n)

    got = evaluate.regression(y_true, y_pred)
    assert got["r2"] == pytest.approx(m.r2_score(y_true, y_pred), abs=1e-10)
    assert got["mse"] == pytest.approx(m.mean_squared_error(y_true, y_pred), abs=1e-10)
    assert got["mae"] == pytest.approx(m.mean_absolute_error(y_true, y_pred), abs=1e-10)
    assert got["median_absolute_error"] == pytest.approx(
        m.median_absolute_error(y_true, y_pred), abs=1e-10
    )
    assert got["max_error"] == pytest.approx(m.max_error(y_true, y_pred), abs=1e-10)
    assert got["explained_variance"] == pytest.approx(
        m.explained_variance_score(y_true, y_pred), abs=1e-10
    )
    assert got["mape"] == pytest.approx(
        m.mean_absolute_percentage_error(y_true, y_pred), abs=1e-10
    )


def test_mape_is_omitted_rather_than_infinite():
    scores = evaluate.regression([0.0, 1.0, 2.0], [0.1, 1.1, 2.1])
    assert "mape" not in scores
    assert any("mape" in note for note in scores.notes)


# --------------------------------------------------------------------------- #
# clustering parity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(5))
def test_clustering_matches_sklearn(seed):
    m = _sk()
    rng = np.random.default_rng(seed)
    centres = rng.normal(scale=4.0, size=(3, 2))
    labels = rng.integers(0, 3, size=75)
    X = centres[labels] + rng.normal(scale=0.6, size=(75, 2))
    assigned = np.where(rng.random(75) < 0.85, labels, rng.integers(0, 3, size=75))

    got = evaluate.clustering(X, assigned, y_true=labels)
    assert got["silhouette"] == pytest.approx(m.silhouette_score(X, assigned), abs=1e-10)
    assert got["davies_bouldin"] == pytest.approx(m.davies_bouldin_score(X, assigned), abs=1e-10)
    assert got["adjusted_rand"] == pytest.approx(m.adjusted_rand_score(labels, assigned), abs=1e-10)
    assert got["normalized_mutual_info"] == pytest.approx(
        m.normalized_mutual_info_score(labels, assigned), abs=1e-10
    )


def test_single_cluster_is_reported_not_raised():
    X = np.random.default_rng(0).normal(size=(20, 2))
    scores = evaluate.clustering(X, np.zeros(20, dtype=int))
    assert np.isnan(scores["silhouette"])
    assert any("one cluster" in note for note in scores.notes)


# --------------------------------------------------------------------------- #
# generative
# --------------------------------------------------------------------------- #
def test_generative_distances_have_their_defining_properties():
    rng = np.random.default_rng(0)
    p = rng.random(8)
    q = rng.random(8)
    same = evaluate.generative(p, p)
    assert same["total_variation"] == pytest.approx(0.0, abs=1e-15)
    assert same["js_distance"] == pytest.approx(0.0, abs=1e-8)
    assert same["hellinger"] == pytest.approx(0.0, abs=1e-15)

    both = evaluate.generative(p, q)
    assert both["total_variation"] == pytest.approx(evaluate.generative(q, p)["total_variation"])
    assert 0.0 <= both["total_variation"] <= 1.0
    # Pinsker: TV <= sqrt(KL/2), whenever the KL is finite
    assert both["total_variation"] <= np.sqrt(both["kl_model_target"] / 2) + 1e-12


def test_generative_accepts_counts_dicts_and_flags_missed_support():
    model = {"00": 50, "11": 50}
    target = {"00": 40, "01": 20, "11": 40}
    scores = evaluate.generative(model, target)
    assert np.isinf(scores["kl_target_model"])
    assert np.isfinite(scores["total_variation"])
    assert any("zero mass" in note for note in scores.notes)
    assert scores["support_coverage"] == pytest.approx(2 / 3)


def test_generative_rejects_mixed_types():
    with pytest.raises(TypeError, match="both distributions"):
        evaluate.generative({"0": 1.0}, np.array([0.5, 0.5]))


# --------------------------------------------------------------------------- #
# the behaviour scikit-learn has no opinion about
# --------------------------------------------------------------------------- #
def test_accuracy_at_the_majority_rate_is_called_out():
    y = np.array([0] * 95 + [1] * 5)
    scores = evaluate.classification(y, np.zeros(100, dtype=int))
    assert scores["accuracy"] == pytest.approx(0.95)
    assert scores["balanced_accuracy"] == pytest.approx(0.5)
    assert scores.primary == "balanced_accuracy"
    assert scores.score == pytest.approx(0.5)
    assert any("majority-class rate" in note for note in scores.notes)


def test_balanced_data_keeps_accuracy_primary_and_stays_quiet():
    y = np.array([0, 1] * 50)
    pred = y.copy()
    pred[:5] = 1 - pred[:5]
    scores = evaluate.classification(y, pred)
    assert scores.primary == "accuracy"
    assert not any("majority-class" in note for note in scores.notes)


def test_scores_behaves_like_a_mapping():
    scores = evaluate.regression([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert "r2" in scores
    assert set(dict(scores)) == set(scores.keys())
    assert len(scores) == len(scores.values)
    assert "r2" in str(scores)


def test_unknown_metric_suggests_the_right_name():
    scores = evaluate.regression([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(KeyError, match="Did you mean 'r2'"):
        scores["R2"]


def test_length_mismatch_is_an_error_not_a_broadcast():
    with pytest.raises(ValueError, match="length mismatch"):
        evaluate.classification([0, 1, 0], [0, 1])


def test_scores_for_dispatches_and_suggests():
    got = evaluate.scores_for("regression", [1.0, 2.0], [1.0, 2.0])
    assert got.task == "regression"
    with pytest.raises(ValueError, match="classification"):
        evaluate.scores_for("classifcation", [0, 1], [0, 1])
