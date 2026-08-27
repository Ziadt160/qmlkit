"""The classical baseline table.

Two properties carry the module. Every row must be scored on the *same* folds, or
the comparison measures splits rather than models; and the verdict must not call a
lead a result when the lead is smaller than the fold-to-fold spread. Both are
asserted here directly.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.baselines import (
    BaselineRow,
    BaselineTable,
    LinearLeastSquares,
    MajorityClassifier,
    NearestCentroid,
    RBFKernelRidge,
    baseline,
)


@pytest.fixture
def moons():
    return qk.datasets.make_moons(n_samples=90, seed=0)


# --------------------------------------------------------------------------- #
# the NumPy-only estimators
# --------------------------------------------------------------------------- #
def test_majority_predicts_the_most_frequent_label():
    y = np.array([0] * 8 + [1] * 2)
    fitted = MajorityClassifier().fit(np.zeros((10, 2)), y)
    assert set(fitted.predict(np.zeros((4, 2))).tolist()) == {0}


def test_nearest_centroid_separates_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-2, 0.3, (20, 2)), rng.normal(2, 0.3, (20, 2))])
    y = np.array([0] * 20 + [1] * 20)
    assert (NearestCentroid().fit(X, y).predict(X) == y).mean() == 1.0


def test_rbf_kernel_ridge_matches_sklearn_on_the_same_kernel(moons):
    """The classification foil is the same algorithm scikit-learn would run."""
    kr = pytest.importorskip("sklearn.kernel_ridge")
    X, y = moons
    onehot = np.eye(2)[y]
    ours = RBFKernelRidge(alpha=1.0).fit(X, y)
    theirs = kr.KernelRidge(alpha=1.0, kernel="rbf", gamma=ours.gamma_).fit(X, onehot)
    np.testing.assert_allclose(
        ours.predict(X), theirs.predict(X).argmax(axis=1), atol=0, rtol=0
    )


def test_rbf_gamma_follows_sklearns_scale_convention(moons):
    X, _ = moons
    fitted = RBFKernelRidge().fit(X, np.zeros(len(X), dtype=int))
    assert fitted.gamma_ == pytest.approx(1.0 / (X.shape[1] * X.var()))


def test_linear_least_squares_recovers_a_linear_target():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.5, -2.0, 0.25]) + 4.0
    assert LinearLeastSquares().fit(X, y).predict(X) == pytest.approx(y, abs=1e-6)


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_classification_table_ranks_and_names_the_bar(moons):
    X, y = moons
    table = baseline(X, y, cv=3, seed=0)
    assert table.task == "classification"
    assert table.metric == "balanced_accuracy"
    names = {row.name for row in table.ran}
    assert {"majority", "nearest-centroid", "rbf-kernel-ridge"} <= names
    assert table.model_row is None
    best = table.best_classical
    assert best is not None and best.name != "majority"
    assert "bar to beat" in table.verdict


def test_majority_is_the_floor_it_claims_to_be(moons):
    X, y = moons
    table = baseline(X, y, cv=3, seed=0, include=["majority", "rbf-kernel-ridge"])
    floor = next(r for r in table.rows if r.name == "majority")
    kernel = next(r for r in table.rows if r.name == "rbf-kernel-ridge")
    assert floor.mean == pytest.approx(0.5, abs=1e-9)  # balanced accuracy of a constant
    assert kernel.mean > floor.mean


def test_every_row_is_scored_on_identical_folds(moons):
    X, y = moons
    table = baseline(X, y, cv=3, seed=0)
    folds = table.extras["folds"]
    assert len(folds) == 3
    # a second call with the same seed reproduces the same partition exactly
    again = baseline(X, y, cv=3, seed=0)
    for (train_a, test_a), (train_b, test_b) in zip(
        folds, again.extras["folds"], strict=True
    ):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_a_model_lands_in_the_same_table_and_is_judged(moons):
    X, y = moons
    table = baseline(X, y, model=NearestCentroid(), cv=3, seed=0)
    row = table.model_row
    assert row is not None and row.ran
    assert row.name == "NearestCentroid"
    assert table.beats_classical in (True, False)
    assert row.name in table.verdict


def test_a_lead_inside_the_fold_spread_is_not_called_a_result():
    table = BaselineTable(
        "classification",
        "balanced_accuracy",
        60,
        3,
        (
            BaselineRow("quantum", 0.81, 0.05, is_model=True),
            BaselineRow("svc-rbf", 0.80, 0.05),
        ),
    )
    assert table.beats_classical is True  # the mean is genuinely higher
    assert "not yet a result" in table.verdict  # but the spread swallows it


def test_a_lead_outside_the_fold_spread_is_called_a_result():
    table = BaselineTable(
        "classification",
        "balanced_accuracy",
        60,
        3,
        (
            BaselineRow("quantum", 0.95, 0.01, is_model=True),
            BaselineRow("svc-rbf", 0.80, 0.01),
        ),
    )
    assert "outside the fold spread" in table.verdict


def test_losing_is_stated_plainly():
    table = BaselineTable(
        "classification",
        "balanced_accuracy",
        60,
        3,
        (
            BaselineRow("quantum", 0.70, 0.01, is_model=True),
            BaselineRow("svc-rbf", 0.85, 0.01),
        ),
    )
    assert table.beats_classical is False
    assert "does NOT beat" in table.verdict


def test_a_model_that_raises_is_reported_not_propagated(moons):
    """The classical table is still worth having when the model under test breaks."""
    X, y = moons

    class Broken:
        def fit(self, X, y):
            raise RuntimeError("no convergence")

        def predict(self, X):  # pragma: no cover - never reached
            return np.zeros(len(X))

    table = baseline(X, y, model=Broken(), cv=3, seed=0, include=["majority"])
    row = table.model_row
    assert row is not None and not row.ran and "no convergence" in row.skipped
    assert any(r.ran for r in table.rows if not r.is_model)


def test_regression_table_infers_the_task_and_metric():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=60)
    table = baseline(X, y, cv=3, seed=0)
    assert table.task == "regression"
    assert table.metric == "r2"
    mean_row = next(r for r in table.rows if r.name == "mean")
    assert mean_row.mean < 0.1  # predicting the mean explains nothing
    assert next(r for r in table.rows if r.name == "linear").mean > 0.9


def test_integer_labels_are_classification_and_floats_are_regression():
    from qmlkit.baselines import _infer_task

    assert _infer_task(np.array([0, 1, 1, 0])) == "classification"
    assert _infer_task(np.array(["a", "b", "a"])) == "classification"
    assert _infer_task(np.linspace(0, 1, 50)) == "regression"


def test_subsampling_is_recorded_rather_than_silent():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    table = baseline(X, y, cv=3, seed=0, max_samples=60, include=["majority"])
    assert table.n_samples == 60
    assert any("subsampled" in note for note in table.notes)


def test_imbalance_switches_the_ranking_metric_and_says_so():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = np.array([0] * 100 + [1] * 20)
    table = baseline(X, y, cv=3, seed=0, include=["majority"])
    assert table.metric == "balanced_accuracy"
    assert any("imbalance_report" in note for note in table.notes)


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
def test_registering_a_baseline_puts_it_in_every_later_table(moons):
    X, y = moons
    qk.register_baseline("always-one", "classification", lambda: _AlwaysOne())
    try:
        assert "always-one" in qk.list_baselines("classification")
        table = baseline(X, y, cv=3, seed=0, include=["always-one", "majority"])
        assert "always-one" in {r.name for r in table.ran}
    finally:
        from qmlkit.baselines import _BASELINES

        _BASELINES.pop("classification/always-one", None)


class _AlwaysOne:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(len(np.atleast_2d(X)), dtype=int)


def test_the_same_name_can_exist_for_both_tasks():
    assert "rbf-kernel-ridge" in qk.list_baselines("classification")
    assert "rbf-kernel-ridge" in qk.list_baselines("regression")
    assert qk.get_baseline("rbf-kernel-ridge", "regression").task == "regression"


def test_unknown_names_suggest_a_real_one():
    with pytest.raises(KeyError, match="majority"):
        qk.get_baseline("majorty")
    with pytest.raises(ValueError, match="classification"):
        qk.register_baseline("x", "clustering", lambda: None)


def test_an_unavailable_extra_is_listed_as_skipped_not_dropped(monkeypatch, moons):
    """A table that quietly omits the strong baseline is the problem being solved."""
    module = importlib.import_module('qmlkit.baselines')

    monkeypatch.setattr(module, "_available", lambda requires: requires is None)
    X, y = moons
    table = baseline(X, y, cv=3, seed=0)
    skipped = [r for r in table.rows if not r.ran]
    assert skipped and all("pip install" in r.skipped for r in skipped)
    assert "svc-rbf" in {r.name for r in skipped}
    assert "skipped" in str(table)
