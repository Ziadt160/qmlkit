"""Class weighting, resampling and splitting on skewed labels.

The weights are checked against scikit-learn's, because they are meant to be
interchangeable with ``class_weight="balanced"`` there — a caller who switches
between the two should not see the loss scale change. The splitting guarantees are
checked directly: the property that matters is that a rare class cannot vanish from
a fold, and that is a statement about indices rather than about numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from qmlkit import imbalance


def _skewed(n_majority: int = 90, n_minority: int = 10) -> np.ndarray:
    return np.array([0] * n_majority + [1] * n_minority)


# --------------------------------------------------------------------------- #
# weights
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(6))
def test_balanced_weights_match_sklearn(seed):
    sk = pytest.importorskip("sklearn.utils.class_weight")
    rng = np.random.default_rng(seed)
    y = rng.choice([0, 1, 2], size=200, p=[0.7, 0.2, 0.1])
    classes = np.unique(y)
    expected = sk.compute_class_weight("balanced", classes=classes, y=y)
    got = imbalance.class_weights(y, "balanced")
    for label, want in zip(classes.tolist(), expected.tolist(), strict=True):
        assert got[label] == pytest.approx(want, abs=1e-12)


def test_balanced_weights_average_to_one_over_the_data():
    """The loss keeps its scale, so the learning rate does not need retuning."""
    y = _skewed()
    weights = imbalance.sample_weights(y, "balanced")
    assert weights.mean() == pytest.approx(1.0, abs=1e-12)


def test_pos_weight_is_the_negative_to_positive_ratio():
    assert imbalance.pos_weight(_skewed(90, 10)) == pytest.approx(9.0)
    assert imbalance.pos_weight(np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_pos_weight_refuses_multiclass_and_says_what_to_use():
    with pytest.raises(ValueError, match="class_weights"):
        imbalance.pos_weight(np.array([0, 1, 2, 2]))


def test_none_scheme_is_a_no_op():
    assert set(imbalance.class_weights(_skewed(), "none").values()) == {1.0}


def test_unknown_scheme_suggests_a_real_one():
    with pytest.raises(ValueError, match="balanced"):
        imbalance.class_weights(_skewed(), "balence")


def test_imbalance_ratio():
    assert imbalance.imbalance_ratio(_skewed(90, 10)) == pytest.approx(9.0)
    assert imbalance.imbalance_ratio(np.array([0, 1])) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# resampling
# --------------------------------------------------------------------------- #
def test_oversampling_balances_without_discarding_anything():
    y = _skewed(90, 10)
    index = imbalance.resample(y, "oversample", seed=0)
    counts = imbalance.class_counts(y[index])
    assert counts[0] == counts[1] == 90
    # every original row still appears
    assert set(np.unique(index).tolist()) == set(range(100))


def test_undersampling_balances_by_dropping_majority_rows():
    y = _skewed(90, 10)
    index = imbalance.resample(y, "undersample", seed=0)
    counts = imbalance.class_counts(y[index])
    assert counts[0] == counts[1] == 10
    assert index.size == 20


def test_partial_ratio_closes_half_the_gap():
    y = _skewed(90, 10)
    counts = imbalance.class_counts(y[imbalance.resample(y, "oversample", seed=0, ratio=0.5)])
    assert counts[1] == 50  # 10 + 0.5 * (90 - 10)


def test_resampling_is_reproducible_and_ratio_is_validated():
    y = _skewed()
    a = imbalance.resample(y, "oversample", seed=7)
    b = imbalance.resample(y, "oversample", seed=7)
    np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        imbalance.resample(y, "oversample", ratio=1.5)


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #
def test_stratified_split_keeps_the_minority_class_in_test():
    """The whole point: a random split loses this class about a third of the time."""
    y = _skewed(95, 5)
    for seed in range(20):
        _, test = imbalance.stratified_split(y, test_size=0.2, seed=seed)
        assert (y[test] == 1).sum() >= 1


def test_stratified_split_partitions_exactly():
    y = _skewed()
    train, test = imbalance.stratified_split(y, test_size=0.3, seed=0)
    assert train.size + test.size == y.size
    assert not set(train.tolist()) & set(test.tolist())


def test_stratified_folds_partition_and_carry_every_class():
    y = np.array([0] * 60 + [1] * 25 + [2] * 15)
    folds = imbalance.stratified_folds(y, n_folds=5, seed=0)
    assert len(folds) == 5
    seen: list[int] = []
    for train, test in folds:
        assert train.size + test.size == y.size
        assert not set(train.tolist()) & set(test.tolist())
        assert set(np.unique(y[test]).tolist()) == {0, 1, 2}
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(y.size))  # every row tested exactly once


def test_too_few_members_for_the_fold_count_raises_with_the_number():
    y = np.array([0] * 50 + [1] * 3)
    with pytest.raises(ValueError, match="3 sample"):
        imbalance.stratified_folds(y, n_folds=5)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def test_report_is_falsy_on_balanced_labels():
    assert not imbalance.imbalance_report(np.array([0, 1] * 50))


def test_severe_skew_is_an_error_with_the_remedy_in_the_fix():
    report = imbalance.imbalance_report(_skewed(95, 5))
    assert "imbalance.severe" in report.codes
    finding = next(f for f in report if f.code == "imbalance.severe")
    assert finding.severity == "error"
    assert "class_weight" in finding.fix


def test_moderate_skew_warns_rather_than_errors():
    report = imbalance.imbalance_report(np.array([0] * 70 + [1] * 30))
    assert "imbalance.skewed" in report.codes
    assert "imbalance.severe" not in report.codes


def test_single_class_is_caught_first():
    report = imbalance.imbalance_report(np.zeros(20, dtype=int))
    assert report.codes == ("imbalance.single-class",)


def test_report_flags_a_class_too_small_for_the_fold_count():
    report = imbalance.imbalance_report(np.array([0] * 50 + [1] * 3), n_folds=5)
    assert "imbalance.too-few-for-cv" in report.codes


# --------------------------------------------------------------------------- #
# the torch losses
# --------------------------------------------------------------------------- #
def test_class_weight_tensor_covers_every_output_class():
    torch = pytest.importorskip("torch")
    from qmlkit.nn.losses import class_weight_tensor

    weights = class_weight_tensor(np.array([0, 0, 0, 1]), n_classes=3)
    assert weights.shape == (3,)
    assert float(weights[2]) == 1.0  # absent class defaults rather than vanishing
    assert isinstance(weights, torch.Tensor)


def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero():
    torch = pytest.importorskip("torch")
    from qmlkit.nn.losses import FocalLoss

    logits = torch.tensor([[2.0, -1.0], [0.3, 0.7], [-1.5, 2.2]])
    target = torch.tensor([0, 1, 1])
    focal = FocalLoss(gamma=0.0)(logits, target)
    plain = torch.nn.CrossEntropyLoss()(logits, target)
    assert float(focal) == pytest.approx(float(plain), abs=1e-6)


def test_focal_loss_downweights_the_confident_example():
    torch = pytest.importorskip("torch")
    from qmlkit.nn.losses import FocalLoss

    easy = torch.tensor([[6.0, -6.0]])
    hard = torch.tensor([[0.1, -0.1]])
    target = torch.tensor([0])
    loss = FocalLoss(gamma=2.0, reduction="none")
    plain = torch.nn.CrossEntropyLoss(reduction="none")
    # the focal factor shrinks the easy example far more than the hard one
    easy_ratio = float(loss(easy, target)) / float(plain(easy, target))
    hard_ratio = float(loss(hard, target)) / float(plain(hard, target))
    assert easy_ratio < hard_ratio < 1.0


def test_focal_loss_validates_its_arguments():
    pytest.importorskip("torch")
    from qmlkit.nn.losses import FocalLoss

    with pytest.raises(ValueError, match="non-negative"):
        FocalLoss(gamma=-1.0)
    with pytest.raises(ValueError, match="reduction"):
        FocalLoss(reduction="average")


def test_vqc_accepts_class_weight_and_uses_it():
    """A constructor that takes `class_weight` and ignores it looks identical outside."""
    pytest.importorskip("torch")
    import torch

    import qmlkit as qk

    y = np.array([0] * 12 + [1] * 4)
    model = qk.VQC(n_features=2, n_classes=2, class_weight="balanced", seed=0)
    loss_fn = model._loss_fn(y)
    assert isinstance(loss_fn, torch.nn.CrossEntropyLoss)
    assert loss_fn.weight is not None
    # 3:1 skew, so the minority class carries twice the mean weight
    assert float(loss_fn.weight[1]) == pytest.approx(2.0, abs=1e-6)
    assert float(loss_fn.weight[0]) == pytest.approx(2 / 3, abs=1e-6)


def test_vqc_focal_gamma_selects_the_focal_loss():
    pytest.importorskip("torch")
    import qmlkit as qk
    from qmlkit.nn.losses import FocalLoss

    model = qk.VQC(n_features=2, n_classes=2, focal_gamma=2.0, seed=0)
    assert isinstance(model._loss_fn(np.array([0, 1, 1, 0])), FocalLoss)
