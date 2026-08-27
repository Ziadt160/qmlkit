"""Skewed classes: measuring the skew, weighting the loss, and splitting safely.

Quantum classifiers are usually trained on small datasets, because every sample
costs circuits. Small and skewed is the common case, and it breaks three things at
once:

1. **The loss.** Cross-entropy on a 95/5 split is minimised by ignoring the
   minority class. The model converges, the loss curve looks healthy, and the
   circuit has learned a constant.
2. **The split.** A random 80/20 split of 100 samples with 5 positives puts one
   positive in test *on average*, and none at all about a third of the time. The
   test score is then noise.
3. **The score.** Accuracy rewards the constant model. :mod:`qmlkit.evaluate`
   handles that end; this module handles the first two.

Everything here is NumPy and returns plain arrays or indices, so it composes with
scikit-learn, with torch, and with neither::

    >>> import numpy as np, qmlkit as qk
    >>> y = np.array([0] * 90 + [1] * 10)
    >>> qk.imbalance.pos_weight(y)
    9.0
    >>> train, test = qk.imbalance.stratified_split(y, test_size=0.2, seed=0)
    >>> int(y[test].sum())          # two positives, not "on average two"
    2

The torch losses that consume these weights live in :mod:`qmlkit.nn.losses`, so
importing this module never requires torch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "class_counts",
    "imbalance_ratio",
    "class_weights",
    "sample_weights",
    "pos_weight",
    "resample",
    "stratified_split",
    "stratified_folds",
    "imbalance_report",
]

Array = npt.NDArray[Any]

#: Above this majority share, accuracy stops being a usable training signal and
#: the remedies in this module start to matter. Two classes at 65/35 are still
#: learnable untouched; at 80/20 an unweighted loss usually collapses.
_SKEWED = 0.65


def _labels(y: Any) -> Array:
    arr = np.asarray(y).ravel()
    if arr.size == 0:
        raise ValueError("y is empty")
    return arr


def class_counts(y: Any) -> dict[Any, int]:
    """``{label: count}``, in sorted label order."""
    labels, counts = np.unique(_labels(y), return_counts=True)
    return {
        label: int(count)
        for label, count in zip(labels.tolist(), counts.tolist(), strict=True)
    }


def imbalance_ratio(y: Any) -> float:
    """Majority count over minority count. ``1.0`` is perfectly balanced."""
    counts = np.asarray(list(class_counts(y).values()), dtype=float)
    return float(counts.max() / counts.min())


def class_weights(y: Any, scheme: str = "balanced") -> dict[Any, float]:
    """``{label: weight}`` for a weighted loss.

    ``"balanced"`` gives ``n / (n_classes * count)`` — scikit-learn's convention,
    so the weights are interchangeable with ``class_weight="balanced"`` there. The
    weights average to 1 over the *data*, which keeps the loss on the same scale as
    the unweighted one and means the learning rate does not need retuning.

    ``"inverse"`` gives ``1 / count`` normalised to a mean of 1, which is more
    aggressive on severe skew. ``"none"`` gives 1 everywhere, so a caller can pass
    the scheme through without branching.
    """
    counts = class_counts(y)
    n = float(sum(counts.values()))
    k = len(counts)
    if scheme == "none":
        return dict.fromkeys(counts, 1.0)
    if scheme == "balanced":
        return {label: n / (k * count) for label, count in counts.items()}
    if scheme == "inverse":
        raw = {label: 1.0 / count for label, count in counts.items()}
        mean = float(np.mean([raw[label] * counts[label] for label in counts])) / (n / k)
        return {label: value / mean for label, value in raw.items()}
    from qmlkit.utils.errors import unknown

    raise unknown("weighting scheme", scheme, ("balanced", "inverse", "none"))


def sample_weights(y: Any, scheme: str = "balanced") -> Array:
    """Per-sample weights, one per row of ``y`` — the array form of the above."""
    arr = _labels(y)
    table = class_weights(arr, scheme)
    return np.array([table[label] for label in arr.tolist()], dtype=float)


def pos_weight(y: Any, positive: Any = None) -> float:
    """``n_negative / n_positive`` — the weight for a single-logit binary loss.

    This is what ``torch.nn.BCEWithLogitsLoss(pos_weight=...)`` takes, and the
    number that stops a 90/10 problem training to a constant. Multiclass problems
    want :func:`class_weights` instead.
    """
    arr = _labels(y)
    counts = class_counts(arr)
    if len(counts) != 2:
        raise ValueError(
            f"pos_weight is for two classes, got {len(counts)}: "
            f"{sorted(counts)}. Use class_weights(y) for a multiclass loss."
        )
    label = max(counts) if positive is None else positive
    if label not in counts:
        raise ValueError(
            f"positive label {label!r} does not occur in y; labels are {sorted(counts)}"
        )
    n_pos = counts[label]
    return float((arr.size - n_pos) / n_pos)


def resample(
    y: Any, strategy: str = "oversample", seed: int | None = None, ratio: float = 1.0
) -> Array:
    """Indices that rebalance the classes. Apply them to ``X`` and ``y`` alike.

    ``"oversample"`` draws the minority classes up with replacement; ``"undersample"``
    draws the majority classes down without it. ``ratio`` is how far towards balance
    to go — ``1.0`` is fully balanced, ``0.5`` closes half the gap, which is often
    the better trade because oversampling a tiny class to parity mostly duplicates
    the same few points.

    Oversampling is the safer default on quantum models: undersampling throws away
    data that cost circuits to label, and these datasets are small already.
    """
    arr = _labels(y)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"ratio must be in [0, 1], got {ratio}")
    if strategy == "none":
        return np.arange(arr.size)

    rng = np.random.default_rng(seed)
    counts = class_counts(arr)
    by_label = {label: np.flatnonzero(arr == label) for label in counts}

    if strategy == "oversample":
        target = max(counts.values())
        picked = []
        for index in by_label.values():
            want = int(round(index.size + ratio * (target - index.size)))
            picked.append(index)
            if want > index.size:
                picked.append(rng.choice(index, size=want - index.size, replace=True))
    elif strategy == "undersample":
        target = min(counts.values())
        picked = []
        for index in by_label.values():
            want = int(round(index.size - ratio * (index.size - target)))
            picked.append(
                rng.choice(index, size=want, replace=False) if want < index.size else index
            )
    else:
        from qmlkit.utils.errors import unknown

        raise unknown("resampling strategy", strategy, ("oversample", "undersample", "none"))

    out = np.concatenate(picked)
    rng.shuffle(out)
    return out


def stratified_split(
    y: Any, test_size: float = 0.25, seed: int | None = None
) -> tuple[Array, Array]:
    """``(train_index, test_index)`` holding each class's share in both halves.

    Every class contributes at least one row to the test set whenever it has two
    or more samples, so a rare class cannot vanish from the evaluation — the
    failure that makes a small-data test score meaningless.
    """
    arr = _labels(y)
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")
    rng = np.random.default_rng(seed)
    train: list[Array] = []
    test: list[Array] = []
    for label in class_counts(arr):
        index = np.flatnonzero(arr == label)
        rng.shuffle(index)
        n_test = int(round(test_size * index.size))
        n_test = min(max(n_test, 1 if index.size > 1 else 0), index.size - 1)
        test.append(index[:n_test])
        train.append(index[n_test:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def stratified_folds(
    y: Any, n_folds: int = 5, seed: int | None = None
) -> list[tuple[Array, Array]]:
    """``n_folds`` ``(train_index, test_index)`` pairs, each class spread evenly.

    Raises when a class has fewer members than folds, rather than silently
    producing folds that cannot contain it — a cross-validated score whose folds
    disagree about which classes exist is not a score.
    """
    arr = _labels(y)
    counts = class_counts(arr)
    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")
    smallest = min(counts.values())
    if smallest < n_folds:
        label = min(counts, key=lambda k: counts[k])
        raise ValueError(
            f"class {label!r} has {smallest} sample(s) but {n_folds} folds were asked for. "
            f"Use n_folds<={smallest}, or resample(y) first."
        )

    rng = np.random.default_rng(seed)
    assignment = np.empty(arr.size, dtype=int)
    for label in counts:
        index = np.flatnonzero(arr == label)
        rng.shuffle(index)
        assignment[index] = np.arange(index.size) % n_folds
    everything = np.arange(arr.size)
    return [
        (everything[assignment != fold], everything[assignment == fold]) for fold in range(n_folds)
    ]


def imbalance_report(y: Any, n_folds: int = 5) -> Any:
    """What the skew in ``y`` will break, and the call that fixes each thing.

    Returns a :class:`~qmlkit.diagnostics.Report`, so it prints, is falsy when
    there is nothing wrong, and carries codes to branch on. Findings use the
    ``imbalance.*`` code prefix.
    """
    from qmlkit.diagnostics import Finding, Report

    arr = _labels(y)
    counts = class_counts(arr)
    n = arr.size
    majority = max(counts.values()) / n
    smallest_label = min(counts, key=lambda k: counts[k])
    smallest = counts[smallest_label]
    findings: list[Finding] = []

    if len(counts) < 2:
        findings.append(
            Finding(
                "imbalance.single-class",
                "error",
                f"y has one class ({smallest_label!r}), so there is nothing to classify",
                "check the labels reaching fit(); a filtered split can drop a class entirely",
                1.0,
            )
        )
        return Report(f"labels (n={n})", tuple(findings))

    if majority >= 0.9:
        findings.append(
            Finding(
                "imbalance.severe",
                "error",
                f"the majority class is {majority:.1%} of the data (ratio "
                f"{imbalance_ratio(arr):.1f}:1), so an unweighted loss is minimised by "
                "predicting it always",
                f"VQC(..., class_weight='balanced'), or pos_weight={pos_weight(arr):.2f} "
                "for a single-logit loss"
                if len(counts) == 2
                else "VQC(..., class_weight='balanced')",
                float(majority),
            )
        )
    elif majority >= _SKEWED:
        findings.append(
            Finding(
                "imbalance.skewed",
                "warning",
                f"the majority class is {majority:.1%} of the data: accuracy will overstate "
                "any model trained on it",
                "class_weight='balanced' when training, and read balanced_accuracy or mcc "
                "from qk.evaluate.classification",
                float(majority),
            )
        )

    if smallest < n_folds:
        findings.append(
            Finding(
                "imbalance.too-few-for-cv",
                "error",
                f"class {smallest_label!r} has {smallest} sample(s), fewer than the "
                f"{n_folds} folds asked for, so some folds cannot contain it",
                f"n_folds<={smallest}, or a single stratified_split, or collect more of "
                "that class",
                float(smallest),
            )
        )
    elif smallest < 10:
        findings.append(
            Finding(
                "imbalance.tiny-class",
                "warning",
                f"class {smallest_label!r} has {smallest} samples, so every per-class score "
                "for it moves by more than 0.1 with one sample",
                "report the confusion matrix alongside any per-class metric",
                float(smallest),
            )
        )

    if majority >= _SKEWED:
        findings.append(
            Finding(
                "imbalance.split",
                "info",
                "a random split can leave the minority class out of the test set entirely",
                "qk.imbalance.stratified_split(y) or stratified_folds(y)",
                None,
            )
        )
    return Report(f"labels (n={n}, {len(counts)} classes)", tuple(findings))
