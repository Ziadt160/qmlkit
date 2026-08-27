"""Every score a task needs, in one call — and a note when a score is lying.

:mod:`qmlkit.metrics` measures *circuits*: expressibility, entanglement, how flat
the gradients are. This module measures *predictions*, which is a different
question with a different failure mode.

The failure mode is that a single number is easy to report and easy to be misled
by. Accuracy on a 95/5 split is 0.95 for a model that has learned to say "no", and
that model looks better than one which actually separates the classes at 0.91.

So each task returns *all* of its metrics at once, and the ones that disagree with
each other stay visible::

    >>> import numpy as np, qmlkit as qk
    >>> y = np.array([0] * 95 + [1] * 5)
    >>> scores = qk.evaluate.classification(y, np.zeros(100, dtype=int))
    >>> round(scores["accuracy"], 3), round(scores["balanced_accuracy"], 3)
    (0.95, 0.5)
    >>> bool(scores.notes)
    True

A :class:`Scores` object indexes like a dict, iterates like one, and prints as a
table. Nothing here needs scikit-learn; everything here is cross-checked against
scikit-learn in the test suite, the same way the library is cross-checked against
PennyLane.

Four tasks are covered: :func:`classification`, :func:`regression`,
:func:`clustering` and :func:`generative`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "Scores",
    "classification",
    "regression",
    "clustering",
    "generative",
    "confusion_matrix",
    "roc_auc",
    "average_precision",
    "scores_for",
]

Array = npt.NDArray[Any]

#: Probabilities are clipped this far from 0 and 1 before a logarithm sees them.
_EPS = 1e-15


# --------------------------------------------------------------------------- #
# the container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Scores:
    """The metrics for one task, plus what they mean together.

    Indexes and iterates like a mapping, so ``scores["f1_macro"]`` and
    ``dict(scores)`` both work. ``primary`` is the single number to quote when a
    single number is unavoidable — chosen to be the one that survives imbalance.
    """

    task: str
    values: dict[str, float]
    primary: str
    n_samples: int
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        try:
            return self.values[key]
        except KeyError:
            from qmlkit.utils.errors import did_you_mean

            near = did_you_mean(key, self.values)
            hint = (" Did you mean " + " or ".join(repr(s) for s in near) + "?") if near else ""
            raise KeyError(
                f"{self.task} has no metric {key!r}.{hint} Available: "
                + ", ".join(sorted(self.values))
            ) from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.values)

    def __len__(self) -> int:
        return len(self.values)

    def __contains__(self, key: object) -> bool:
        return key in self.values

    def keys(self) -> Any:
        return self.values.keys()

    def items(self) -> Any:
        return self.values.items()

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.values.get(key, default)

    @property
    def score(self) -> float:
        """The value of :attr:`primary`."""
        return self.values[self.primary]

    def __str__(self) -> str:
        width = max((len(k) for k in self.values), default=0)
        lines = [f"{self.task} (n={self.n_samples})"]
        for name, value in self.values.items():
            mark = " *" if name == self.primary else "  "
            lines.append(f"{mark} {name:<{width}}  {value: .4f}")
        lines.extend(f"\n  note: {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _as_labels(y: Any, name: str) -> Array:
    arr = np.asarray(y).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    return arr


def _check_same_length(a: Array, b: Array) -> None:
    if a.shape[0] != b.shape[0]:
        raise ValueError(
            f"length mismatch: {a.shape[0]} true values against {b.shape[0]} predicted"
        )


def _safe_divide(num: Array, den: Array) -> Array:
    """``num/den``, with ``0`` where the denominator is ``0``.

    A class with no predictions has undefined precision. Reporting it as zero is
    the convention scikit-learn uses with ``zero_division=0``, and matching it is
    what lets the parity tests be exact.
    """
    out = np.zeros_like(num, dtype=float)
    np.divide(num, den, out=out, where=den != 0)
    return out


# --------------------------------------------------------------------------- #
# classification
# --------------------------------------------------------------------------- #
def confusion_matrix(
    y_true: Any, y_pred: Any, labels: Sequence[Any] | None = None
) -> tuple[Array, Array]:
    """``(matrix, labels)`` with rows the truth and columns the prediction."""
    truth = _as_labels(y_true, "y_true")
    pred = _as_labels(y_pred, "y_pred")
    _check_same_length(truth, pred)
    classes = np.asarray(labels) if labels is not None else np.unique(np.concatenate([truth, pred]))
    index = {label: i for i, label in enumerate(classes.tolist())}
    matrix = np.zeros((classes.size, classes.size), dtype=np.int64)
    for t, p in zip(truth.tolist(), pred.tolist(), strict=True):
        if t in index and p in index:
            matrix[index[t], index[p]] += 1
    return matrix, classes


def roc_auc(y_true: Any, y_score: Any) -> float:
    """Binary ROC AUC by the rank identity, so ties are handled exactly.

    ``AUC = (sum of positive ranks - n_pos(n_pos+1)/2) / (n_pos * n_neg)`` with
    average ranks over tied scores — identical to integrating the ROC curve with
    the trapezoid rule, and cheaper.
    """
    truth = _as_labels(y_true, "y_true").astype(int)
    score = np.asarray(y_score, dtype=float).ravel()
    _check_same_length(truth, score)
    n_pos = int((truth == 1).sum())
    n_neg = int(truth.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(score, kind="mergesort")
    sorted_scores = score[order]
    ranks = np.empty(score.size, dtype=float)
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # average rank, 1-based
        i = j + 1
    return float((ranks[truth == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(y_true: Any, y_score: Any) -> float:
    """Area under the precision-recall curve, ``sum (R_n - R_{n-1}) P_n``.

    The right summary when the positive class is rare: unlike ROC AUC, it does not
    flatter a model for correctly rejecting an abundant negative class.
    """
    truth = _as_labels(y_true, "y_true").astype(int)
    score = np.asarray(y_score, dtype=float).ravel()
    _check_same_length(truth, score)
    n_pos = int((truth == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-score, kind="mergesort")
    truth, score = truth[order], score[order]
    tp = np.cumsum(truth)
    fp = np.cumsum(1 - truth)
    # collapse ties: no threshold can separate two identical scores
    keep = np.r_[np.diff(score) != 0, True]
    tp, fp = tp[keep], fp[keep]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _log_loss(onehot: Array, proba: Array) -> float:
    p = np.clip(np.asarray(proba, dtype=float), _EPS, 1.0 - _EPS)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.sum(onehot * np.log(p), axis=1)))


def _brier(onehot: Array, proba: Array) -> float:
    """Multiclass Brier score: mean squared error over the probability vector.

    Two-column binary input reduces to the familiar ``mean (p - y)^2``, which is
    what scikit-learn's binary ``brier_score_loss`` reports.
    """
    p = np.asarray(proba, dtype=float)
    if p.shape[1] == 2:
        return float(np.mean((p[:, 1] - onehot[:, 1]) ** 2))
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def classification(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
    labels: Sequence[Any] | None = None,
) -> Scores:
    """Every classification metric worth reporting, and a note when one misleads.

    Parameters
    ----------
    y_true, y_pred:
        Labels. Any hashable label type; they are matched by value.
    y_score:
        Optional. Either an ``(n,)`` vector of scores for the positive class, or an
        ``(n, n_classes)`` array of probabilities. Unlocks ``roc_auc``,
        ``average_precision``, ``log_loss`` and ``brier``.
    labels:
        Optional fixed class order, for when a fold is missing a class.

    Notes
    -----
    The primary metric is ``balanced_accuracy`` on imbalanced data and ``accuracy``
    otherwise, because quoting accuracy on a skewed problem is the most common way
    a quantum classifier gets reported as working when it is not.
    """
    truth = _as_labels(y_true, "y_true")
    pred = _as_labels(y_pred, "y_pred")
    _check_same_length(truth, pred)
    matrix, classes = confusion_matrix(truth, pred, labels)
    n = int(matrix.sum())
    if n == 0:
        raise ValueError("no samples fall inside the given labels")

    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    correct = np.diag(matrix).astype(float)

    recall = _safe_divide(correct, support)
    precision = _safe_divide(correct, predicted)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    weight = support / n

    accuracy = float(correct.sum() / n)
    present = support > 0
    balanced = float(recall[present].mean()) if present.any() else 0.0

    # Matthews correlation straight from the confusion matrix — the multiclass
    # generalisation, which reduces to the familiar 2x2 formula for two classes.
    total_sq = float(n) ** 2
    cov_pt = float(correct.sum() * n - predicted @ support)
    cov_pp = total_sq - float(predicted @ predicted)
    cov_tt = total_sq - float(support @ support)
    denom = float(np.sqrt(cov_pp * cov_tt))
    mcc = cov_pt / denom if denom > 0 else 0.0

    chance = float(predicted @ support) / total_sq
    kappa = (accuracy - chance) / (1.0 - chance) if chance < 1.0 else 0.0

    values: dict[str, float] = {
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision_macro": float(precision[present].mean()) if present.any() else 0.0,
        "recall_macro": float(recall[present].mean()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean()) if present.any() else 0.0,
        "precision_weighted": float(precision @ weight),
        "recall_weighted": float(recall @ weight),
        "f1_weighted": float(f1 @ weight),
        "mcc": float(mcc),
        "cohen_kappa": float(kappa),
    }

    notes: list[str] = []
    extras: dict[str, Any] = {
        "confusion_matrix": matrix,
        "labels": classes,
        "support": support.astype(int),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(classes.tolist())
        },
    }

    # ---- scores, when they were supplied ---------------------------------- #
    if y_score is not None:
        score_arr = np.asarray(y_score, dtype=float)
        if score_arr.ndim == 1:
            score_arr = score_arr.reshape(-1, 1)
        _check_same_length(truth, score_arr)
        onehot = (truth[:, None] == classes[None, :]).astype(float)

        if classes.size == 2:
            positive = score_arr[:, -1]
            binary_truth = (truth == classes[1]).astype(int)
            values["roc_auc"] = roc_auc(binary_truth, positive)
            values["average_precision"] = average_precision(binary_truth, positive)
            if score_arr.shape[1] == 2:
                values["log_loss"] = _log_loss(onehot, score_arr)
                values["brier"] = _brier(onehot, score_arr)
        elif score_arr.shape[1] == classes.size:
            # one-vs-rest, averaged over the classes that actually occur
            aucs = [
                roc_auc((truth == label).astype(int), score_arr[:, i])
                for i, label in enumerate(classes.tolist())
                if support[i] > 0
            ]
            finite = [a for a in aucs if np.isfinite(a)]
            if finite:
                values["roc_auc_ovr"] = float(np.mean(finite))
            values["log_loss"] = _log_loss(onehot, score_arr)
            values["brier"] = _brier(onehot, score_arr)
        else:
            notes.append(
                f"y_score has {score_arr.shape[1]} column(s) for {classes.size} classes, "
                "so the threshold metrics were skipped"
            )

    # ---- the notes that stop a number being misread ----------------------- #
    majority = float(support.max() / n)
    smallest = int(support[present].min()) if present.any() else 0
    if accuracy <= majority + 1e-12:
        argmax = int(np.argmax(support))
        notes.append(
            f"accuracy {accuracy:.3f} is at the majority-class rate ({majority:.3f}): this "
            f'model has not beaten "always predict class {classes[argmax]}". Read '
            f"balanced_accuracy ({balanced:.3f}) or mcc ({mcc:.3f}) instead."
        )
    elif majority >= 0.65:
        notes.append(
            f"classes are imbalanced ({majority:.1%} majority), so accuracy {accuracy:.3f} "
            f"overstates this model: balanced_accuracy is {balanced:.3f}, mcc {mcc:.3f}"
        )
    if 0 < smallest < 10:
        notes.append(
            f"the smallest class has {smallest} sample(s), so its precision and recall "
            "carry very wide error bars"
        )
    if classes.size == 2 and majority >= 0.8 and "average_precision" in values:
        notes.append(
            f"at {majority:.1%} majority, average_precision "
            f"({values['average_precision']:.3f}) is the more honest curve summary; "
            f"roc_auc is {values['roc_auc']:.3f}"
        )

    primary = "balanced_accuracy" if majority >= 0.65 else "accuracy"
    return Scores("classification", values, primary, n, tuple(notes), extras)


# --------------------------------------------------------------------------- #
# regression
# --------------------------------------------------------------------------- #
def regression(y_true: Any, y_pred: Any) -> Scores:
    """Every regression metric worth reporting, with R2 as the primary."""
    truth = np.asarray(y_true, dtype=float).ravel()
    pred = np.asarray(y_pred, dtype=float).ravel()
    _check_same_length(truth, pred)
    n = truth.size
    residual = truth - pred

    ss_res = float(residual @ residual)
    centred = truth - truth.mean()
    ss_tot = float(centred @ centred)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mse = ss_res / n

    values = {
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.abs(residual).mean()),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "max_error": float(np.abs(residual).max()),
        "explained_variance": (
            float(1.0 - residual.var() / truth.var()) if truth.var() > 0 else 0.0
        ),
    }

    notes: list[str] = []
    nonzero = truth != 0
    if bool(nonzero.all()):
        values["mape"] = float(np.abs(residual / truth).mean())
    else:
        notes.append(
            f"{int((~nonzero).sum())} target(s) are exactly zero, so mape is undefined "
            "and was omitted"
        )
    if ss_tot <= 0:
        notes.append("every target is identical, so r2 is undefined and reported as 0.0")
    elif r2 <= 0.0:
        notes.append(
            f"r2 {r2:.3f} is at or below zero: predicting the mean ({truth.mean():.4g}) "
            "everywhere would do as well or better"
        )
    return Scores("regression", values, "r2", n, tuple(notes))


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def _pairwise_sq_distances(X: Array) -> Array:
    sq = np.einsum("ij,ij->i", X, X)
    out: Array = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    return out


def _silhouette(X: Array, labels: Array) -> float:
    """Mean silhouette, computed from the full distance matrix.

    ``O(n^2)`` in memory, which is the right trade for a report: a diagnostic run
    on a few thousand points should be exact rather than sampled.
    """
    classes = np.unique(labels)
    if classes.size < 2:
        return float("nan")
    distances = np.sqrt(_pairwise_sq_distances(X))
    scores = np.zeros(labels.size, dtype=float)
    for i in range(labels.size):
        own = labels == labels[i]
        n_own = int(own.sum())
        if n_own <= 1:
            scores[i] = 0.0  # a singleton is neither cohesive nor separated
            continue
        a = float(distances[i, own].sum() / (n_own - 1))
        b = min(float(distances[i, labels == c].mean()) for c in classes if c != labels[i])
        scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(scores.mean())


def _davies_bouldin(X: Array, labels: Array) -> float:
    classes = np.unique(labels)
    if classes.size < 2:
        return float("nan")
    centroids = np.stack([X[labels == c].mean(axis=0) for c in classes])
    spread = np.array(
        [
            float(np.linalg.norm(X[labels == c] - centroids[i], axis=1).mean())
            for i, c in enumerate(classes)
        ]
    )
    worst = []
    for i in range(classes.size):
        ratios = [
            (spread[i] + spread[j]) / float(np.linalg.norm(centroids[i] - centroids[j]))
            for j in range(classes.size)
            if j != i and np.linalg.norm(centroids[i] - centroids[j]) > 0
        ]
        worst.append(max(ratios) if ratios else 0.0)
    return float(np.mean(worst))


def _contingency(a: Array, b: Array) -> Array:
    _, ra = np.unique(a, return_inverse=True)
    _, rb = np.unique(b, return_inverse=True)
    table = np.zeros((int(ra.max()) + 1, int(rb.max()) + 1), dtype=np.int64)
    np.add.at(table, (ra.ravel(), rb.ravel()), 1)
    return table


def _comb2(x: Array | float) -> Any:
    return np.asarray(x, dtype=float) * (np.asarray(x, dtype=float) - 1.0) / 2.0


def _adjusted_rand(table: Array) -> float:
    n = int(table.sum())
    if n < 2:
        return float("nan")
    sum_ij = float(_comb2(table).sum())
    sum_i = float(_comb2(table.sum(axis=1)).sum())
    sum_j = float(_comb2(table.sum(axis=0)).sum())
    total = float(_comb2(float(n)))
    expected = sum_i * sum_j / total
    maximum = 0.5 * (sum_i + sum_j)
    return (sum_ij - expected) / (maximum - expected) if maximum != expected else 0.0


def _entropy(counts: Array) -> float:
    p = counts[counts > 0].astype(float)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def _normalised_mutual_info(table: Array) -> float:
    n = float(table.sum())
    row, col = table.sum(axis=1), table.sum(axis=0)
    h_row, h_col = _entropy(row), _entropy(col)
    if h_row == 0.0 or h_col == 0.0:
        return 0.0
    nz = table > 0
    joint = table[nz].astype(float) / n
    outer = np.outer(row, col)[nz].astype(float) / (n * n)
    mutual = float((joint * np.log(joint / outer)).sum())
    return mutual / ((h_row + h_col) / 2.0)


def clustering(X: Any, labels: Any, y_true: Any | None = None) -> Scores:
    """Cluster quality — internal always, external when ``y_true`` is given.

    Internal metrics (silhouette, Davies-Bouldin) need no ground truth and say
    whether the clusters are separated at all. External ones (ARI, NMI, purity)
    say whether they are the clusters you were looking for; the two routinely
    disagree, which is the useful part.
    """
    data = np.atleast_2d(np.asarray(X, dtype=float))
    assigned = _as_labels(labels, "labels")
    if data.shape[0] != assigned.size:
        raise ValueError(f"X has {data.shape[0]} rows but {assigned.size} labels were given")
    classes = np.unique(assigned)

    values = {
        "n_clusters": float(classes.size),
        "silhouette": _silhouette(data, assigned),
        "davies_bouldin": _davies_bouldin(data, assigned),
    }
    notes: list[str] = []
    extras: dict[str, Any] = {
        "cluster_sizes": {str(c): int((assigned == c).sum()) for c in classes.tolist()}
    }
    primary = "silhouette"

    if y_true is not None:
        truth = _as_labels(y_true, "y_true")
        _check_same_length(truth, assigned)
        table = _contingency(truth, assigned)
        values["adjusted_rand"] = _adjusted_rand(table)
        values["normalized_mutual_info"] = _normalised_mutual_info(table)
        values["purity"] = float(table.max(axis=0).sum() / table.sum())
        extras["contingency"] = table
        primary = "adjusted_rand"
        if values["adjusted_rand"] < 0.05:
            notes.append(
                f"adjusted_rand {values['adjusted_rand']:.3f} is near zero: these clusters "
                "agree with the labels about as well as a random partition would"
            )

    if classes.size < 2:
        notes.append("only one cluster was assigned, so the internal metrics are undefined")
    elif np.isfinite(values["silhouette"]) and values["silhouette"] < 0.1:
        notes.append(
            f"silhouette {values['silhouette']:.3f} is near zero: points sit about as close "
            "to a neighbouring cluster as to their own, so the partition is weak"
        )
    return Scores("clustering", values, primary, int(assigned.size), tuple(notes), extras)


# --------------------------------------------------------------------------- #
# generative
# --------------------------------------------------------------------------- #
def _align(p: Any, q: Any) -> tuple[Array, Array]:
    """Two distributions on a common support, each normalised.

    Accepts dicts keyed by bitstring — what :func:`~qmlkit.core.execute.run_counts`
    returns — or dense arrays. Dicts are unioned, so an outcome one distribution
    never produced still contributes, which is exactly where KL diverges and the
    caller needs to know.
    """
    if isinstance(p, Mapping) or isinstance(q, Mapping):
        if not (isinstance(p, Mapping) and isinstance(q, Mapping)):
            raise TypeError("pass both distributions as mappings, or both as arrays")
        support = sorted(set(p) | set(q))
        a = np.array([float(p.get(k, 0.0)) for k in support], dtype=float)
        b = np.array([float(q.get(k, 0.0)) for k in support], dtype=float)
    else:
        a = np.asarray(p, dtype=float).ravel()
        b = np.asarray(q, dtype=float).ravel()
        if a.size != b.size:
            raise ValueError(f"distributions have different supports: {a.size} against {b.size}")
    for name, arr in (("p_model", a), ("q_target", b)):
        if bool(np.any(arr < 0)):
            raise ValueError(f"{name} has negative entries, so it is not a distribution")
        if arr.sum() <= 0:
            raise ValueError(f"{name} sums to zero")
    return a / a.sum(), b / b.sum()


def _kl(a: Array, b: Array) -> float:
    mask = a > 0
    if bool(np.any(b[mask] == 0)):
        return float("inf")
    return float((a[mask] * np.log(a[mask] / b[mask])).sum())


def generative(p_model: Any, q_target: Any) -> Scores:
    """How far a generated distribution is from the one it was fitted to.

    ``p_model`` is the model's distribution, ``q_target`` the data's. Both may be
    ``{bitstring: count}`` dicts or dense arrays; counts are normalised.

    Divergences are in **nats**. Total variation is the primary because it is a
    metric, is bounded in ``[0, 1]``, and stays finite when the model puts zero
    mass where the target has some — the case in which KL is ``inf`` and stops
    being a training signal at all.
    """
    p, q = _align(p_model, q_target)

    tv = float(0.5 * np.abs(p - q).sum())
    hellinger = float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2.0))
    m = 0.5 * (p + q)
    js = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)

    values = {
        "total_variation": tv,
        "hellinger": hellinger,
        "js_divergence": float(js),
        "js_distance": float(np.sqrt(max(js, 0.0))),
        "kl_model_target": _kl(p, q),
        "kl_target_model": _kl(q, p),
        "support_coverage": float((q[p > 0] > 0).sum() / max(int((q > 0).sum()), 1)),
    }

    notes: list[str] = []
    missed = int(((q > 0) & (p == 0)).sum())
    if missed:
        notes.append(
            f"the model puts zero mass on {missed} outcome(s) the target reaches, so "
            "kl_target_model is infinite; total_variation and js_distance stay finite "
            "and are the ones to optimise"
        )
    if tv > 0.5:
        notes.append(
            f"total_variation {tv:.3f} exceeds 0.5: the two distributions disagree on more "
            "than half their mass, which is not a small modelling error"
        )
    return Scores("generative", values, "total_variation", int(p.size), tuple(notes))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def scores_for(task: str, *args: Any, **kwargs: Any) -> Scores:
    """Call one of the four by name — for code that is generic over the task."""
    table: dict[str, Callable[..., Scores]] = {
        "classification": classification,
        "regression": regression,
        "clustering": clustering,
        "generative": generative,
    }
    if task not in table:
        from qmlkit.utils.errors import unknown

        raise unknown("task", task, table)
    return table[task](*args, **kwargs)
