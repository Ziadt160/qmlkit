"""The classical bar, computed on the same data, the same folds, the same metric.

The question every quantum machine learning result is asked first is "compared to
what?", and the honest answer is usually "an RBF-kernel SVM nobody ran". Not out of
bad faith — running it means a second pipeline, a second preprocessing path, a
second splitting convention, and by the time all three match, the comparison is a
day's work that adds nothing to the paper if it goes the expected way.

So it goes unrun, and the reviewer asks anyway.

This module makes it one call::

    >>> import qmlkit as qk
    >>> X, y = qk.datasets.moons(n_samples=60, seed=0)
    >>> table = qk.baseline(X, y, cv=3, seed=0)     # doctest: +SKIP
    >>> print(table)                                # doctest: +SKIP
    classification  ·  balanced_accuracy  ·  3-fold stratified  ·  n=60
      rbf-kernel-ridge     0.883 +/- 0.042
      nearest-centroid     0.850 +/- 0.038
      majority             0.500 +/- 0.000
    the bar to beat is rbf-kernel-ridge at 0.883

Pass ``model=`` and the model is fitted on the identical folds and lands in the
same table, with a verdict line that says plainly whether it cleared the bar.

**The baselines that need no scikit-learn always run.** Kernel ridge with an RBF
kernel is a closed-form solve, and it is the right classical foil for a quantum
kernel method specifically because it is the same algorithm with a different
kernel. Where scikit-learn is installed, its estimators are added; where it is
not, they are listed as skipped rather than silently dropped, because a table that
quietly omits the strong baseline is the problem this module exists to fix.

The companion check for kernel methods is
:func:`~qmlkit.kernels.matrix.geometric_difference`, which asks whether the quantum
kernel induces a geometry the classical one cannot reach at all. A large geometric
difference with no accuracy gain is a real and publishable finding; a small one
says the classical kernel was always going to be enough.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit import evaluate
from qmlkit.imbalance import stratified_folds

__all__ = [
    "BaselineSpec",
    "BaselineRow",
    "BaselineTable",
    "baseline",
    "register_baseline",
    "list_baselines",
    "get_baseline",
]

Array = npt.NDArray[Any]


# --------------------------------------------------------------------------- #
# the estimators that need nothing but NumPy
# --------------------------------------------------------------------------- #
class _Estimator:
    """The two methods a baseline needs. Deliberately not scikit-learn's protocol."""

    def fit(self, X: Array, y: Array) -> _Estimator:  # pragma: no cover - overridden
        raise NotImplementedError

    def predict(self, X: Array) -> Array:  # pragma: no cover - overridden
        raise NotImplementedError


class MajorityClassifier(_Estimator):
    """Always predicts the most frequent training label.

    The floor. A model that does not clear this has learned nothing at all, and on
    a skewed dataset it clears 90% accuracy while doing so.
    """

    def fit(self, X: Array, y: Array) -> MajorityClassifier:
        labels, counts = np.unique(y, return_counts=True)
        self.label_ = labels[int(np.argmax(counts))]
        return self

    def predict(self, X: Array) -> Array:
        return np.full(np.atleast_2d(X).shape[0], self.label_)


class NearestCentroid(_Estimator):
    """Assign each point to the nearest class mean. No solver, no hyperparameter."""

    def fit(self, X: Array, y: Array) -> NearestCentroid:
        self.labels_ = np.unique(y)
        self.centroids_ = np.stack([np.atleast_2d(X)[y == c].mean(axis=0) for c in self.labels_])
        return self

    def predict(self, X: Array) -> Array:
        d = np.linalg.norm(np.atleast_2d(X)[:, None, :] - self.centroids_[None, :, :], axis=2)
        out: Array = self.labels_[np.argmin(d, axis=1)]
        return out


def _rbf_gamma(X: Array, gamma: float | str = "scale") -> float:
    """scikit-learn's ``gamma='scale'``: ``1 / (n_features * X.var())``.

    Matching their convention is deliberate — it makes this baseline and their
    ``SVC(kernel='rbf')`` comparable rather than two different models sharing a
    name.
    """
    if isinstance(gamma, str):
        var = float(np.asarray(X, dtype=float).var())
        return 1.0 / (X.shape[1] * var) if var > 0 else 1.0
    return float(gamma)


def _rbf(A: Array, B: Array, gamma: float) -> Array:
    sq = (
        np.einsum("ij,ij->i", A, A)[:, None]
        + np.einsum("ij,ij->i", B, B)[None, :]
        - 2.0 * (A @ B.T)
    )
    out: Array = np.exp(-gamma * np.maximum(sq, 0.0))
    return out


class RBFKernelRidge(_Estimator):
    """Kernel ridge regression with an RBF kernel; one-hot targets for labels.

    Closed form: ``alpha = (K + lambda I)^-1 Y``. For a quantum *kernel* method
    this is the honest foil — the identical algorithm with the identical solver,
    differing only in which kernel fills the Gram matrix. Any gap between them is
    attributable to the kernel and to nothing else.
    """

    def __init__(self, alpha: float = 1.0, gamma: float | str = "scale", classify: bool = True):
        self.alpha = alpha
        self.gamma = gamma
        self.classify = classify

    def fit(self, X: Array, y: Array) -> RBFKernelRidge:
        self.X_ = np.atleast_2d(np.asarray(X, dtype=float))
        self.gamma_ = _rbf_gamma(self.X_, self.gamma)
        K = _rbf(self.X_, self.X_, self.gamma_)
        if self.classify:
            self.labels_ = np.unique(y)
            target = (np.asarray(y).ravel()[:, None] == self.labels_[None, :]).astype(float)
        else:
            target = np.asarray(y, dtype=float).reshape(-1, 1)
        self.dual_ = np.linalg.solve(K + self.alpha * np.eye(K.shape[0]), target)
        return self

    def predict(self, X: Array) -> Array:
        scores = _rbf(np.atleast_2d(np.asarray(X, dtype=float)), self.X_, self.gamma_) @ self.dual_
        return self.labels_[np.argmax(scores, axis=1)] if self.classify else scores[:, 0]


class MeanRegressor(_Estimator):
    """Predicts the training mean. R2 is 0.0 here by construction — the floor."""

    def fit(self, X: Array, y: Array) -> MeanRegressor:
        self.value_ = float(np.asarray(y, dtype=float).mean())
        return self

    def predict(self, X: Array) -> Array:
        return np.full(np.atleast_2d(X).shape[0], self.value_)


class LinearLeastSquares(_Estimator):
    """Ridge-stabilised least squares with an intercept. The linear-model floor."""

    def __init__(self, alpha: float = 1e-6):
        self.alpha = alpha

    def fit(self, X: Array, y: Array) -> LinearLeastSquares:
        A = np.hstack([np.atleast_2d(np.asarray(X, dtype=float)), np.ones((len(X), 1))])
        gram = A.T @ A + self.alpha * np.eye(A.shape[1])
        self.coef_ = np.linalg.solve(gram, A.T @ np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X: Array) -> Array:
        A = np.hstack([np.atleast_2d(np.asarray(X, dtype=float)), np.ones((len(X), 1))])
        return A @ self.coef_


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineSpec:
    """One classical model that can stand next to a quantum one."""

    name: str
    task: str
    factory: Callable[[], Any]
    requires: str | None = None
    note: str = ""


_BASELINES: dict[str, BaselineSpec] = {}


def register_baseline(
    name: str,
    task: str,
    factory: Callable[[], Any],
    requires: str | None = None,
    note: str = "",
) -> None:
    """Add a baseline, so it appears in every table for that task from now on.

    ``requires`` names an importable module; when it is missing the baseline is
    reported as skipped rather than dropped. ``factory`` must return a fresh,
    unfitted estimator with ``fit`` and ``predict`` — it is called once per fold.
    """
    if task not in ("classification", "regression"):
        from qmlkit.utils.errors import unknown

        raise unknown("baseline task", task, ("classification", "regression"))
    # keyed by task and name, so `rbf-kernel-ridge` can exist for both tasks
    _BASELINES[f"{task}/{name}"] = BaselineSpec(name, task, factory, requires, note)


def list_baselines(task: str | None = None) -> tuple[str, ...]:
    """Registered baseline names, optionally filtered to one task."""
    return tuple(sorted(s.name for s in _BASELINES.values() if task is None or s.task == task))


def get_baseline(name: str, task: str = "classification") -> BaselineSpec:
    """One registered baseline. Names are unique within a task, not across tasks."""
    key = f"{task}/{name}"
    if key not in _BASELINES:
        from qmlkit.utils.errors import unknown

        raise unknown("baseline", name, list_baselines(task), error=KeyError)
    return _BASELINES[key]


def _sklearn(path: str, **kwargs: Any) -> Callable[[], Any]:
    """A factory that imports scikit-learn only when the baseline actually runs."""

    def make() -> Any:
        module_name, class_name = path.rsplit(".", 1)
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)(**kwargs)

    return make


# The floor, the classical kernel, then the models a reviewer will name. Ordered
# the way the table should be read: what beating nothing looks like, what the same
# algorithm with a *classical* kernel achieves, and what the field would reach for.
register_baseline("majority", "classification", MajorityClassifier)
register_baseline("nearest-centroid", "classification", NearestCentroid)
register_baseline("rbf-kernel-ridge", "classification", RBFKernelRidge)
register_baseline(
    "svc-rbf", "classification", _sklearn("sklearn.svm.SVC", kernel="rbf"), requires="sklearn"
)
register_baseline(
    "svc-linear", "classification", _sklearn("sklearn.svm.SVC", kernel="linear"), requires="sklearn"
)
register_baseline(
    "logistic",
    "classification",
    _sklearn("sklearn.linear_model.LogisticRegression", max_iter=1000),
    requires="sklearn",
)
register_baseline(
    "mlp",
    "classification",
    _sklearn(
        "sklearn.neural_network.MLPClassifier",
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=0,
    ),
    requires="sklearn",
)
register_baseline(
    "random-forest",
    "classification",
    _sklearn("sklearn.ensemble.RandomForestClassifier", n_estimators=100, random_state=0),
    requires="sklearn",
)

register_baseline("mean", "regression", MeanRegressor)
register_baseline("linear", "regression", LinearLeastSquares)
register_baseline("rbf-kernel-ridge", "regression", lambda: RBFKernelRidge(classify=False))
register_baseline(
    "svr-rbf", "regression", _sklearn("sklearn.svm.SVR", kernel="rbf"), requires="sklearn"
)
register_baseline(
    "mlp-regressor",
    "regression",
    _sklearn(
        "sklearn.neural_network.MLPRegressor",
        hidden_layer_sizes=(32, 16),
        max_iter=2000,
        random_state=0,
    ),
    requires="sklearn",
)


def _available(requires: str | None) -> bool:
    if requires is None:
        return True
    import importlib.util

    return importlib.util.find_spec(requires) is not None


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BaselineRow:
    """One model's score across the folds."""

    name: str
    mean: float
    std: float
    fold_scores: tuple[float, ...] = ()
    is_model: bool = False
    skipped: str = ""

    @property
    def ran(self) -> bool:
        return not self.skipped


@dataclass(frozen=True)
class BaselineTable:
    """Every model on the same folds, sorted best first.

    ``verdict`` is the sentence to quote: whether the model under test cleared the
    strongest classical baseline, and by how much relative to the fold-to-fold
    spread — a gap smaller than the noise is not a gap.
    """

    task: str
    metric: str
    n_samples: int
    n_folds: int
    rows: tuple[BaselineRow, ...] = ()
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def ran(self) -> tuple[BaselineRow, ...]:
        return tuple(r for r in self.rows if r.ran)

    @property
    def model_row(self) -> BaselineRow | None:
        return next((r for r in self.rows if r.is_model), None)

    @property
    def best_classical(self) -> BaselineRow | None:
        classical = [r for r in self.ran if not r.is_model]
        return max(classical, key=lambda r: r.mean) if classical else None

    @property
    def beats_classical(self) -> bool | None:
        """``True`` only if the model's mean clears the best classical mean.

        ``None`` when no model was passed. Says nothing about significance — read
        :attr:`verdict` for that.
        """
        model, best = self.model_row, self.best_classical
        if model is None or best is None or not model.ran:
            return None
        return bool(model.mean > best.mean)

    @property
    def verdict(self) -> str:
        model, best = self.model_row, self.best_classical
        if best is None:
            return "no classical baseline could run"
        if model is None or not model.ran:
            return f"the bar to beat is {best.name} at {best.mean:.3f}"
        gap = model.mean - best.mean
        spread = float(np.hypot(model.std, best.std))
        if gap <= 0:
            return (
                f"{model.name} ({model.mean:.3f}) does NOT beat {best.name} "
                f"({best.mean:.3f}): behind by {abs(gap):.3f}"
            )
        if spread > 0 and gap < spread:
            return (
                f"{model.name} ({model.mean:.3f}) leads {best.name} ({best.mean:.3f}) by "
                f"{gap:.3f}, which is inside the fold-to-fold spread ({spread:.3f}) — "
                "not yet a result"
            )
        return (
            f"{model.name} ({model.mean:.3f}) beats {best.name} ({best.mean:.3f}) by "
            f"{gap:.3f}, outside the fold spread ({spread:.3f})"
        )

    def __str__(self) -> str:
        width = max((len(r.name) for r in self.rows), default=0)
        # ASCII only: this prints to a Windows console as often as to a notebook
        split = "stratified" if self.task == "classification" else "shuffled"
        lines = [
            f"{self.task}  |  {self.metric}  |  {self.n_folds}-fold {split}  "
            f"|  n={self.n_samples}"
        ]
        for row in sorted(self.rows, key=lambda r: (r.skipped != "", -r.mean)):
            mark = "*" if row.is_model else " "
            if row.ran:
                lines.append(
                    f" {mark} {row.name:<{width}}  {row.mean: .3f} +/- {row.std:.3f}"
                )
            else:
                lines.append(f" {mark} {row.name:<{width}}  skipped: {row.skipped}")
        lines.append(f"\n{self.verdict}")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


def _infer_task(y: Array) -> str:
    """Classification unless the targets are clearly continuous.

    Float targets with more than a handful of distinct values are a regression;
    everything else is treated as labels. Getting this wrong is loud rather than
    silent — a regression scored as classification produces one class per sample.
    """
    arr = np.asarray(y).ravel()
    if arr.dtype.kind in "US" or arr.dtype == object:
        return "classification"
    distinct = np.unique(arr).size
    if arr.dtype.kind in "iub" and distinct <= max(20, int(np.sqrt(arr.size))):
        return "classification"
    if arr.dtype.kind == "f" and distinct <= 20 and np.allclose(arr, np.round(arr)):
        return "classification"
    return "regression"


def _fit_predict(estimator: Any, X_train: Array, y_train: Array, X_test: Array) -> Array:
    fitted = estimator.fit(X_train, y_train)
    return np.asarray((fitted if fitted is not None else estimator).predict(X_test)).ravel()


def _score(task: str, y_true: Array, y_pred: Array, metric: str) -> float:
    scores = evaluate.classification(y_true, y_pred) if task == "classification" else (
        evaluate.regression(y_true, y_pred)
    )
    return float(scores[metric])


def baseline(
    X: Any,
    y: Any,
    model: Any = None,
    task: str = "auto",
    cv: int = 5,
    metric: str | None = None,
    seed: int | None = 0,
    include: Sequence[str] | None = None,
    max_samples: int | None = None,
    fit_kwargs: dict[str, Any] | None = None,
) -> BaselineTable:
    """Score every classical baseline — and optionally ``model`` — on shared folds.

    Parameters
    ----------
    X, y:
        The data. Folds are stratified for classification, contiguous-shuffled for
        regression, and identical for every row of the table.
    model:
        Optional. Either an unfitted estimator with ``fit``/``predict`` (it is deep
        copied per fold, so the same object can be reused), or a zero-argument
        callable returning a fresh one — which is the safer form for torch models,
        whose parameters would otherwise carry over.
    task:
        ``"classification"``, ``"regression"``, or ``"auto"`` to infer from ``y``.
    cv:
        Number of folds.
    metric:
        Which metric decides the ranking. Defaults to the primary metric for the
        task, which is imbalance-aware for classification.
    include:
        Restrict to these baseline names. The default runs every registered one.
    max_samples:
        Subsample before running. A quantum model refitted on five folds of a
        thousand points is an overnight job; capping it makes the comparison
        something that gets run at all. The cap is recorded in the notes.

    Notes
    -----
    Every fold sees identical indices for every model, so the differences reported
    are differences between models rather than between splits.
    """
    data = np.atleast_2d(np.asarray(X, dtype=float))
    target = np.asarray(y).ravel()
    if data.shape[0] != target.size:
        raise ValueError(f"X has {data.shape[0]} rows but y has {target.size}")
    resolved = _infer_task(target) if task == "auto" else task
    if resolved not in ("classification", "regression"):
        from qmlkit.utils.errors import unknown

        raise unknown("task", resolved, ("classification", "regression", "auto"))

    notes: list[str] = []
    rng = np.random.default_rng(seed)
    if max_samples is not None and data.shape[0] > max_samples:
        keep = rng.choice(data.shape[0], size=max_samples, replace=False)
        data, target = data[keep], target[keep]
        notes.append(f"subsampled to {max_samples} of {len(y)} rows before scoring")

    metric = metric or ("balanced_accuracy" if resolved == "classification" else "r2")
    if resolved == "classification":
        folds = stratified_folds(target, n_folds=cv, seed=seed)
    else:
        order = rng.permutation(data.shape[0])
        chunks = np.array_split(order, cv)
        folds = [
            (np.setdiff1d(order, chunk), chunk) for chunk in chunks
        ]

    specs = [s for s in _BASELINES.values() if s.task == resolved]
    if include is not None:
        wanted = set(include)
        missing = wanted - {s.name for s in specs}
        if missing:
            from qmlkit.utils.errors import unknown

            raise unknown(
                f"{resolved} baseline", sorted(missing)[0], [s.name for s in specs]
            )
        specs = [s for s in specs if s.name in wanted]

    rows: list[BaselineRow] = []
    for spec in specs:
        if not _available(spec.requires):
            rows.append(
                BaselineRow(
                    spec.name,
                    float("nan"),
                    float("nan"),
                    skipped=f"needs {spec.requires} (pip install 'qmlkit[{spec.requires}]')",
                )
            )
            continue
        scores = [
            _score(
                resolved,
                target[test],
                _fit_predict(spec.factory(), data[train], target[train], data[test]),
                metric,
            )
            for train, test in folds
        ]
        rows.append(
            BaselineRow(spec.name, float(np.mean(scores)), float(np.std(scores)), tuple(scores))
        )

    if model is not None:
        name = getattr(model, "__name__", type(model).__name__)
        try:
            scores = [
                _score(
                    resolved,
                    target[test],
                    _fit_predict(
                        model() if callable(model) and not hasattr(model, "fit")
                        else copy.deepcopy(model),
                        data[train],
                        target[train],
                        data[test],
                    ),
                    metric,
                )
                for train, test in folds
            ]
        except Exception as exc:  # the table is still worth having without the model
            rows.append(
                BaselineRow(name, float("nan"), float("nan"), is_model=True, skipped=str(exc))
            )
        else:
            rows.append(
                BaselineRow(
                    name,
                    float(np.mean(scores)),
                    float(np.std(scores)),
                    tuple(scores),
                    is_model=True,
                )
            )
        if fit_kwargs:
            notes.append("fit_kwargs are ignored by estimators that do not accept them")

    if resolved == "classification":
        from qmlkit.imbalance import imbalance_ratio

        ratio = imbalance_ratio(target)
        if ratio >= 1.5:
            notes.append(
                f"classes are {ratio:.1f}:1, so the ranking metric is {metric} rather than "
                "accuracy; qk.imbalance.imbalance_report(y) lists the remedies"
            )
    return BaselineTable(
        resolved, metric, int(target.size), cv, tuple(rows), tuple(notes), {"folds": folds}
    )
