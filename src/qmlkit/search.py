"""One call to sweep everything tunable, with the broken configurations skipped.

    best = qk.search(X, y, ansatz=["hardware_efficient", "strongly_entangling"],
                     n_qubits=[4, 6], n_layers=[2, 3], lr=[0.05, 0.1])

Every axis takes a list; anything you leave out keeps its default. The result ranks
the configurations on the metric the task deserves — imbalance-aware for
classification — scores all of them on identical folds, and refuses to call a winner
when the gap is inside the fold-to-fold spread, exactly as
:func:`~qmlkit.baselines.baseline` does.

**The part no other library can do.** A grid search normally trains everything and
sorts the results, so a configuration that *cannot work* still costs a full fit — and
then sits in the table looking merely unlucky. Before fitting anything, this runs
:func:`~qmlkit.diagnostics.diagnose` on the assembled model and drops the ones with an
error-level finding: an ansatz whose weights cannot move the state, a re-uploading
block that collapses to one frequency, a model with no trainable parameters. They are
reported as *pruned, with the reason*, not silently dropped and not quietly ranked
last.

On a grid where a third of the points are unlearnable that is a third of the compute,
and — more to the point — a third of the rows in your results table that would
otherwise have been noise you might have read as signal.

**It tells you the cost first.** ``qk.search(..., dry_run=True)`` returns the plan
without fitting anything: how many configurations, how many fits, and how many
circuits, so a sweep that would take a week is something you find out in a second.
"""

from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.ansatz.library import Ansatz, get_ansatz
from qmlkit.encoding.feature_maps import (
    AngleFeatureMap,
    FeatureMap,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
)

__all__ = [
    "search",
    "SearchResult",
    "SearchRow",
    "register_feature_map",
    "list_feature_maps",
    "AXES",
]

Array = npt.NDArray[Any]

#: Feature maps reachable by name. Registering adds to every later search.
_FEATURE_MAPS: dict[str, Callable[..., FeatureMap]] = {
    "angle": lambda n_qubits, **kw: AngleFeatureMap(n_qubits, entangle=n_qubits > 1, **kw),
    "angle-plain": lambda n_qubits, **kw: AngleFeatureMap(n_qubits, entangle=False, **kw),
    "z": lambda n_qubits, **kw: ZFeatureMap(n_qubits, **kw),
    "zz": lambda n_qubits, **kw: ZZFeatureMap(n_qubits, **kw),
    "pauli": lambda n_qubits, **kw: PauliFeatureMap(n_qubits, **kw),
}


def register_feature_map(name: str, factory: Callable[..., FeatureMap]) -> None:
    """Make a feature map reachable by name, here and in every later search."""
    _FEATURE_MAPS[name] = factory


def list_feature_maps() -> tuple[str, ...]:
    return tuple(sorted(_FEATURE_MAPS))


#: Every axis a search understands, with the value used when you do not give one.
#: A name outside this set is refused with a suggestion rather than silently ignored,
#: because a typo'd axis in a grid search is a sweep that quietly varies nothing.
#: ``n_qubits`` defaults to ``None``, meaning "read it off the data" — a fixed default
#: would ask a 2-feature dataset for 4 qubits, and a feature pipeline can drop columns
#: but cannot invent them.
AXES: dict[str, Any] = {
    "n_qubits": None,
    "ansatz": "hardware_efficient",
    "n_layers": 2,
    "feature_map": "angle",
    "lr": 0.05,
    "epochs": 30,
    "batch_size": 256,
    "class_weight": None,
    "focal_gamma": 0.0,
    "grad_method": "auto",
    "shots": None,
}


@dataclass(frozen=True)
class SearchRow:
    """One configuration's outcome: a score, or the reason it was never fitted."""

    config: dict[str, Any]
    mean: float = float("nan")
    std: float = float("nan")
    fold_scores: tuple[float, ...] = ()
    pruned: str = ""
    seconds: float = 0.0
    #: what diagnose() said about this configuration, whether or not it was pruned
    findings: tuple[str, ...] = ()
    fitted: bool = True

    @property
    def ran(self) -> bool:
        return self.fitted and not self.pruned

    def label(self, axes: Sequence[str]) -> str:
        """Only the axes that actually varied, so the table stays readable."""
        return " ".join(f"{k}={self.config[k]}" for k in axes)


@dataclass(frozen=True)
class SearchResult:
    """Every configuration on identical folds, best first."""

    task: str
    metric: str
    n_samples: int
    n_folds: int
    varied: tuple[str, ...]
    rows: tuple[SearchRow, ...] = ()
    notes: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def ran(self) -> tuple[SearchRow, ...]:
        return tuple(r for r in self.rows if r.ran)

    @property
    def pruned(self) -> tuple[SearchRow, ...]:
        """Configurations skipped *because a diagnosis said so*.

        Not the same as "did not run": a dry run fits nothing and prunes only what it
        would have pruned for real, so these two have to stay distinguishable.
        """
        return tuple(r for r in self.rows if r.pruned)

    @property
    def best(self) -> SearchRow | None:
        return max(self.ran, key=lambda r: r.mean) if self.ran else None

    @property
    def best_config(self) -> dict[str, Any]:
        row = self.best
        if row is None:
            raise ValueError("no configuration ran; see .pruned for why")
        return dict(row.config)

    @property
    def verdict(self) -> str:
        """Whether the winner is actually a winner, or is inside the noise."""
        if not self.ran:
            if any(r.pruned for r in self.rows) and not all(r.pruned for r in self.rows):
                return "nothing was fitted; the rest were pruned before fitting"
            if not any(r.pruned for r in self.rows):
                return f"nothing was fitted: {len(self.rows)} configurations would run"
            return "every configuration was pruned before fitting"
        ordered = sorted(self.ran, key=lambda r: -r.mean)
        best = ordered[0]
        if len(ordered) == 1:
            return f"only one configuration ran: {best.label(self.varied)} at {best.mean:.3f}"
        runner_up = ordered[1]
        gap = best.mean - runner_up.mean
        spread = float(np.hypot(best.std, runner_up.std))
        if spread > 0 and gap < spread:
            return (
                f"{best.label(self.varied)} leads at {best.mean:.3f}, but by {gap:.3f} over "
                f"{runner_up.label(self.varied)} -- inside the fold spread ({spread:.3f}), so "
                "this grid has not separated them"
            )
        return (
            f"{best.label(self.varied)} wins at {best.mean:.3f}, ahead of "
            f"{runner_up.label(self.varied)} by {gap:.3f} (fold spread {spread:.3f})"
        )

    def model(self, **overrides: Any) -> Any:
        """A fresh, unfitted model built from the winning configuration."""
        return _build_model(self.best_config | overrides, self.extras["n_classes"])

    def __str__(self) -> str:
        width = max((len(r.label(self.varied)) for r in self.rows), default=10)
        split = "stratified" if self.task == "classification" else "shuffled"
        lines = [
            f"{self.task}  |  {self.metric}  |  {self.n_folds}-fold {split}  "
            f"|  n={self.n_samples}  |  {len(self.rows)} configurations"
        ]
        for row in sorted(self.rows, key=lambda r: (not r.ran, -r.mean)):
            label = row.label(self.varied)
            flags = f"   [{', '.join(row.findings)}]" if row.findings else ""
            if row.ran:
                lines.append(f"  {label:<{width}}  {row.mean: .3f} +/- {row.std:.3f}{flags}")
            elif row.pruned:
                lines.append(f"  {label:<{width}}  pruned: {row.pruned}")
            else:
                lines.append(f"  {label:<{width}}  would run{flags}")
        lines.append(f"\n{self.verdict}")
        lines.extend(f"note: {note}" for note in self.notes)
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# building a model from one point of the grid
# --------------------------------------------------------------------------- #
def _resolve_ansatz(spec: Any, n_qubits: int, n_layers: int) -> Ansatz:
    if isinstance(spec, Ansatz):
        return spec
    if callable(spec):
        built: Ansatz = spec(n_qubits, n_layers)
        return built
    try:
        return get_ansatz(spec, n_qubits=n_qubits, n_layers=n_layers)
    except TypeError:  # a registered factory that does not take n_layers
        return get_ansatz(spec, n_qubits=n_qubits)


def _resolve_feature_map(spec: Any, n_qubits: int) -> FeatureMap:
    if isinstance(spec, FeatureMap):
        return spec
    if callable(spec):
        built: FeatureMap = spec(n_qubits)
        return built
    if spec not in _FEATURE_MAPS:
        from qmlkit.utils.errors import unknown

        raise unknown("feature map", spec, _FEATURE_MAPS)
    return _FEATURE_MAPS[spec](n_qubits)


def _build_model(config: dict[str, Any], n_classes: int) -> Any:
    from qmlkit.nn.models import VQC, VQRegressor

    n_qubits = int(config["n_qubits"])
    ansatz = _resolve_ansatz(config["ansatz"], n_qubits, int(config["n_layers"]))
    feature_map = _resolve_feature_map(config["feature_map"], n_qubits)
    shared = dict(
        n_features=n_qubits,
        n_qubits=n_qubits,
        ansatz=ansatz,
        feature_map=feature_map,
        grad_method=config["grad_method"],
        shots=config["shots"],
        seed=0,
    )
    if n_classes:
        return VQC(
            n_classes=n_classes,
            class_weight=config["class_weight"],
            focal_gamma=float(config["focal_gamma"]),
            **shared,
        )
    return VQRegressor(**shared)


# --------------------------------------------------------------------------- #
# the search
# --------------------------------------------------------------------------- #
def search(
    X: Any,
    y: Any,
    task: str = "auto",
    cv: int = 3,
    metric: str | None = None,
    seed: int | None = 0,
    prune: str | Sequence[str] = "error",
    max_configs: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
    **axes: Any,
) -> SearchResult:
    """Sweep every tunable axis, skipping the configurations that cannot work.

    Parameters
    ----------
    X, y:
        The data. Folds are stratified for classification and identical for every
        configuration, so the table compares models rather than splits.
    cv:
        Folds per configuration.
    metric:
        What to rank on. Defaults to the task's primary metric, which is
        imbalance-aware for classification.
    prune:
        Which diagnoses stop a configuration being fitted, as a named level or an
        explicit list of finding codes.

        ``"error"`` (the default) skips only what cannot work at all: no trainable
        weights, a re-uploading block collapsed to one frequency, input features the
        circuit discards. ``"untrainable"`` also skips flat gradients and circuits that
        never entangle. ``"warning"`` additionally skips merely *wasteful* ones — dead
        or unmeasurable weights — which is aggressive: ``hardware_efficient`` carries an
        ``UNMEASURABLE_WEIGHTS`` finding by construction, so that level can empty a
        grid. ``"none"`` fits everything.

        Every configuration is diagnosed regardless of this setting, and its findings
        are printed beside its score — a point that scores well *and* carries
        ``DEAD_WEIGHTS`` is worth seeing as exactly that.
    max_configs:
        Sample this many points from the grid instead of taking all of them. The
        sample is seeded, so it is reproducible.
    dry_run:
        Build and prune the grid, report what it would cost, and fit nothing.
    **axes:
        Any key in :data:`AXES`, given a list of values (a bare value is treated as a
        one-element list). An unrecognised axis is an error with a suggestion, because
        a typo'd axis name is a sweep that silently varies nothing.

    Examples
    --------
    >>> import qmlkit as qk
    >>> X, y = qk.datasets.make_moons(n_samples=40, seed=0)
    >>> plan = qk.search(X, y, n_layers=[1, 2], cv=2, dry_run=True)
    >>> len(plan.rows)
    2
    """
    from qmlkit.baselines import _infer_task
    from qmlkit.imbalance import stratified_folds

    data = np.atleast_2d(np.asarray(X, dtype=float))
    target = np.asarray(y).ravel()
    if data.shape[0] != target.size:
        raise ValueError(f"X has {data.shape[0]} rows but y has {target.size}")

    unknown_axes = set(axes) - set(AXES)
    if unknown_axes:
        from qmlkit.utils.errors import unknown

        raise unknown("search axis", sorted(unknown_axes)[0], AXES)

    n_features = data.shape[1]
    resolved = _infer_task(target) if task == "auto" else task
    metric = metric or ("balanced_accuracy" if resolved == "classification" else "r2")
    n_classes = int(np.unique(target).size) if resolved == "classification" else 0

    grid = {k: (list(v) if isinstance(v, list | tuple) else [v]) for k, v in axes.items()}
    too_wide = [int(q) for q in grid.get("n_qubits", []) if q is not None and int(q) > n_features]
    if too_wide:
        raise ValueError(
            f"n_qubits={too_wide} exceeds the {n_features} feature(s) in X. A feature "
            "pipeline reduces columns; it cannot invent them. Either widen the data or "
            f"keep n_qubits at or below {n_features}."
        )
    varied = tuple(k for k, v in grid.items() if len(v) > 1) or tuple(grid) or ("n_qubits",)

    notes: list[str] = []
    if "n_qubits" in grid:
        width = AXES["n_qubits"]
    else:
        # one qubit per feature, capped: past six the sweep costs more than it teaches
        width = min(n_features, 6)
        notes.append(
            f"n_qubits was not given, so it is {width} "
            + ("one per feature" if width == n_features else f"capped from {n_features}")
        )

    defaults = {**AXES, "n_qubits": width}
    full = {**{k: [v] for k, v in defaults.items()}, **grid}
    names = list(full)
    points = [dict(zip(names, combo, strict=True)) for combo in itertools.product(*full.values())]

    rng = np.random.default_rng(seed)
    if max_configs is not None and max_configs < len(points):
        chosen = rng.choice(len(points), size=max_configs, replace=False)
        notes.append(f"sampled {max_configs} of {len(points)} grid points")
        points = [points[i] for i in sorted(chosen.tolist())]

    # ---- prune before fitting, and say why -------------------------------- #
    if isinstance(prune, str):
        if prune not in _PRUNE_LEVELS:
            from qmlkit.utils.errors import unknown

            raise unknown("prune level", prune, _PRUNE_LEVELS)
        blocking = _PRUNE_LEVELS[prune]
    else:  # an explicit list of finding codes
        blocking = tuple(prune)

    rows: list[SearchRow] = []
    runnable: list[tuple[dict[str, Any], tuple[str, ...]]] = []
    for config in points:
        codes, reason = _diagnose_config(config, n_classes, blocking)
        if reason:
            rows.append(SearchRow(config, pruned=reason, findings=codes, fitted=False))
        else:
            runnable.append((config, codes))

    if dry_run:
        notes.append(
            f"dry run: {len(runnable)} configurations x {cv} folds = {len(runnable) * cv} fits"
            + (f", {len(rows)} pruned before fitting" if rows else "")
        )
        rows.extend(SearchRow(c, findings=codes, fitted=False) for c, codes in runnable)
        return SearchResult(
            resolved, metric, int(target.size), cv, varied, tuple(rows), tuple(notes),
            {"n_classes": n_classes},
        )

    # ---- identical folds for every configuration -------------------------- #
    if resolved == "classification":
        folds = stratified_folds(target, n_folds=cv, seed=seed)
    else:
        order = rng.permutation(data.shape[0])
        folds = [(np.setdiff1d(order, chunk), chunk) for chunk in np.array_split(order, cv)]

    for index, (config, codes) in enumerate(runnable, start=1):
        started = time.perf_counter()
        try:
            scores = _score_config(config, data, target, folds, resolved, metric, n_classes)
        except Exception as exc:  # a broken point must not lose the rest of the table
            rows.append(SearchRow(config, pruned=f"failed: {exc}", findings=codes, fitted=False))
            continue
        row = SearchRow(
            config,
            float(np.mean(scores)),
            float(np.std(scores)),
            tuple(scores),
            seconds=time.perf_counter() - started,
            findings=codes,
        )
        rows.append(row)
        if verbose:
            print(
                f"  [{index}/{len(runnable)}] {row.label(varied)}"
                f"  {row.mean:.3f} +/- {row.std:.3f}  ({row.seconds:.0f}s)",
                flush=True,
            )

    if rows and any(not r.ran and "dry run" not in r.pruned for r in rows):
        n_pruned = sum(1 for r in rows if not r.ran)
        notes.append(
            f"{n_pruned} of {len(rows)} configurations were skipped before fitting; "
            "they are listed with their reason rather than ranked last"
        )
    if resolved == "classification":
        from qmlkit.imbalance import imbalance_ratio

        ratio = imbalance_ratio(target)
        if ratio >= 1.5:
            notes.append(
                f"classes are {ratio:.1f}:1, so the ranking metric is {metric}; "
                "class_weight=['balanced'] is worth putting on the grid"
            )
    return SearchResult(
        resolved, metric, int(target.size), cv, varied, tuple(rows), tuple(notes),
        {"n_classes": n_classes, "folds": folds},
    )


#: Named prune levels, as sets of finding *codes* rather than severities.
#:
#: Severity alone is the wrong axis here. ``DEAD_WEIGHTS`` and ``UNMEASURABLE_WEIGHTS``
#: are warnings that mean "wasteful" — a model carrying them still learns, and
#: ``hardware_efficient`` carries one by construction, so pruning on severity would drop
#: the library's own default and leave an empty grid. ``FLAT_GRADIENTS`` is also a
#: warning and means "cannot learn". So the levels name codes.
_CANNOT_WORK = ("NO_TRAINABLE_PARAMETERS", "ENCODING_COMMUTES", "INPUTS_UNUSED")
_CANNOT_LEARN = (*_CANNOT_WORK, "FLAT_GRADIENTS", "NO_ENTANGLEMENT")
_WASTEFUL = (*_CANNOT_LEARN, "DEAD_WEIGHTS", "UNMEASURABLE_WEIGHTS")
_PRUNE_LEVELS: dict[str, tuple[str, ...]] = {
    "none": (),
    "error": _CANNOT_WORK,
    "untrainable": _CANNOT_LEARN,
    "warning": _WASTEFUL,
}


def _first_sentence(text: str, limit: int = 96) -> str:
    """The first sentence, capped — and without splitting a decimal in half.

    Naively splitting on ``"."`` turns "gradient variance 1.14e-33" into "gradient
    variance 1", so the break has to be a period *followed by a space*.
    """
    for i in range(len(text) - 1):
        if text[i] == "." and text[i + 1] == " ":
            text = text[: i + 1]
            break
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _diagnose_config(
    config: dict[str, Any], n_classes: int, blocking: tuple[str, ...]
) -> tuple[tuple[str, ...], str]:
    """``(finding codes, reason to skip)``. Never raises — a build failure is a reason.

    Every configuration is diagnosed whether or not it is pruned, because a point that
    *scores* well while carrying ``DEAD_WEIGHTS`` is worth seeing as exactly that. Only
    findings at or above ``levels`` stop it being fitted.
    """
    from qmlkit.diagnostics import diagnose

    try:
        model = _build_model(config, n_classes)
    except Exception as exc:  # noqa: BLE001 - the reason is the point
        return (), f"could not be built ({exc})"
    try:
        report = diagnose(model)
    except Exception:  # noqa: BLE001 - a diagnosis that fails must not block a fit
        return (), ""
    codes = tuple(f.code for f in report)
    stoppers = [f for f in report if f.code in blocking]
    if not stoppers:
        return codes, ""
    worst = stoppers[0]
    return codes, f"{worst.code}: {_first_sentence(worst.message)}"


def _score_config(
    config: dict[str, Any],
    X: Array,
    y: Array,
    folds: list[tuple[Array, Array]],
    task: str,
    metric: str,
    n_classes: int,
) -> list[float]:
    """One configuration across the folds. The pipeline is refitted per fold."""
    from qmlkit import evaluate
    from qmlkit.encoding.pipeline import FeaturePipeline

    n_qubits = int(config["n_qubits"])
    scores = []
    for train, test in folds:
        pipeline = FeaturePipeline(n_qubits=n_qubits).fit(X[train])
        model = _build_model(config, n_classes)
        model.fit(
            pipeline.transform(X[train]),
            y[train],
            epochs=int(config["epochs"]),
            lr=float(config["lr"]),
            batch_size=int(config["batch_size"]),
        )
        predicted = model.predict(pipeline.transform(X[test]))
        got = (
            evaluate.classification(y[test], predicted)
            if task == "classification"
            else evaluate.regression(y[test], predicted)
        )
        scores.append(float(got[metric]))
    return scores
