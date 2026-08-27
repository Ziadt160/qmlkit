"""Getting a real dataset onto a small number of qubits, once and reproducibly.

Almost every quantum model starts the same way: standardise the features, reduce them
to as many columns as you have qubits, and scale those into rotation angles. Done by
hand that is three objects to keep in sync and one easy mistake — fitting the reducer
on the test set — so it is one object here.

    pipeline = FeaturePipeline(n_qubits=4).fit(X_train)
    Z_train, Z_test = pipeline.transform(X_train), pipeline.transform(X_test)

``fit`` sees only the training data, and ``transform`` reuses exactly what it learned.
:attr:`FeaturePipeline.explained_variance_` reports what the reduction cost, because a
model that never saw 20% of the variance is not underperforming — it was never shown
the data.

Everything here is duck-typed to scikit-learn's estimator protocol (``get_params`` /
``set_params`` / ``fit`` / ``transform``), so it drops into ``Pipeline`` and
``GridSearchCV`` without qmlkit depending on scikit-learn.
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.encoding.scaling import AngleScaler, PCAReducer

__all__ = ["FeaturePipeline", "SklearnCompatible"]


class SklearnCompatible:
    """``get_params`` / ``set_params``, read off the constructor signature.

    scikit-learn duck-types: ``clone``, ``Pipeline`` and ``GridSearchCV`` need these
    two methods, not a base class. Implementing them directly is what lets a qmlkit
    estimator sit in a scikit-learn workflow while scikit-learn stays an *optional*
    dependency — which matters, because the NumPy backend is meant to work alone.

    The one rule this imposes: an ``__init__`` parameter must be stored on an
    attribute of the same name, unchanged.
    """

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        names = [
            p
            for p in inspect.signature(type(self).__init__).parameters
            if p not in ("self", "args", "kwargs")
        ]
        out: dict[str, Any] = {}
        for name in names:
            value = getattr(self, name, None)
            out[name] = value
            if deep and hasattr(value, "get_params"):
                for key, sub in value.get_params(deep=True).items():
                    out[f"{name}__{key}"] = sub
        return out

    def __sklearn_tags__(self) -> Any:
        """Hand scikit-learn its own tags object, built only when it asks.

        scikit-learn 1.6 stopped duck-typing on ``_estimator_type`` and now requires
        this method. The ``Tags`` dataclass it wants is version-specific, so rather
        than reconstructing it — and breaking on the next release — this borrows one
        from a throwaway estimator carrying the right mixin. The import is inside the
        method so that scikit-learn stays an optional dependency: nothing calls this
        unless scikit-learn is already running.
        """
        from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

        mixin = {"classifier": ClassifierMixin, "regressor": RegressorMixin}.get(
            getattr(self, "_estimator_type", "")
        )
        if mixin is None:
            return BaseEstimator().__sklearn_tags__()

        class _Probe(mixin, BaseEstimator):  # type: ignore[misc, valid-type]
            pass

        return _Probe().__sklearn_tags__()

    def set_params(self, **params: Any) -> SklearnCompatible:
        valid = self.get_params(deep=True)
        nested: dict[str, dict[str, Any]] = {}
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"invalid parameter {key!r} for {type(self).__name__}; "
                    f"valid ones are {sorted(k for k in valid if '__' not in k)}"
                )
            if "__" in key:
                owner, _, rest = key.partition("__")
                nested.setdefault(owner, {})[rest] = value
            else:
                setattr(self, key, value)
        for owner, sub in nested.items():
            getattr(self, owner).set_params(**sub)
        return self


class FeaturePipeline(SklearnCompatible):
    """Standardise, reduce to ``n_qubits`` columns, and scale into rotation angles.

    Parameters
    ----------
    n_qubits
        How many columns to come out with — one rotation angle per qubit.
    method
        ``"pca"`` keeps the leading principal components. ``"truncate"`` keeps the
        first ``n_qubits`` columns, which is only honest when the features are
        already ordered by importance.
    standardize
        Centre and scale to unit variance first. PCA without this is dominated by
        whichever feature happens to be measured in the largest units.
    angle_range
        Where the output lands. The default ``(0, 2pi)`` uses the full period of a
        rotation; a narrower range trades expressiveness for a gentler landscape.
    """

    def __init__(
        self,
        n_qubits: int,
        method: str = "pca",
        standardize: bool = True,
        angle_range: tuple[float, float] = (0.0, 2 * np.pi),
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        if method not in ("pca", "truncate"):
            raise ValueError(f"method must be 'pca' or 'truncate', got {method!r}")
        self.n_qubits = n_qubits
        self.method = method
        self.standardize = standardize
        self.angle_range = angle_range
        self.mean_: npt.NDArray[Any] | None = None
        self.scale_: npt.NDArray[Any] | None = None
        self.reducer_: PCAReducer | None = None
        self.scaler_: AngleScaler | None = None
        self.explained_variance_: float | None = None

    # ------------------------------------------------------------------- fit --
    def fit(self, X: npt.NDArray[Any], y: Any = None) -> FeaturePipeline:
        """Learn every step from the training data alone."""
        data = np.atleast_2d(np.asarray(X, dtype=float))
        if data.shape[1] < self.n_qubits:
            raise ValueError(
                f"cannot map {data.shape[1]} features onto {self.n_qubits} qubits; "
                "reduction can only remove columns, not invent them"
            )

        if self.standardize:
            self.mean_ = data.mean(axis=0)
            spread = data.std(axis=0)
            self.scale_ = np.where(spread > 1e-12, spread, 1.0)  # constant columns survive
            data = (data - self.mean_) / self.scale_
        else:
            self.mean_ = self.scale_ = None

        if self.method == "pca" and data.shape[1] != self.n_qubits:
            self.reducer_ = PCAReducer(self.n_qubits).fit(data)
            data = self.reducer_.transform(data)
            self.explained_variance_ = float(np.sum(self.reducer_.explained_variance_ratio_))
        else:
            self.reducer_ = None
            data = data[:, : self.n_qubits]
            self.explained_variance_ = None

        low, high = self.angle_range
        self.scaler_ = AngleScaler(lo=low, hi=high).fit(data)
        return self

    def transform(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Apply the fitted steps. Never re-fits — that is the whole point."""
        if self.scaler_ is None:
            raise ValueError("FeaturePipeline must be fitted before transform()")
        data = np.atleast_2d(np.asarray(X, dtype=float))
        if self.standardize and self.mean_ is not None and self.scale_ is not None:
            data = (data - self.mean_) / self.scale_
        if self.reducer_ is not None:
            data = self.reducer_.transform(data)
        else:
            data = data[:, : self.n_qubits]
        return self.scaler_.transform(data)

    def fit_transform(self, X: npt.NDArray[Any], y: Any = None) -> npt.NDArray[Any]:
        return self.fit(X, y).transform(X)

    def __repr__(self) -> str:
        kept = (
            ""
            if self.explained_variance_ is None
            else f", variance kept {self.explained_variance_:.1%}"
        )
        return f"FeaturePipeline(n_qubits={self.n_qubits}, method={self.method!r}{kept})"
