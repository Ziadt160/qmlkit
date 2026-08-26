"""Getting real data into the range and width an encoding needs.

Two problems every quantum model hits before any quantum step happens: features
arrive on arbitrary scales when rotations want radians, and there are usually more
features than qubits.

These are *preprocessing*, not classical baselines — no model is fitted here, and
there is no sklearn dependency. The PCA reduction is a plain SVD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["AngleScaler", "to_angle_range", "reduce_to_qubits", "PCAReducer"]


def to_angle_range(
    x: npt.NDArray[Any],
    lo: float = 0.0,
    hi: float = 2 * np.pi,
    data_min: npt.NDArray[Any] | None = None,
    data_max: npt.NDArray[Any] | None = None,
) -> npt.NDArray[Any]:
    """Rescale features into an angle window, per column.

    Fit the range on training data and reuse it on test data by passing
    ``data_min``/``data_max`` explicitly — otherwise each call rescales to its own
    extremes, which silently makes train and test incomparable. A constant column
    maps to the middle of the window rather than dividing by zero.
    """
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    lo_v = np.asarray(arr.min(axis=0) if data_min is None else data_min, dtype=float)
    hi_v = np.asarray(arr.max(axis=0) if data_max is None else data_max, dtype=float)
    span = hi_v - lo_v
    flat = span == 0
    span = np.where(flat, 1.0, span)
    unit = (arr - lo_v) / span
    unit = np.where(flat, 0.5, unit)
    return (lo + unit * (hi - lo)).reshape(np.asarray(x).shape)


@dataclass
class AngleScaler:
    """Fit-then-transform angle scaling, so train and test share one range."""

    lo: float = 0.0
    hi: float = 2 * np.pi
    data_min: npt.NDArray[Any] | None = None
    data_max: npt.NDArray[Any] | None = None

    def fit(self, x: npt.NDArray[Any]) -> AngleScaler:
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        self.data_min = arr.min(axis=0)
        self.data_max = arr.max(axis=0)
        return self

    def transform(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.data_min is None or self.data_max is None:
            raise ValueError("AngleScaler must be fitted before transform()")
        return to_angle_range(x, self.lo, self.hi, self.data_min, self.data_max)

    def fit_transform(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.fit(x).transform(x)


@dataclass
class PCAReducer:
    """Project features onto their leading principal components, via SVD.

    Angle encoding needs one qubit per feature, so a 64-feature dataset needs 64
    qubits — usually out of reach. Reducing first is the ordinary way through, and
    ``explained_variance_ratio_`` says how much you gave up doing it.
    """

    n_components: int
    mean_: npt.NDArray[Any] | None = None
    components_: npt.NDArray[Any] | None = None
    explained_variance_ratio_: npt.NDArray[Any] | None = None

    def fit(self, x: npt.NDArray[Any]) -> PCAReducer:
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        if self.n_components > arr.shape[1]:
            raise ValueError(
                f"cannot reduce {arr.shape[1]} features to {self.n_components} components"
            )
        self.mean_ = arr.mean(axis=0)
        centred = arr - self.mean_
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        variance = singular**2
        total = variance.sum()
        self.components_ = vt[: self.n_components]
        self.explained_variance_ratio_ = (
            variance[: self.n_components] / total if total > 0 else np.zeros(self.n_components)
        )
        return self

    def transform(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.components_ is None or self.mean_ is None:
            raise ValueError("PCAReducer must be fitted before transform()")
        arr = np.atleast_2d(np.asarray(x, dtype=float))
        return (arr - self.mean_) @ self.components_.T

    def fit_transform(self, x: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self.fit(x).transform(x)


def reduce_to_qubits(
    x: npt.NDArray[Any],
    n_qubits: int,
    method: str = "pca",
    to_angles: bool = True,
    lo: float = 0.0,
    hi: float = 2 * np.pi,
) -> npt.NDArray[Any]:
    """Reduce a feature matrix to ``n_qubits`` columns, ready for angle encoding.

    ``method="pca"`` keeps the leading principal components; ``method="truncate"``
    keeps the first ``n_qubits`` columns unchanged, which is only sensible when the
    features are already ordered by importance.
    """
    arr = np.atleast_2d(np.asarray(x, dtype=float))
    if n_qubits < 1:
        raise ValueError("n_qubits must be at least 1")
    if method == "pca":
        reduced = arr if arr.shape[1] == n_qubits else PCAReducer(n_qubits).fit_transform(arr)
    elif method == "truncate":
        if arr.shape[1] < n_qubits:
            raise ValueError(f"only {arr.shape[1]} features available, need {n_qubits}")
        reduced = arr[:, :n_qubits]
    else:
        raise ValueError(f"unknown method {method!r}; expected 'pca' or 'truncate'")
    return to_angle_range(reduced, lo, hi) if to_angles else reduced
