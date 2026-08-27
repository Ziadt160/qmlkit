"""q-means — Lloyd's algorithm with a quantum distance.

The unsupervised gap. k-means is entirely defined by one operation, "how far apart
are these two points", so replacing that with a quantum kernel distance is the whole
algorithm:

.. math::  d(x, x')^2 = 2\\bigl(1 - k(x, x')\\bigr)

for a normalised kernel. Everything else — assign, recentre, repeat — is Lloyd's, and
is deliberately unchanged so that any difference in the result is attributable to the
distance and nothing else.

The feature map is the argument, exactly as in :class:`~qmlkit.QSVC`: a clustering
method built on a kernel *is* its embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.execute import BackendLike
from qmlkit.encoding.feature_maps import FeatureMap
from qmlkit.kernels.matrix import QuantumKernel

__all__ = ["QMeans", "QMeansResult"]


@dataclass
class QMeansResult:
    labels: npt.NDArray[Any]
    centroids: npt.NDArray[Any]
    inertia: float
    n_iterations: int
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"QMeansResult(k={len(self.centroids)}, inertia={self.inertia:.6f}, "
            f"iterations={self.n_iterations})"
        )


class QMeans:
    """k-means where the distance comes from a quantum kernel.

    Parameters
    ----------
    n_clusters
        ``k``.
    feature_map
        The embedding the distance is measured in. This is the only quantum part,
        and swapping it is the entire experiment.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        feature_map: FeatureMap | None = None,
        max_iterations: int = 50,
        tol: float = 1e-6,
        shots: int | None = None,
        backend: BackendLike = None,
        seed: int | None = None,
    ) -> None:
        if n_clusters < 1:
            raise ValueError("n_clusters must be at least 1")
        self.n_clusters = n_clusters
        self.feature_map = feature_map
        self.max_iterations = max_iterations
        self.tol = tol
        self.shots = shots
        self.backend = backend
        self.seed = seed
        self.centroids_: npt.NDArray[Any] | None = None
        self.labels_: npt.NDArray[Any] | None = None

    # ---------------------------------------------------------------- distance --
    def _kernel(self, n_features: int) -> QuantumKernel:
        from qmlkit.encoding.feature_maps import AngleFeatureMap

        fmap = self.feature_map or AngleFeatureMap(n_features, entangle=n_features > 1)
        return QuantumKernel(fmap, shots=self.shots, backend=self.backend, seed=self.seed)

    def distances(self, X: npt.NDArray[Any], centroids: npt.NDArray[Any]) -> npt.NDArray[Any]:
        r"""``(n_samples, k)`` of :math:`2(1 - k(x, c))`.

        A kernel with unit diagonal induces a genuine squared distance this way, so
        the assignment step below is the ordinary one — no special-casing.
        """
        kernel = self._kernel(X.shape[1])
        gram = kernel(np.asarray(X, dtype=float), np.asarray(centroids, dtype=float))
        return 2.0 * (1.0 - gram)

    # -------------------------------------------------------------------- fit --
    def fit(self, X: npt.NDArray[Any], seed: int | None = None) -> QMeansResult:
        data = np.asarray(X, dtype=float)
        if len(data) < self.n_clusters:
            raise ValueError(f"cannot form {self.n_clusters} clusters from {len(data)} samples")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        # k-means++ style start: distinct rows, so two centroids cannot collide
        chosen = rng.choice(len(data), size=self.n_clusters, replace=False)
        centroids = data[chosen].copy()

        history: list[float] = []
        labels = np.zeros(len(data), dtype=int)
        iteration = 0
        for iteration in range(self.max_iterations):  # noqa: B007 - used after the loop
            d = self.distances(data, centroids)
            labels = np.argmin(d, axis=1)
            inertia = float(d[np.arange(len(data)), labels].sum())
            history.append(inertia)

            moved = centroids.copy()
            for k in range(self.n_clusters):
                members = data[labels == k]
                if len(members):
                    moved[k] = members.mean(axis=0)
                # An empty cluster keeps its centroid rather than drifting or
                # collapsing onto another; duplicated points make that reachable.
            shift = float(np.abs(moved - centroids).max())
            centroids = moved
            if shift < self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return QMeansResult(
            labels=labels,
            centroids=centroids,
            inertia=history[-1],
            n_iterations=iteration + 1,
            history=history,
        )

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if self.centroids_ is None:
            raise ValueError("QMeans must be fitted before predicting")
        return np.argmin(self.distances(np.asarray(X, dtype=float), self.centroids_), axis=1)

    def fit_predict(self, X: npt.NDArray[Any], seed: int | None = None) -> npt.NDArray[Any]:
        return self.fit(X, seed=seed).labels

    def __repr__(self) -> str:
        name = type(self.feature_map).__name__ if self.feature_map else "AngleFeatureMap"
        return f"QMeans(n_clusters={self.n_clusters}, feature_map={name})"
