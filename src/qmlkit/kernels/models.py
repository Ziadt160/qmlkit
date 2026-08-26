"""Kernel models: sklearn estimators, a trainable kernel, and projected kernels.

The division of labour a quantum kernel method rests on: the **quantum** part fills
the Gram matrix, and the **classical** part solves a convex problem on it. That
means `QSVC` is a real SVM — same convergence guarantees, same solver — with one
matrix supplied from a circuit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from qmlkit.core.execute import BackendLike
from qmlkit.encoding.feature_maps import FeatureMap
from qmlkit.info import reduced_dm
from qmlkit.kernels.matrix import QuantumKernel, closest_psd_matrix, is_psd, target_alignment

__all__ = [
    "QSVC",
    "QSVR",
    "NearestFidelityClassifier",
    "TrainableKernel",
    "projected_kernel_matrix",
    "rkhs_model",
]


def _require_sklearn(what: str) -> Any:
    try:
        import sklearn.svm as svm
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"{what} wraps scikit-learn's solver, which is an optional extra:\n"
            "    pip install 'qmlkit[sklearn]'"
        ) from exc
    return svm


class _KernelEstimator:
    """Shared plumbing: fill the Gram matrix, hand it to a precomputed-kernel solver."""

    _svm: Any

    def __init__(
        self,
        feature_map: FeatureMap,
        shots: int | None = None,
        backend: BackendLike = None,
        bandwidth: float = 1.0,
        estimator: str = "inversion",
        repair_psd: str | None = "threshold",
        seed: int | None = None,
        **solver_kwargs: Any,
    ) -> None:
        self.kernel = QuantumKernel(
            feature_map,
            estimator=estimator,
            shots=shots,
            backend=backend,
            bandwidth=bandwidth,
            seed=seed,
        )
        self.repair_psd = repair_psd
        self.solver_kwargs = solver_kwargs
        self.X_train_: np.ndarray | None = None

    def _gram(self, X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        K = self.kernel(X, Y)
        # only a square training matrix needs (or admits) a PSD repair
        if Y is None and self.repair_psd and not is_psd(K):
            K = closest_psd_matrix(K, self.repair_psd)
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> _KernelEstimator:
        self.X_train_ = np.atleast_2d(np.asarray(X, dtype=float))
        self._svm.fit(self._gram(self.X_train_), np.asarray(y).ravel())
        return self

    def _check_fitted(self) -> np.ndarray:
        if self.X_train_ is None:
            raise ValueError(f"{type(self).__name__} must be fitted before use")
        return self.X_train_

    def predict(self, X: np.ndarray) -> np.ndarray:
        train = self._check_fitted()
        return np.asarray(self._svm.predict(self.kernel(np.atleast_2d(X), train)))

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        train = self._check_fitted()
        return np.asarray(self._svm.decision_function(self.kernel(np.atleast_2d(X), train)))

    @property
    def n_circuit_evaluations(self) -> int:
        return self.kernel.n_evaluations


class QSVC(_KernelEstimator):
    """Quantum-kernel support vector classifier.

    clf = QSVC(qk.ZZFeatureMap(2)).fit(X, y)
    clf.score(X_test, y_test)
    """

    def __init__(self, feature_map: FeatureMap, C: float = 1.0, **kwargs: Any) -> None:
        super().__init__(feature_map, **kwargs)
        svm = _require_sklearn("QSVC")
        self._svm = svm.SVC(kernel="precomputed", C=C, **self.solver_kwargs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


class QSVR(_KernelEstimator):
    """Quantum-kernel support vector regressor."""

    def __init__(
        self, feature_map: FeatureMap, C: float = 1.0, epsilon: float = 0.1, **kwargs: Any
    ) -> None:
        super().__init__(feature_map, **kwargs)
        svm = _require_sklearn("QSVR")
        self._svm = svm.SVR(kernel="precomputed", C=C, epsilon=epsilon, **self.solver_kwargs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2."""
        pred = np.asarray(self.predict(X)).ravel()
        truth = np.asarray(y, dtype=float).ravel()
        ss_res = float(((truth - pred) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class NearestFidelityClassifier:
    """Classify by fidelity to each class centroid — no solver, no sklearn.

    The simplest quantum classifier there is: encode every training point, average
    within each class, and predict whichever class anchor a new point overlaps most.
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        shots: int | None = None,
        backend: BackendLike = None,
    ) -> None:
        self.feature_map = feature_map
        self.shots = shots
        self.backend = backend
        self.classes_: np.ndarray | None = None
        self.anchors_: dict[Any, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> NearestFidelityClassifier:
        from qmlkit.core.execute import statevector

        rows = np.atleast_2d(np.asarray(X, dtype=float))
        labels = np.asarray(y).ravel()
        self.classes_ = np.unique(labels)
        for c in self.classes_:
            states = [
                statevector(self.feature_map.build(r), backend=self.backend)
                for r in rows[labels == c]
            ]
            mean = np.mean(states, axis=0)
            norm = np.linalg.norm(mean)
            self.anchors_[c] = mean / norm if norm > 1e-12 else mean
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        from qmlkit.core.execute import statevector

        if self.classes_ is None:
            raise ValueError("NearestFidelityClassifier must be fitted before use")
        rows = np.atleast_2d(np.asarray(X, dtype=float))
        out = []
        for r in rows:
            psi = statevector(self.feature_map.build(r), backend=self.backend)
            scores = {c: abs(np.vdot(a, psi)) ** 2 for c, a in self.anchors_.items()}
            out.append(max(scores, key=lambda c: scores[c]))
        return np.array(out)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


class TrainableKernel:
    """Train the *feature map itself* by maximising kernel-target alignment.

    A fixed feature map is a guess. Alignment gives a differentiable score for how
    well a kernel matches the labels, so the embedding's own parameters can be
    optimised before any classifier is fitted — usually a bigger win than tuning the
    classifier afterwards.
    """

    def __init__(
        self,
        feature_map_factory: Any,
        n_params: int,
        shots: int | None = None,
        backend: BackendLike = None,
    ) -> None:
        self.factory = feature_map_factory
        self.n_params = n_params
        self.shots = shots
        self.backend = backend
        self.params_: np.ndarray | None = None
        self.history_: list[float] = []

    def alignment(self, params: Sequence[float], X: np.ndarray, y: np.ndarray) -> float:
        kernel = QuantumKernel(self.factory(params), shots=self.shots, backend=self.backend)
        return target_alignment(kernel(X), y)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_iterations: int = 40,
        theta0: Sequence[float] | None = None,
        seed: int | None = None,
    ) -> TrainableKernel:
        """Maximise alignment with SPSA — two evaluations per step, any parameter count."""
        from qmlkit.gradients.spsa import minimize_spsa

        rng = np.random.default_rng(seed)
        start = (
            np.asarray(theta0, dtype=float)
            if theta0 is not None
            else rng.uniform(0.5, 1.5, self.n_params)
        )

        def loss(p: np.ndarray) -> float:
            return -self.alignment(p, X, y)  # minimise the negative

        best, history = minimize_spsa(loss, start, n_iterations=n_iterations, seed=seed)
        self.params_ = best
        self.history_ = [-h for h in history]
        return self

    def kernel(self) -> QuantumKernel:
        if self.params_ is None:
            raise ValueError("TrainableKernel must be fitted before use")
        return QuantumKernel(self.factory(self.params_), shots=self.shots, backend=self.backend)


def projected_kernel_matrix(
    feature_map: FeatureMap,
    X: np.ndarray,
    gamma: float = 1.0,
    backend: BackendLike = None,
) -> np.ndarray:
    r"""Projected quantum kernel — the standard answer to exponential concentration.

    Instead of a *global* fidelity, compare the **one-qubit reduced density
    matrices**:

    .. math::  k(x, x') = \exp\left(-\gamma \sum_i \|\rho_i(x) - \rho_i(x')\|_F^2\right)

    Global overlaps concentrate as the register widens — every pair of inputs ends
    up looking equally similar, and the kernel stops carrying information. Local
    reduced states do not, so this stays informative where the fidelity kernel has
    already collapsed (Huang et al. 2021).
    """
    from qmlkit.core.execute import statevector

    rows = np.atleast_2d(np.asarray(X, dtype=float))
    n = feature_map.n_qubits
    # one reduced density matrix per qubit per sample
    rdms = []
    for r in rows:
        psi = statevector(feature_map.build(r), backend=backend)
        rdms.append([reduced_dm(psi, [q], n) for q in range(n)])

    m = len(rows)
    out = np.ones((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            dist = sum(float(np.linalg.norm(rdms[i][q] - rdms[j][q], "fro") ** 2) for q in range(n))
            out[i, j] = out[j, i] = float(np.exp(-gamma * dist))
    return out


def rkhs_model(
    alphas: Sequence[float],
    anchors: np.ndarray,
    x: Sequence[float],
    kernel: Any,
) -> float:
    """``f(x) = sum_i alpha_i k(x_i, x)`` — a kernel model is a weighted similarity sum."""
    a = np.asarray(alphas, dtype=float)
    pts = np.atleast_2d(np.asarray(anchors, dtype=float))
    return float(sum(ai * kernel(p, x) for ai, p in zip(a, pts, strict=True)))
