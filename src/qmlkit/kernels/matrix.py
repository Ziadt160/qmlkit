r"""Gram matrices, PSD repair, and the diagnostics that say whether any of it will work.

Filling a Gram matrix is the expensive half of a quantum kernel method: ``m(m-1)/2``
circuit evaluations for a training set of size ``m``, since the diagonal is exactly
1 and the matrix is symmetric. :func:`kernel_matrix` exploits both.

**Shot noise breaks positive semi-definiteness.** Every entry is an estimate, so the
estimated Gram matrix can have small negative eigenvalues even though the true one
cannot — and an SVM solver will either refuse it or return nonsense. The repair
functions here project back onto the PSD cone.

**Exponential concentration is the real limit.** As the feature map widens, distinct
inputs produce states whose overlaps all collapse toward the same value, at a rate
around ``2^-n``. Resolving that against shot noise costs about ``4^n`` shots. The
diagnostics report both, because a kernel method that has concentrated looks like a
model that simply does not learn.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.execute import BackendLike
from qmlkit.encoding.feature_maps import FeatureMap
from qmlkit.kernels.estimators import fidelity_kernel

__all__ = [
    "kernel_matrix",
    "square_kernel_matrix",
    "QuantumKernel",
    "is_psd",
    "min_eigenvalue",
    "threshold_matrix",
    "displace_matrix",
    "flip_matrix",
    "closest_psd_matrix",
    "center_kernel",
    "normalize_kernel",
    "target_alignment",
    "kernel_shot_cost",
    "kernel_spread",
    "shots_to_resolve",
    "concentration_report",
    "geometric_difference",
]

KernelFn = Callable[[Sequence[float], Sequence[float]], float]


# --------------------------------------------------------------------------- #
# building the matrix
# --------------------------------------------------------------------------- #
def square_kernel_matrix(
    X: npt.NDArray[Any], kernel: KernelFn, assume_unit_diagonal: bool = True
) -> npt.NDArray[Any]:
    """Symmetric Gram matrix, evaluating only the upper triangle.

    ``m(m-1)/2`` evaluations instead of ``m^2``. ``assume_unit_diagonal`` sets
    ``k(x, x) = 1`` without measuring it, which is exact for a fidelity kernel and
    saves ``m`` more evaluations.
    """
    rows = np.atleast_2d(np.asarray(X, dtype=float))
    m = rows.shape[0]
    out = np.eye(m) if assume_unit_diagonal else np.zeros((m, m))
    if not assume_unit_diagonal:
        for i in range(m):
            out[i, i] = kernel(rows[i], rows[i])
    for i in range(m):
        for j in range(i + 1, m):
            out[i, j] = out[j, i] = kernel(rows[i], rows[j])
    return out


def kernel_matrix(
    X: npt.NDArray[Any], Y: npt.NDArray[Any] | None = None, kernel: KernelFn | None = None
) -> npt.NDArray[Any]:
    """Gram matrix of ``X`` against ``Y`` (or itself, exploiting symmetry)."""
    if kernel is None:
        raise ValueError("kernel_matrix needs a kernel callable")
    if Y is None:
        return square_kernel_matrix(X, kernel)
    a = np.atleast_2d(np.asarray(X, dtype=float))
    b = np.atleast_2d(np.asarray(Y, dtype=float))
    return np.array([[kernel(u, v) for v in b] for u in a], dtype=float)


class QuantumKernel:
    """A feature map, as a kernel you can hand to any kernel method.

    kernel = QuantumKernel(qk.ZZFeatureMap(2))
    K = kernel(X)                 # training Gram matrix
    K_test = kernel(X_test, X)    # rectangular, test against train
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        estimator: str = "inversion",
        shots: int | None = None,
        backend: BackendLike = None,
        bandwidth: float = 1.0,
        seed: int | None = None,
        cache: bool = True,
    ) -> None:
        self.feature_map = feature_map
        self.estimator = estimator
        self.shots = shots
        self.backend = backend
        self.bandwidth = bandwidth
        self.seed = seed
        self.cache = cache
        self._cache: dict[tuple, float] = {}
        self._evaluations = 0

    # ------------------------------------------------------------------------
    def _estimate(self, x: npt.NDArray[Any], xp: npt.NDArray[Any]) -> float:
        from qmlkit.kernels.estimators import hadamard_test, swap_test_kernel

        fns = {
            "inversion": fidelity_kernel,
            "swap": swap_test_kernel,
            "hadamard": lambda fm, a, b, **kw: hadamard_test(fm, a, b, **kw) ** 2,
        }
        try:
            fn = fns[self.estimator]
        except KeyError:
            raise ValueError(
                f"unknown estimator {self.estimator!r}; expected inversion, swap or hadamard"
            ) from None
        self._evaluations += 1
        return float(
            fn(
                self.feature_map,
                x,
                xp,
                shots=self.shots,
                backend=self.backend,
                seed=self.seed,
            )
        )

    def evaluate(self, x: Sequence[float], xp: Sequence[float]) -> float:
        """One kernel entry, with the bandwidth rescaling applied."""
        a = self.bandwidth * np.asarray(x, dtype=float)
        b = self.bandwidth * np.asarray(xp, dtype=float)
        if not self.cache:
            return self._estimate(a, b)
        # the kernel is symmetric, so k(a, b) and k(b, a) share one cache entry --
        # which halves the evaluations a rectangular test matrix needs
        ka, kb = tuple(np.round(a, 12)), tuple(np.round(b, 12))
        key = (ka, kb) if ka <= kb else (kb, ka)
        if key not in self._cache:
            self._cache[key] = self._estimate(a, b)
        return self._cache[key]

    def __call__(self, X: npt.NDArray[Any], Y: npt.NDArray[Any] | None = None) -> npt.NDArray[Any]:
        return kernel_matrix(X, Y, self.evaluate)

    @property
    def n_evaluations(self) -> int:
        """Circuits actually run — cache hits do not count."""
        return self._evaluations

    def reset(self) -> None:
        self._cache.clear()
        self._evaluations = 0

    def __repr__(self) -> str:
        return (
            f"QuantumKernel({self.feature_map!r}, estimator={self.estimator!r}, "
            f"shots={self.shots}, bandwidth={self.bandwidth})"
        )


# --------------------------------------------------------------------------- #
# PSD repair — shot noise puts a Gram matrix outside the cone
# --------------------------------------------------------------------------- #
def min_eigenvalue(K: npt.NDArray[Any]) -> float:
    return float(np.linalg.eigvalsh(np.asarray(K, dtype=float)).min())


def is_psd(K: npt.NDArray[Any], tol: float = 1e-9) -> bool:
    """True if every eigenvalue is non-negative to within ``tol``."""
    return min_eigenvalue(K) >= -abs(tol)


def threshold_matrix(K: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Clip negative eigenvalues to zero — the standard projection onto the cone."""
    vals, vecs = np.linalg.eigh(np.asarray(K, dtype=float))
    return (vecs * np.clip(vals, 0.0, None)) @ vecs.T


def displace_matrix(K: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Shift the whole spectrum up until it is non-negative.

    Keeps every eigenvector's relative weight, unlike thresholding, at the cost of
    inflating the diagonal.
    """
    arr = np.asarray(K, dtype=float)
    low = min_eigenvalue(arr)
    return arr if low >= 0 else arr + abs(low) * np.eye(arr.shape[0])


def flip_matrix(K: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Take the absolute value of each eigenvalue."""
    vals, vecs = np.linalg.eigh(np.asarray(K, dtype=float))
    return (vecs * np.abs(vals)) @ vecs.T


def closest_psd_matrix(K: npt.NDArray[Any], method: str = "threshold") -> npt.NDArray[Any]:
    """Nearest PSD matrix by the named method."""
    fns = {"threshold": threshold_matrix, "displace": displace_matrix, "flip": flip_matrix}
    try:
        return fns[method](K)
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; expected threshold, displace or flip"
        ) from None


def center_kernel(K: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Centre the induced feature space at the origin."""
    arr = np.asarray(K, dtype=float)
    m = arr.shape[0]
    ones = np.ones((m, m)) / m
    return arr - ones @ arr - arr @ ones + ones @ arr @ ones


def normalize_kernel(K: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Rescale to a unit diagonal — the cosine of the feature-space angle."""
    arr = np.asarray(K, dtype=float)
    d = np.sqrt(np.clip(np.diag(arr), 1e-15, None))
    return arr / np.outer(d, d)


# --------------------------------------------------------------------------- #
# is this kernel any good?
# --------------------------------------------------------------------------- #
def target_alignment(K: npt.NDArray[Any], y: npt.NDArray[Any], rescale: bool = True) -> float:
    """Kernel-target alignment: how much the Gram matrix looks like the labels.

    ``<K, yy^T>_F / (||K||_F ||yy^T||_F)`` in ``[-1, 1]``. This is the objective you
    maximise to *train* a feature map, and a cheap way to compare candidates without
    fitting an SVM to each.
    """
    arr = np.asarray(K, dtype=float)
    labels = np.asarray(y, dtype=float).ravel()
    if rescale and set(np.unique(labels)) <= {0.0, 1.0}:
        labels = 2 * labels - 1  # {0,1} -> {-1,+1}
    target = np.outer(labels, labels)
    denom = np.linalg.norm(arr) * np.linalg.norm(target)
    return float(np.sum(arr * target) / denom) if denom > 0 else 0.0


def kernel_shot_cost(m: int, shots: int, include_diagonal: bool = False) -> int:
    """Total shots to fill an ``m x m`` Gram matrix."""
    entries = m * (m - 1) // 2 + (m if include_diagonal else 0)
    return entries * shots


def kernel_spread(n_qubits: int) -> float:
    """Rough off-diagonal spread of a concentrated kernel: ``2^-n``."""
    return float(2.0**-n_qubits)


def shots_to_resolve(n_qubits: int) -> int:
    """Shots needed to see a ``2^-n`` signal above ``1/sqrt(N)`` noise: about ``4^n``."""
    return int(4**n_qubits)


def concentration_report(K: npt.NDArray[Any], n_qubits: int, shots: int | None = None) -> dict:
    """Is this Gram matrix telling you anything, or has it concentrated?

    A concentrated kernel has near-identical off-diagonal entries: every pair of
    inputs looks equally similar, so no model built on it can separate them.
    """
    arr = np.asarray(K, dtype=float)
    off = arr[~np.eye(arr.shape[0], dtype=bool)]
    spread = float(off.std())
    noise = float(np.sqrt(0.25 / shots)) if shots else 0.0
    return {
        "off_diagonal_mean": float(off.mean()),
        "off_diagonal_std": spread,
        "predicted_spread": kernel_spread(n_qubits),
        "shot_noise": noise,
        "resolvable": bool(spread > noise) if shots else True,
        "shots_to_resolve": shots_to_resolve(n_qubits),
        "min_eigenvalue": min_eigenvalue(arr),
        "is_psd": is_psd(arr),
    }


def geometric_difference(k_quantum: npt.NDArray[Any], k_classical: npt.NDArray[Any]) -> float:
    r"""``g(K_c, K_q)`` — the statistic that says whether quantum *could* help.

    Large means the two kernels induce genuinely different geometries, so a
    separation is at least possible. Small means the classical kernel already sees
    everything the quantum one does, and there is nothing to gain (Huang et al. 2021).
    """
    kq = np.asarray(k_quantum, dtype=float)
    kc = np.asarray(k_classical, dtype=float)
    if kq.shape != kc.shape:
        raise ValueError(f"kernels have different shapes: {kq.shape} vs {kc.shape}")
    n = kq.shape[0]
    sqrt_kq = _sqrtm_psd(kq)
    kc_inv = np.linalg.pinv(kc + 1e-12 * np.eye(n))
    m = sqrt_kq @ kc_inv @ sqrt_kq
    return float(np.sqrt(np.linalg.norm(m, ord=2) * n))


def _sqrtm_psd(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
