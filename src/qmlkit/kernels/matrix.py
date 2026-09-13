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
from qmlkit.kernels.estimators import fidelity_kernel, hadamard_test
from qmlkit.progress import task as progress_task
from qmlkit.utils.errors import unknown

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
    # this is the path that takes hours on a few hundred points, so it is the one
    # worth being able to watch
    with progress_task("kernel gram", m * (m - 1) // 2) as tracked:
        for i in range(m):
            for j in range(i + 1, m):
                out[i, j] = out[j, i] = kernel(rows[i], rows[j])
                tracked.advance()
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
    with progress_task("kernel gram", a.shape[0] * b.shape[0]) as tracked:
        out = np.empty((a.shape[0], b.shape[0]), dtype=float)
        for i, u in enumerate(a):
            for j, v in enumerate(b):
                out[i, j] = kernel(u, v)
                tracked.advance()
    return out


def _hadamard_kernel(
    fmap: FeatureMap, x: Sequence[float], xp: Sequence[float], **kwargs: Any
) -> float:
    r"""``|<phi(x')|phi(x)>|^2`` from a Hadamard test — **two** circuits, not one.

    A Hadamard test measures one *component* of a complex overlap: the real part, or
    with an ``Sdg`` on the ancilla, the imaginary one. The kernel is the squared
    modulus, so it needs both: :math:`\mathrm{Re}^2 + \mathrm{Im}^2`.

    Squaring the real part alone is a different quantity that agrees with the kernel
    whenever the overlap happens to be real — which is every product state, every
    real-amplitude feature map, and every ``K(x, x) = 1`` diagonal entry. It also
    leaves a Gram matrix that is still symmetric, still PSD and still unit-diagonal,
    so nothing downstream can notice it is wrong.
    """
    real = hadamard_test(fmap, x, xp, part="real", **kwargs)
    imag = hadamard_test(fmap, x, xp, part="imag", **kwargs)
    return float(real * real + imag * imag)


#: Circuits one pair costs, per estimator. The Hadamard test reads one component of
#: the overlap per circuit, so the modulus is two runs; ``n_evaluations`` says so.
_CIRCUITS_PER_PAIR = {"inversion": 1, "swap": 1, "hadamard": 2}


def _hardware_pairs(m: int, k: int | None) -> int:
    """Pair circuits a device would run for one Gram matrix.

    A square matrix is symmetric with a unit diagonal, so only the strict upper
    triangle costs anything; a rectangular one costs every entry.
    """
    return m * (m - 1) // 2 if k is None else m * k


class QuantumKernel:
    """A feature map, as a kernel you can hand to any kernel method.

    kernel = QuantumKernel(qk.ZZFeatureMap(2))
    K = kernel(X)                 # training Gram matrix
    K_test = kernel(X_test, X)    # rectangular, test against train

    Arguments
    ---------
    estimator
        How the overlap is measured. ``"inversion"`` (the default) runs the
        compute-uncompute circuit and reads the all-zeros probability; ``"swap"``
        uses a swap test; ``"hadamard"`` runs **two** Hadamard tests and adds the
        squares of the real and imaginary parts, because one Hadamard test measures
        one component of a complex overlap and the kernel is its modulus. All three
        agree on a simulator — :attr:`n_evaluations` is what differs, and on a device
        so do the width and the connectivity each one needs.
    shots
        ``None`` reads the exact probability. A budget samples it, which is what a
        device does — and a sampled kernel is not positive semi-definite by
        construction, so pair it with :func:`threshold_matrix`.
    bandwidth
        **The first thing to try when a kernel has concentrated.** Every feature
        vector is scaled by this before encoding, so it sets how far apart two points
        are in the feature map rather than in the data. At the default ``1.0`` a
        fidelity kernel over a wide register drives every off-diagonal entry toward
        the same small number — every pair of points looks equally dissimilar, the
        Gram matrix approaches the identity, and no amount of training recovers what
        the encoding threw away. Shrinking the bandwidth (``0.1``-``0.5`` is the usual
        range) compresses the data into a smaller region of state space and pulls the
        off-diagonals back apart. :func:`concentration_report` measures whether you
        have the problem, and ``qk.diagnose(K)`` names it as ``KERNEL_CONCENTRATED``.
        The alternative fix is a projected kernel, which survives width by measuring
        local reduced states instead — see the kernels tutorial for when each applies.
    cache
        Memoises pair evaluations, which matters because a Gram matrix asks for the
        same circuit many times. ``n_evaluations`` counts the circuits actually run.
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
        self._cache: dict[tuple[Any, ...], float] = {}
        #: psi(x) per row, for the state-overlap path — see :meth:`_state_gram`. Kept
        #: separate from the pair cache above because the two are keyed differently
        #: and a row cache is what makes a rectangular test matrix cheap.
        self._state_cache: dict[tuple[Any, ...], npt.NDArray[Any]] = {}
        self._evaluations = 0
        self._hardware_circuits = 0

    # ------------------------------------------------------------------------
    def _estimate(self, x: npt.NDArray[Any], xp: npt.NDArray[Any]) -> float:
        from qmlkit.kernels.estimators import swap_test_kernel

        fns: dict[str, Callable[..., float]] = {
            "inversion": fidelity_kernel,
            "swap": swap_test_kernel,
            "hadamard": _hadamard_kernel,
        }
        try:
            fn = fns[self.estimator]
        except KeyError:
            raise unknown("estimator", self.estimator, ("inversion", "swap", "hadamard")) from None
        self._evaluations += _CIRCUITS_PER_PAIR[self.estimator]
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

    @staticmethod
    def _cache_key(a: npt.NDArray[Any], b: npt.NDArray[Any]) -> tuple[Any, ...]:
        """One entry per unordered pair, so k(a, b) and k(b, a) share it.

        Both the pair-at-a-time and the batched path build the key here, so a matrix
        computed one way is reused by the other.
        """
        ka, kb = tuple(np.round(a, 12)), tuple(np.round(b, 12))
        return (ka, kb) if ka <= kb else (kb, ka)

    def evaluate(self, x: Sequence[float], xp: Sequence[float]) -> float:
        """One kernel entry, with the bandwidth rescaling applied."""
        a = self.bandwidth * np.asarray(x, dtype=float)
        b = self.bandwidth * np.asarray(xp, dtype=float)
        if not self.cache:
            return self._estimate(a, b)
        # the kernel is symmetric, so k(a, b) and k(b, a) share one cache entry --
        # which halves the evaluations a rectangular test matrix needs
        key = self._cache_key(a, b)
        if key not in self._cache:
            self._cache[key] = self._estimate(a, b)
        return self._cache[key]

    # ------------------------------------------------------------------------
    def _batched_gram(
        self, X: npt.NDArray[Any], Y: npt.NDArray[Any] | None
    ) -> npt.NDArray[Any] | None:
        """The whole Gram matrix in one call, or ``None`` if that is not available.

        A compute-uncompute kernel is ``U(x) U(x')†`` — one circuit *structure*, with
        the two feature vectors' angles as its parameters. Every entry of the Gram
        matrix is therefore the same circuit at a different parameter vector, which is
        exactly what a batched backend evaluates in one pass. The per-pair loop was
        throwing that away.

        Returns ``None`` — so the caller falls back to the pair-at-a-time path —
        when the kernel is sampled (a shot budget is spent per circuit either way),
        when the estimator is not the inversion test, or when the backend cannot hand
        back a state.
        """
        from qmlkit.core.backends.registry import get_backend

        if self.estimator != "inversion" or self.shots is not None:
            return None
        backend = get_backend(self.backend)
        if not backend.supports_statevector:
            return None
        fmap = self.feature_map
        if not hasattr(fmap, "build_parametric"):  # pragma: no cover - defensive
            return None

        n_angles = fmap.n_angles
        spec = fmap.build_parametric(offset=0).compose(
            fmap.build_parametric(offset=n_angles).adjoint(), param_offset=0
        )
        rows = self.bandwidth * np.atleast_2d(np.asarray(X, dtype=float))
        columns = rows if Y is None else self.bandwidth * np.atleast_2d(np.asarray(Y, dtype=float))
        row_angles = np.stack([fmap.angles(r) for r in rows])
        column_angles = row_angles if Y is None else np.stack([fmap.angles(c) for c in columns])

        # a square matrix is symmetric with a unit diagonal, so only the strict upper
        # triangle is worth evaluating -- m(m-1)/2 circuits instead of m^2
        m, k = rows.shape[0], columns.shape[0]
        if Y is None:
            pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        else:
            pairs = [(i, j) for i in range(m) for j in range(k)]
        if not pairs:
            return np.eye(m) if Y is None else np.zeros((m, k))

        # the cache still applies: batching is about how the misses are evaluated, not
        # about re-running work already done. A rectangular test matrix against the
        # training set is mostly cache hits, and losing them would cost more than
        # batching gains.
        keys = [self._cache_key(rows[i], columns[j]) for i, j in pairs] if self.cache else None
        misses = (
            [n for n, key in enumerate(keys) if key not in self._cache]
            if keys is not None
            else list(range(len(pairs)))
        )

        if misses:
            thetas = np.stack(
                [
                    np.concatenate([row_angles[pairs[n][0]], column_angles[pairs[n][1]]])
                    for n in misses
                ]
            )
            states = backend.statevector_batch(spec, thetas)
            values = np.abs(states[:, 0]) ** 2  # P(all zeros) *is* the kernel
            self._evaluations += len(misses)
            if keys is not None:
                for n, value in zip(misses, values, strict=True):
                    self._cache[keys[n]] = float(value)
        else:
            values = np.empty(0)

        resolved = dict(zip(misses, values.tolist(), strict=True))
        out = np.eye(m) if Y is None else np.zeros((m, k))
        for n, (i, j) in enumerate(pairs):
            value = self._cache[keys[n]] if keys is not None else resolved[n]
            out[i, j] = float(value)
            if Y is None:
                out[j, i] = out[i, j]
        return out

    def _state_gram(
        self, X: npt.NDArray[Any], Y: npt.NDArray[Any] | None
    ) -> npt.NDArray[Any] | None:
        """The whole Gram matrix from ``m`` states instead of ``m(m-1)/2`` circuits.

        The inversion test reads ``P(0...0)`` of ``U(x')† U(x)|0>``, which *is*
        ``|<psi(x')|psi(x)>|**2``. On a backend that hands back a state there is no
        reason to build the composed circuit at all: evaluate each row once and let
        BLAS form every overlap. That turns the cost from quadratic in the dataset to
        linear, and the matrix product is not where the time goes.

        Measured on a ZZ feature map, reps=2, full entanglement — composed pairs
        against this:

        | qubits | rows | pairs | composed | states | |
        |---|---|---|---|---|---|
        | 4 | 24 | 276 | 24.2 ms | 2.3 ms | 10.4x |
        | 4 | 64 | 2016 | 166.6 ms | 4.9 ms | 34.0x |
        | 6 | 64 | 2016 | 929.4 ms | 19.8 ms | 46.9x |
        | 8 | 64 | 2016 | 4911.3 ms | 81.0 ms | 60.6x |

        **This is a simulator-only shortcut, and it changes what a circuit count
        means.** A device cannot hand back a state, so hardware still pays the
        quadratic inversion test; :attr:`n_evaluations` reports what actually ran
        here, and :attr:`circuits_on_hardware` reports what the same Gram matrix
        would have cost on a device. Estimating a hardware run from the first number
        would under-count it by a factor of ``(m-1)/2``.

        Returns ``None`` — falling back to :meth:`_batched_gram` — under the same
        conditions as that method, since sampling, a different estimator or a
        density-matrix backend each break the identity above or the access to a state.
        """
        from qmlkit.core.backends.registry import get_backend

        if self.estimator != "inversion" or self.shots is not None:
            return None
        backend = get_backend(self.backend)
        if not backend.supports_statevector:
            return None
        fmap = self.feature_map
        if not hasattr(fmap, "build_parametric") or not hasattr(fmap, "angles"):
            return None

        spec = fmap.build_parametric(offset=0)
        rows = self.bandwidth * np.atleast_2d(np.asarray(X, dtype=float))
        row_states = self._states(backend, spec, rows)
        if Y is None:
            column_states = row_states
        else:
            columns = self.bandwidth * np.atleast_2d(np.asarray(Y, dtype=float))
            column_states = self._states(backend, spec, columns)

        out = np.abs(row_states.conj() @ column_states.T) ** 2
        if Y is None:
            # |<psi|psi>|**2 is 1 by construction; say so exactly rather than to 1e-16
            np.fill_diagonal(out, 1.0)
        self._hardware_circuits += _hardware_pairs(len(rows), None if Y is None else len(columns))
        return np.asarray(out, dtype=float)

    def _states(
        self, backend: Any, spec: Any, rows: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """``psi(x)`` for each row, reusing any this kernel has already evaluated.

        The cache here is per *row*, not per pair, which is the whole point: a
        rectangular test matrix against a training set re-encodes the training rows
        for every test point under the pair-keyed cache and encodes them once under
        this one.
        """
        fmap = self.feature_map
        angles = np.stack([fmap.angles(r) for r in rows])
        if not self.cache:
            self._evaluations += len(angles)
            return np.asarray(backend.statevector_batch(spec, angles))

        keys = [tuple(np.round(a, 12)) for a in angles]
        misses = [n for n, key in enumerate(keys) if key not in self._state_cache]
        if misses:
            computed = backend.statevector_batch(spec, angles[misses])
            self._evaluations += len(misses)
            for n, state in zip(misses, computed, strict=True):
                self._state_cache[keys[n]] = np.asarray(state)
        return np.stack([self._state_cache[key] for key in keys])

    def __call__(self, X: npt.NDArray[Any], Y: npt.NDArray[Any] | None = None) -> npt.NDArray[Any]:
        overlaps = self._state_gram(X, Y)
        if overlaps is not None:
            return overlaps
        batched = self._batched_gram(X, Y)
        if batched is not None:
            return batched
        return kernel_matrix(X, Y, self.evaluate)

    @property
    def n_evaluations(self) -> int:
        """Circuits actually run — cache hits do not count."""
        return self._evaluations

    @property
    def circuits_on_hardware(self) -> int:
        """What the same Gram matrices would have cost on a device.

        :attr:`n_evaluations` is what ran. This is what a device would have run for
        the same answers, because the state-overlap shortcut in :meth:`_state_gram`
        needs a statevector and hardware has none. Budget from this number, not from
        the other one.
        """
        return self._hardware_circuits or self._evaluations

    def reset(self) -> None:
        self._cache.clear()
        self._state_cache.clear()
        self._evaluations = 0
        self._hardware_circuits = 0

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
        raise unknown("method", method, ("threshold", "displace", "flip")) from None


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


def concentration_report(
    K: npt.NDArray[Any], n_qubits: int, shots: int | None = None
) -> dict[str, Any]:
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
    r"""``g(K_C || K_Q)`` — the statistic that says whether quantum *could* help.

    .. math::

        g = \sqrt{\lVert \sqrt{K_Q}\, K_C^{-1} \sqrt{K_Q} \rVert_\infty}

    **The number to compare it against is** ``sqrt(N)``, for ``N`` samples — that is
    the threshold in Huang et al. (2021), and it is the caller's to apply. A ``g``
    well below ``sqrt(N)`` says the classical kernel already sees everything the
    quantum one does, so no separation is available whatever a later accuracy table
    claims. A ``g`` at or above it says a separation is *possible*, not that one
    exists.

    Identical kernels give exactly ``1``. The statistic is scale-sensitive, so it
    carries the paper's meaning only when the two kernels are normalised alike —
    which any two with a unit diagonal are, a fidelity kernel and an RBF kernel
    included. ``center_kernel`` or ``normalize_kernel`` will put an odd one right.
    """
    kq = np.asarray(k_quantum, dtype=float)
    kc = np.asarray(k_classical, dtype=float)
    if kq.shape != kc.shape:
        raise ValueError(f"kernels have different shapes: {kq.shape} vs {kc.shape}")
    n = kq.shape[0]
    sqrt_kq = _sqrtm_psd(kq)
    kc_inv = np.linalg.pinv(kc + 1e-12 * np.eye(n))
    m = sqrt_kq @ kc_inv @ sqrt_kq
    return float(np.sqrt(np.linalg.norm(m, ord=2)))


def _sqrtm_psd(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    vals, vecs = np.linalg.eigh(a)
    return (vecs * np.sqrt(np.clip(vals, 0.0, None))) @ vecs.T
