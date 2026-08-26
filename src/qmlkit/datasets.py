"""Datasets for benchmarking quantum models.

Small, self-contained, no downloads, no sklearn. The important one is
:func:`ad_hoc_data`, which is *constructed* to be separable by a specific quantum
feature map and not by a classical kernel — so it distinguishes a working
implementation from one that only appears to work.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "ad_hoc_data",
    "bars_and_stripes",
    "make_moons",
    "make_blobs",
    "make_parity",
    "make_circles",
    "train_test_split",
]


def ad_hoc_data(
    n_samples: int = 40,
    n_features: int = 2,
    gap: float = 0.3,
    seed: int | None = None,
    scale: float = 2 * np.pi,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """The Havlíček-style separable-by-construction dataset.

    Labels come from the sign of a hidden observable measured on a ZZ-feature-mapped
    state, so the ZZ kernel separates it by construction while classical kernels
    struggle. ``gap`` discards points near the boundary, which makes the separation
    clean enough to be a real check.
    """
    from qmlkit.core.execute import expectation
    from qmlkit.core.observables import PauliString
    from qmlkit.encoding.feature_maps import ZZFeatureMap

    rng = np.random.default_rng(seed)
    fmap = ZZFeatureMap(n_features, reps=2)
    witness = PauliString(tuple((q, "Z") for q in range(n_features)))

    xs: list[npt.NDArray[Any]] = []
    ys: list[int] = []
    attempts = 0
    while len(xs) < n_samples and attempts < 200 * n_samples:
        attempts += 1
        x = rng.uniform(0, scale, n_features)
        value = float(expectation(fmap.build(x), witness))
        if abs(value) < gap:
            continue  # too close to the boundary to label cleanly
        xs.append(x)
        ys.append(1 if value > 0 else 0)
    if len(xs) < n_samples:  # pragma: no cover - only with an extreme gap
        raise ValueError(f"could not find {n_samples} samples outside a gap of {gap}")
    return np.array(xs), np.array(ys)


def bars_and_stripes(size: int = 2, seed: int | None = None) -> npt.NDArray[Any]:
    """Every bars-and-stripes pattern on a ``size x size`` grid, flattened.

    The standard target distribution for a quantum circuit Born machine: a sparse,
    highly structured subset of all bitstrings.
    """
    if size < 1:
        raise ValueError("size must be at least 1")
    patterns = set()
    for bits in itertools.product([0, 1], repeat=size):
        grid = np.tile(np.array(bits)[:, None], (1, size))  # bars
        patterns.add(tuple(grid.ravel()))
        patterns.add(tuple(grid.T.ravel()))  # stripes
    out = np.array(sorted(patterns), dtype=int)
    if seed is not None:
        np.random.default_rng(seed).shuffle(out)
    return out


def make_moons(
    n_samples: int = 100, noise: float = 0.1, seed: int | None = None, to_angles: bool = True
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Two interleaving half-circles — not linearly separable."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, np.pi, n_out)
    t_in = np.linspace(0, np.pi, n_in)
    x = np.vstack(
        [
            np.column_stack([np.cos(t_out), np.sin(t_out)]),
            np.column_stack([1 - np.cos(t_in), 1 - np.sin(t_in) - 0.5]),
        ]
    )
    x += rng.normal(0, noise, x.shape)
    y = np.array([0] * n_out + [1] * n_in)
    return (_to_angles(x) if to_angles else x), y


def make_circles(
    n_samples: int = 100,
    noise: float = 0.08,
    factor: float = 0.5,
    seed: int | None = None,
    to_angles: bool = True,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """One circle inside another — needs a nonlinear boundary."""
    rng = np.random.default_rng(seed)
    n_out = n_samples // 2
    n_in = n_samples - n_out
    t_out = np.linspace(0, 2 * np.pi, n_out, endpoint=False)
    t_in = np.linspace(0, 2 * np.pi, n_in, endpoint=False)
    x = np.vstack(
        [
            np.column_stack([np.cos(t_out), np.sin(t_out)]),
            factor * np.column_stack([np.cos(t_in), np.sin(t_in)]),
        ]
    )
    x += rng.normal(0, noise, x.shape)
    y = np.array([0] * n_out + [1] * n_in)
    return (_to_angles(x) if to_angles else x), y


def make_blobs(
    n_samples: int = 100,
    centers: int = 2,
    spread: float = 0.4,
    n_features: int = 2,
    seed: int | None = None,
    to_angles: bool = True,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Gaussian clusters — the easy baseline every model should pass."""
    rng = np.random.default_rng(seed)
    middles = rng.uniform(-2, 2, (centers, n_features))
    per = n_samples // centers
    xs, ys = [], []
    for c in range(centers):
        count = per if c < centers - 1 else n_samples - per * (centers - 1)
        xs.append(rng.normal(middles[c], spread, (count, n_features)))
        ys.extend([c] * count)
    x = np.vstack(xs)
    return (_to_angles(x) if to_angles else x), np.array(ys)


def make_parity(
    n_samples: int = 100, n_features: int = 4, seed: int | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    """Label is the parity of the bits — the classic hard case for shallow models."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, (n_samples, n_features))
    y = bits.sum(axis=1) % 2
    return bits * np.pi, y  # 0 or pi, already an angle


def _to_angles(x: npt.NDArray[Any], lo: float = 0.0, hi: float = np.pi) -> npt.NDArray[Any]:
    from qmlkit.encoding.scaling import to_angle_range

    return to_angle_range(x, lo, hi)


def train_test_split(
    X: npt.NDArray[Any], y: npt.NDArray[Any], test_size: float = 0.3, seed: int | None = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Shuffle and split. Here so a quickstart needs no extra dependency."""
    rows = np.atleast_2d(np.asarray(X))
    labels = np.asarray(y).ravel()
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    cut = int(round(len(rows) * (1 - test_size)))
    tr, te = order[:cut], order[cut:]
    return rows[tr], rows[te], labels[tr], labels[te]
