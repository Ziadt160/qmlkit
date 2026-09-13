"""Running independent work on more than one core must not change the answer.

The whole value of `n_jobs` is that it is free. A parallel option that perturbs a
result is worse than no option at all, because the perturbation is invisible: the
score is still in range, the table still sorts, and the number is different from the
one the same call produced yesterday.

Writing these tests found that `qk.search(seed=0)` was **already** not reproducible,
before any threading - `HybridModel` shuffled its batches off torch's global RNG and
initialised its classical layers from it too, so `seed` reached the quantum weights
and nothing else.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.parallel import parallel_map, resolve_jobs


# --------------------------------------------------------------------------- #
# the helper
# --------------------------------------------------------------------------- #
def test_order_is_preserved() -> None:
    """Results come back in the order given, whatever order they finish in."""
    assert parallel_map(lambda i: i * i, range(12), n_jobs=4) == [i * i for i in range(12)]


@pytest.mark.parametrize("n_jobs", [None, 1, 2, 4, -1])
def test_every_setting_gives_the_same_answers(n_jobs: int | None) -> None:
    assert parallel_map(str, range(9), n_jobs=n_jobs) == [str(i) for i in range(9)]


def test_serial_is_genuinely_serial() -> None:
    """n_jobs=1 must not create a pool of one - a traceback should read the same."""
    seen: list[str] = []
    import threading

    parallel_map(lambda i: seen.append(threading.current_thread().name), range(4), n_jobs=1)
    assert set(seen) == {threading.current_thread().name}


def test_resolve_jobs() -> None:
    import os

    assert resolve_jobs(None) == 1
    assert resolve_jobs(1) == 1
    assert resolve_jobs(4) == 4
    assert resolve_jobs(-1) == (os.cpu_count() or 1)


def test_an_exception_still_reaches_the_caller() -> None:
    def boom(i: int) -> int:
        if i == 3:
            raise ValueError("boom")
        return i

    with pytest.raises(ValueError, match="boom"):
        parallel_map(boom, range(6), n_jobs=4)


# --------------------------------------------------------------------------- #
# the promise that matters
# --------------------------------------------------------------------------- #
@pytest.mark.torch
def test_a_seeded_model_is_reproducible() -> None:
    """`seed` has to reach every source of randomness, not most of them.

    It used to reach the quantum weights only: the batch shuffle came off torch's
    global RNG and so did the `nn.Linear` initialisation, so two fits of the same
    seeded model disagreed.
    """
    pytest.importorskip("torch")
    X, y = qk.datasets.make_moons(n_samples=40, seed=0)
    scores = [
        qk.VQC(n_features=2, n_classes=2, seed=0).fit(X, y, epochs=4, batch_size=8).score(X, y)
        for _ in range(3)
    ]
    assert len(set(scores)) == 1, f"a seeded model gave {scores}"


@pytest.mark.torch
def test_seeding_does_not_disturb_the_callers_rng() -> None:
    """Constructing a seeded model must not reseed the process.

    `torch.manual_seed(seed)` here would have been the easy fix and would have
    silently changed every random number the caller drew afterwards.
    """
    torch = pytest.importorskip("torch")
    torch.manual_seed(1234)
    before = torch.randn(3)
    torch.manual_seed(1234)
    qk.VQC(n_features=2, n_classes=2, seed=7)
    after = torch.randn(3)
    assert torch.allclose(before, after)


@pytest.mark.torch
def test_search_gives_the_same_table_threaded_as_serial() -> None:
    """The point of n_jobs is that it is free. Same rows, same order, same means."""
    pytest.importorskip("torch")
    X, y = qk.datasets.make_moons(n_samples=40, seed=0)
    kw = dict(cv=2, verbose=False, seed=0, n_qubits=[2], n_layers=[1, 2], feature_map=["angle"])
    serial = qk.search(X, y, n_jobs=1, **kw)
    threaded = qk.search(X, y, n_jobs=4, **kw)
    assert [r.config for r in serial.rows] == [r.config for r in threaded.rows]
    np.testing.assert_allclose(
        [r.mean for r in serial.rows], [r.mean for r in threaded.rows], rtol=0, atol=0
    )
