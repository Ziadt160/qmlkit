"""Run independent work on more than one core.

Every other performance measurement in this library says the same thing: threading
*inside* one circuit is slower than not, because at the widths quantum machine
learning runs at the statevector fits in a core's cache and handing the work out
costs more than doing it — measured 0.04x on 8 threads at 8 qubits.

This is the other half of that sentence. A cross-validation fold, a hyperparameter
configuration, a random seed and a Gram matrix are *independent*: they share no state
and each takes seconds. Those are worth spreading, and they are the only things here
that are.

    >>> import qmlkit as qk
    >>> qk.parallel_map(train_one_fold, folds, n_jobs=4)     # doctest: +SKIP

**Threads, not processes, and the ceiling is real.** Measured on six independent
22x22 Gram matrices: 1.38x on 2 workers, 1.59x on 4, **1.75x on 6**. That is well
short of linear because the GIL is held during this library's Python-level dispatch
and released only inside NumPy — the same dispatch cost that dominates everything
else. Processes would scale further and would also have to pickle every argument
across a spawn on Windows, which rules out closures, lambdas and most model objects.
A reliable 1.75x beat an unreliable 4x.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

__all__ = ["parallel_map", "resolve_jobs"]

T = TypeVar("T")
R = TypeVar("R")


def resolve_jobs(n_jobs: int | None) -> int:
    """How many workers ``n_jobs`` means.

    ``None`` or ``1`` is serial — the default everywhere, because a library that
    silently takes every core is a bad guest inside someone else's parallel loop.
    ``-1`` means every core.
    """
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs < 0:
        return os.cpu_count() or 1
    return n_jobs


def parallel_map(fn: Callable[[T], R], items: Iterable[T], n_jobs: int | None = None) -> list[R]:
    """Apply ``fn`` to each item, on ``n_jobs`` threads, preserving order.

    Serial when ``n_jobs`` is 1 or there is nothing to gain — and *genuinely* serial,
    not a pool of one: no executor is created, so nothing changes about how the work
    runs or how a traceback reads.

    Parameters
    ----------
    fn
        Called once per item. It must be safe to run concurrently — which is the
        caller's claim to make, not this function's. Everything in qmlkit it is used
        on evaluates circuits and touches no shared state, with one exception worth
        knowing: a :class:`~qmlkit.kernels.matrix.QuantumKernel` carries a cache, so
        two threads sharing one instance can duplicate work. They cannot corrupt it.
    items
        Consumed eagerly, so that a generator is not shared across threads.
    n_jobs
        ``None``/``1`` serial, ``-1`` every core.
    """
    work: Sequence[T] = list(items)
    workers = min(resolve_jobs(n_jobs), len(work))
    if workers <= 1 or len(work) <= 1:
        return [fn(item) for item in work]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, work))


def parallel_imap(
    fn: Callable[[T], R], items: Iterable[T], n_jobs: int | None = None
) -> Iterator[R]:
    """:func:`parallel_map`, yielding as results arrive in order.

    For the caller that wants to report progress or stop early rather than wait for
    the whole list.
    """
    work: Sequence[T] = list(items)
    workers = min(resolve_jobs(n_jobs), len(work))
    if workers <= 1 or len(work) <= 1:
        yield from (fn(item) for item in work)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(fn, work)
