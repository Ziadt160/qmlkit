"""Watching a run must not change it.

That is the headline assertion in this file and the reason the module is allowed to
exist: a progress reporter that perturbs a seeded result would be a worse defect
than the silence it replaces.

The rest holds it to the other two promises - free when nobody is watching, and
honest about what it does not yet know.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.progress import current, task


# --------------------------------------------------------------------------- #
# it must not change a number
# --------------------------------------------------------------------------- #
def test_watching_a_kernel_does_not_change_it() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (12, 2))
    kernel = qk.QuantumKernel(qk.ZZFeatureMap(2), shots=128, seed=7)

    quiet = kernel(X)
    kernel.reset()
    with qk.progress(live=False):
        watched = kernel(X)

    np.testing.assert_array_equal(quiet, watched)


def test_watching_a_circuit_does_not_change_it() -> None:
    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build()
    theta = ansatz.init(seed=0)

    quiet = qk.expectation(spec, qk.Z(0), theta=theta)
    with qk.progress(live=False):
        watched = qk.expectation(spec, qk.Z(0), theta=theta)

    assert quiet == watched


# --------------------------------------------------------------------------- #
# free when nobody is watching
# --------------------------------------------------------------------------- #
def test_silent_by_default() -> None:
    """No reporter is active unless one is asked for."""
    assert current() is None
    with qk.progress(live=False):
        assert current() is not None
    assert current() is None


def test_a_task_outside_a_reporter_is_a_no_op() -> None:
    with task("nothing is watching", 100) as tracked:
        tracked.advance(50)  # must not raise, must not record
    assert current() is None


def test_nesting_restores_the_outer_reporter() -> None:
    with qk.progress(live=False) as outer:
        with qk.progress(live=False) as inner:
            assert current() is inner
        assert current() is outer


# --------------------------------------------------------------------------- #
# what it records
# --------------------------------------------------------------------------- #
def test_records_the_item_count() -> None:
    with qk.progress(live=False) as run, run.task("work", 10) as tracked:
        for _ in range(10):
            tracked.advance()
    assert [(r.label, r.items) for r in run.records] == [("work", 10)]


def test_records_a_task_that_raised() -> None:
    """A run that died half way is exactly when the report is wanted."""
    with (
        qk.progress(live=False) as run,
        pytest.raises(ValueError),
        run.task("doomed", 10) as tracked,
    ):
        tracked.advance(3)
        raise ValueError("boom")
    assert run.records[0].items == 3


def test_nested_tasks_record_their_depth() -> None:
    with qk.progress(live=False) as run, run.task("outer", 2) as outer:
        with run.task("inner", 5) as inner:
            inner.advance(5)
        outer.advance(2)
    depths = {r.label: r.depth for r in run.records}
    assert depths == {"inner": 1, "outer": 0}


def test_track_counts_an_iterable() -> None:
    with qk.progress(live=False) as run:
        consumed = list(qk.track(range(7), "items"))
    assert consumed == list(range(7))
    assert run.records[0].items == 7


def test_track_without_a_reporter_still_yields_everything() -> None:
    assert list(qk.track(range(5), "items")) == list(range(5))


# --------------------------------------------------------------------------- #
# honest about what it does not know
# --------------------------------------------------------------------------- #
def test_refuses_an_estimate_from_too_few_samples() -> None:
    """Four items in a millisecond says nothing about the next ten thousand."""
    with qk.progress(live=False) as run, run.task("work", 10_000) as tracked:
        tracked.advance(4)
        assert tracked.remaining is None


def test_estimates_once_there_is_evidence() -> None:
    with qk.progress(live=False) as run, run.task("work", 200) as tracked:
        for _ in range(20):
            time.sleep(0.03)
            tracked.advance()
        elapsed = tracked.elapsed
        left = tracked.remaining

    assert left is not None
    # 20 of 200 are done, so the remaining 180 must be estimated at 9x what the 20
    # took. Asserted as that ratio rather than as a wall-clock range: a loaded runner
    # can make sleep(0.03) take as long as it likes and the extrapolation is still
    # exactly right. The earlier version asserted `left < 20.0` and failed on macOS
    # at 20.1 -- a suite whose colour depends on the runner's mood proves nothing.
    assert left / elapsed == pytest.approx(9.0, rel=0.05)


def test_no_estimate_without_a_total() -> None:
    with qk.progress(live=False) as run, run.task("unbounded") as tracked:
        tracked.advance(1000)
        assert tracked.remaining is None


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #
def test_report_names_the_work_and_the_rate() -> None:
    with qk.progress(live=False) as run, run.task("kernel gram", 4) as tracked:
        tracked.advance(4)
    text = run.report()
    assert "kernel gram" in text
    assert "4 items" in text
    assert "Run finished" in text


def test_report_when_nothing_was_tracked_says_so() -> None:
    with qk.progress(live=False) as run:
        pass
    assert "nothing was tracked" in run.report()


def test_live_output_goes_to_the_stream_it_was_given() -> None:
    stream = io.StringIO()
    with qk.progress(live=True, stream=stream) as run, run.task("visible", 2) as tracked:
        tracked.advance()
    assert "visible" in stream.getvalue()


def test_live_output_is_throttled() -> None:
    """A million advances must not become a million redraws."""
    stream = io.StringIO()
    with qk.progress(live=True, stream=stream) as run, run.task("fast", 100_000) as tracked:
        for _ in range(100_000):
            tracked.advance()
    # one forced draw at task start, then at most a handful more
    assert stream.getvalue().count("fast") < 100


# --------------------------------------------------------------------------- #
# the slow path is the one that has to report
# --------------------------------------------------------------------------- #
def test_a_backend_without_a_native_batch_reports_each_circuit() -> None:
    """The case this module exists for: a long run that looks like one call.

    A Gram matrix on Qiskit is one call from the caller's side and hundreds of
    simulations from inside `Backend.statevector_batch_slots`. Reporting at the call
    would sit silent for minutes, which is exactly the complaint about every other
    library in this field.
    """
    from qmlkit.core.backends.base import Backend

    spec = qk.hardware_efficient(2, 1).build()
    rows = np.zeros((7, spec.n_params))
    backend = qk.get_backend("numpy")

    with qk.progress(live=False) as run:
        Backend.statevector_batch_slots(backend, spec, rows)  # the base fallback

    assert [(r.label, r.items) for r in run.records] == [("numpy circuits", 7)]


def test_a_vectorised_backend_reports_nothing() -> None:
    """NumPy overrides the fallback with one vectorised call.

    There is no incremental progress to report there, and inventing some would be a
    progress bar that describes nothing.
    """
    spec = qk.hardware_efficient(2, 1).build()
    rows = np.zeros((7, spec.n_params))

    with qk.progress(live=False) as run:
        qk.get_backend("numpy").statevector_batch_slots(spec, rows)

    assert run.records == []


def test_a_large_untracked_row_explains_itself() -> None:
    """An unexplained number invites someone to chase it.

    On a first run the untracked time is usually an optional SDK being imported
    lazily - `import sklearn.svm` alone is ~1.8s - rather than work the run did. The
    same run warm accounts for 98% of its own time, so the row says where to look
    instead of leaving a large silent gap.
    """
    with qk.progress(live=False) as run:
        with run.task("work", 1) as tracked:
            tracked.advance()
        time.sleep(0.3)  # stand-in for an import happening outside any task

    text = run.report()
    assert "(untracked)" in text
    assert "one-time imports" in text
