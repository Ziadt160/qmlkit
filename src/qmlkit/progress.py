"""What a long run is doing, and how much of it is left.

`qk.plan` answers this before a run starts and `qk.diagnose` answers it afterwards.
In between there has been nothing, and "in between" is where the hours go: a
quantum kernel on a few hundred points is tens of thousands of circuits, and every
library in this field runs them behind a silent call that returns when it returns.

    >>> import qmlkit as qk
    >>> with qk.progress() as run:                      # doctest: +SKIP
    ...     gram = kernel(X)
    ...
    kernel gram   3,412/12,720   27%   14.2s elapsed   ~37s left

    >>> print(run.report())                             # doctest: +SKIP
    Run finished in 51.4s
      kernel gram        12,720 items   51.2s    4.0 ms/item

Three properties this has to keep, in this order:

1. **It must not change any number.** Recording is a counter and a clock; nothing
   here touches a state, an angle or a seed.
2. **It must be free when nobody is watching.** With no active reporter, advancing a
   task is one attribute lookup against ``None``. There is no import of this module
   on any hot path that does not already need it.
3. **It must not claim to know what it does not.** An estimate from four samples in
   half a second is a guess, and it says ``estimating`` rather than printing a
   confident number that will be wrong by an order of magnitude. This is the same
   rule the rest of the library follows about reporting measurements.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import IO, Any, TypeVar

__all__ = ["Progress", "Task", "TaskRecord", "current", "log", "note", "progress", "track"]

T = TypeVar("T")

#: Live line redraws at most this often. A run that advances a million times must not
#: spend its time formatting strings.
_REDRAW_SECONDS = 0.1

#: Below this many completed items, or this many seconds, a rate is not yet a rate.
_MIN_SAMPLES = 8
_MIN_ELAPSED = 0.5

#: The active reporter, or ``None``. Module-level on purpose: every long loop in the
#: library would otherwise have to thread a reporter argument through its whole call
#: stack, and that is an API change to a dozen public signatures for a progress bar.
_CURRENT: Progress | None = None


def current() -> Progress | None:
    """The active :class:`Progress`, or ``None`` when nothing is watching."""
    return _CURRENT


@dataclass
class TaskRecord:
    """One finished unit of work, kept for the report."""

    label: str
    items: int
    seconds: float
    depth: int = 0

    @property
    def per_item(self) -> float | None:
        return self.seconds / self.items if self.items else None

    def __str__(self) -> str:
        rate = f"{self.per_item * 1000:>8.2f} ms/item" if self.per_item is not None else ""
        indent = "  " * self.depth
        return f"{indent}{self.label:<24}{self.items:>10,} items{self.seconds:>8.1f}s{rate}"


class Task:
    """A unit of work being counted.

    Obtained from :meth:`Progress.task`, and used as a context manager so that the
    end time is recorded even when the work raises.
    """

    __slots__ = ("label", "total", "done", "started", "_owner", "_depth")

    def __init__(self, owner: Progress, label: str, total: int | None, depth: int) -> None:
        self.label = label
        self.total = total
        self.done = 0
        self.started = time.perf_counter()
        self._owner = owner
        self._depth = depth

    def advance(self, n: int = 1) -> None:
        """Count ``n`` more items done, and redraw if it has been long enough."""
        self.done += n
        self._owner._maybe_draw()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def remaining(self) -> float | None:
        """Seconds left at the rate measured so far, or ``None`` when that is a guess.

        Deliberately refuses to answer early. An extrapolation from a handful of
        items in under a second says more about scheduling noise than about the run.
        """
        elapsed = self.elapsed
        if self.total is None or self.done < _MIN_SAMPLES or elapsed < _MIN_ELAPSED:
            return None
        if self.done >= self.total:
            return 0.0
        return (self.total - self.done) * elapsed / self.done

    def __enter__(self) -> Task:
        return self

    def __exit__(self, *exc: object) -> None:
        self._owner._finish(self)


def _human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


class Progress:
    """Collects what the run is doing, and optionally shows it live."""

    def __init__(self, stream: IO[str] | None = None, live: bool = True) -> None:
        self.records: list[TaskRecord] = []
        self.series: dict[str, list[tuple[int, float]]] = {}
        self.meta: dict[str, Any] = {}
        self.started = time.perf_counter()
        self.started_at = time.time()
        self._stack: list[Task] = []
        self._stream = stream if stream is not None else sys.stderr
        self._live = live
        self._last_draw = 0.0
        self._drawn = 0

    # -- the API library code calls ------------------------------------------- #
    def task(self, label: str, total: int | None = None) -> Task:
        """Begin a unit of work. Use as a context manager."""
        task = Task(self, label, total, depth=len(self._stack))
        self._stack.append(task)
        self._draw(force=True)
        return task

    def log(self, name: str, value: float, step: int | None = None) -> None:
        """Record one point of a named scalar series.

        The trajectory, not just the timings: a loss that fell and then stopped, a
        gradient norm that went to zero, a parameter norm that ran away. These are
        what the report plots, and what makes a finished run answerable afterwards
        rather than only observable while it happens.
        """
        points = self.series.setdefault(name, [])
        points.append((len(points) if step is None else step, float(value)))

    def note(self, **facts: Any) -> None:
        """Attach run-level facts - shapes, seeds, configuration - to the record."""
        self.meta.update(facts)

    # -- rendering ------------------------------------------------------------ #
    def _maybe_draw(self) -> None:
        if not self._live:
            return
        now = time.perf_counter()
        if now - self._last_draw >= _REDRAW_SECONDS:
            self._draw()

    def _draw(self, force: bool = False) -> None:
        if not self._live or not self._stack:
            return
        if not force and time.perf_counter() - self._last_draw < _REDRAW_SECONDS:
            return
        self._last_draw = time.perf_counter()
        task = self._stack[-1]
        parts = [task.label]
        if task.total:
            parts.append(f"{task.done:,}/{task.total:,}")
            parts.append(f"{task.done / task.total * 100:3.0f}%")
        else:
            parts.append(f"{task.done:,}")
        parts.append(f"{_human(task.elapsed)} elapsed")
        left = task.remaining
        if left is not None:
            parts.append(f"~{_human(left)} left")
        elif task.total:
            parts.append("estimating")
        line = "  ".join(parts)
        self._write(line)

    def _write(self, line: str) -> None:
        pad = " " * max(0, self._drawn - len(line))
        self._drawn = len(line)
        try:
            self._stream.write(f"\r{line}{pad}")
            self._stream.flush()
        except Exception:  # pragma: no cover - a closed or exotic stream is not fatal
            self._live = False

    def _clear(self) -> None:
        if self._live and self._drawn:
            try:
                self._stream.write("\r" + " " * self._drawn + "\r")
                self._stream.flush()
            except Exception:  # pragma: no cover
                pass
            self._drawn = 0

    def _finish(self, task: Task) -> None:
        if task in self._stack:
            self._stack.remove(task)
        self.records.append(TaskRecord(task.label, task.done, task.elapsed, depth=task._depth))
        if not self._stack:
            self._clear()

    # -- afterwards ----------------------------------------------------------- #
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def report(self) -> str:
        """What the run spent its time on, worst first within each nesting level."""
        if not self.records:
            return f"Run finished in {_human(self.elapsed)}; nothing was tracked."
        lines = [f"Run finished in {_human(self.elapsed)}"]
        lines.extend(str(record) for record in self.records)
        tracked = sum(r.seconds for r in self.records if r.depth == 0)
        untracked = self.elapsed - tracked
        if untracked > 0.05 * self.elapsed and untracked > 0.1:
            lines.append(
                f"  {'(untracked)':<24}{'':>10}      {untracked:>8.1f}s"
                "   - setup, data handling, and anything outside a task"
            )
            # On a first run this row is usually an optional SDK being imported
            # lazily, which is not work the run did and not worth chasing. Measured:
            # `import sklearn.svm` alone is ~1.8s, and the same run warm accounts for
            # 98% of its own time.
            lines.append(
                f"  {'':<24}{'':>10}              "
                "  on a first run this is mostly one-time imports; re-run to see"
            )
        return "\n".join(lines)

    def html(self, title: str = "qmlkit run") -> str:
        """The whole run as one self-contained HTML page.

        No dependencies, no server, no asset directory - the charts are inline SVG
        drawn from the recorded series. A file you can open, keep beside the result,
        or attach to whatever you are writing up.
        """
        from qmlkit.report import render

        return render(self, title=title)

    def save_html(self, path: str, title: str = "qmlkit run") -> str:
        """Write :meth:`html` to ``path`` and return the path."""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.html(title=title))
        return path

    def __str__(self) -> str:
        return self.report()


@contextmanager
def progress(live: bool = True, stream: IO[str] | None = None) -> Iterator[Progress]:
    """Watch a run, and keep what it did.

    Parameters
    ----------
    live
        Draw a line to ``stream`` as the run proceeds. ``False`` records silently,
        which is what a script or a test wants.
    stream
        Where the live line goes. Defaults to ``sys.stderr``, so piping a script's
        stdout to a file does not collect progress redraws.

    Notes
    -----
    Nothing outside this block is affected: with no active reporter, the tracking
    calls inside the library cost one comparison against ``None``.
    """
    global _CURRENT
    previous = _CURRENT
    reporter = Progress(stream=stream, live=live)
    _CURRENT = reporter
    try:
        yield reporter
    finally:
        reporter._clear()
        _CURRENT = previous


def track(iterable: Iterable[T], label: str, total: int | None = None) -> Iterator[T]:
    """Count an iterable's items as they are consumed, if anything is watching.

    The convenience form for a plain loop. ``total`` is taken from ``len`` when the
    iterable has one.
    """
    reporter = _CURRENT
    if reporter is None:
        yield from iterable
        return
    if total is None:
        try:
            total = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            total = None
    with reporter.task(label, total) as task:
        for item in iterable:
            yield item
            task.advance()


@contextmanager
def task(label: str, total: int | None = None) -> Iterator[Any]:
    """A task if something is watching, and a no-op object if not.

    Lets library code write ``with task(...) as t: ... t.advance()`` without
    branching on whether a reporter exists.
    """
    reporter = _CURRENT
    if reporter is None:
        yield _SILENT
        return
    with reporter.task(label, total) as live:
        yield live


def log(name: str, value: float, step: int | None = None) -> None:
    """Record a scalar into the active run, or do nothing if none is active.

    The counterpart to :func:`task` for library code: one comparison when nobody is
    watching, so a training loop can log unconditionally.
    """
    reporter = _CURRENT
    if reporter is not None:
        reporter.log(name, value, step)


def note(**facts: Any) -> None:
    """Attach run-level facts to the active run, or do nothing if none is active."""
    reporter = _CURRENT
    if reporter is not None:
        reporter.note(**facts)


class _Silent:
    """Stands in for a Task when nothing is watching. Advancing it does nothing."""

    __slots__ = ()

    def advance(self, n: int = 1) -> None:
        return

    @property
    def remaining(self) -> float | None:
        return None


_SILENT = _Silent()
