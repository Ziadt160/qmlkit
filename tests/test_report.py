"""The HTML report.

Two things are worth testing about a generated page and they are not the ones that
look obvious. First that it is genuinely **self-contained** - the promise is a file
you can open in two years with no network and no dependency, and a single stray
``<script src>`` breaks that silently. Second that it **escapes** what it is given:
labels come from model class names and user-supplied metadata, and a report is
exactly the kind of artefact that gets emailed around.

The chart edge cases are here too, because both of them are real runs: a series with
one point has no range to scale against, and a flat series has zero range, which is
the loss curve of a model that never learned - the run you most want to look at.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

import numpy as np
import pytest

import qmlkit as qk


def _page(build) -> str:
    with qk.progress(live=False) as run:
        build(run)
    return run.html()


class _Balanced(HTMLParser):
    """Minimal well-formedness check: every non-void tag closes, in order."""

    VOID = {"meta", "br", "hr", "img", "input", "link", "circle", "line", "polyline", "path"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or ['nothing']}")
            return
        self.stack.pop()


# --------------------------------------------------------------------------- #
# the two promises
# --------------------------------------------------------------------------- #
def test_the_page_is_self_contained() -> None:
    """No network, no assets, no CDN. That is the whole point of a file."""

    def build(run):
        with run.task("work", 3) as t:
            t.advance(3)
        run.log("loss", 0.5, 0)
        run.log("loss", 0.2, 1)

    page = _page(build)
    assert "<script" not in page.lower()
    assert "http://" not in page and "https://" not in page
    assert not re.search(r"<link\b", page, re.I)
    assert not re.search(r"\bsrc\s*=", page, re.I)


def test_hostile_labels_and_metadata_are_escaped() -> None:
    def build(run):
        run.note(**{"dataset": "<script>alert(1)</script>"})
        with run.task("<img onerror=x>", 1) as t:
            t.advance()
        run.log("<b>loss</b>", 1.0, 0)

    page = _page(build)
    assert "<script>alert" not in page
    assert "<img onerror" not in page
    assert "&lt;script&gt;" in page


def test_the_page_is_well_formed() -> None:
    def build(run):
        with run.task("work", 2) as t:
            t.advance(2)
        run.log("loss", 1.0, 0)
        run.log("loss", 0.4, 1)

    parser = _Balanced()
    parser.feed(_page(build))
    assert parser.errors == []
    assert parser.stack == []


# --------------------------------------------------------------------------- #
# what it says
# --------------------------------------------------------------------------- #
def test_names_the_tasks_and_the_series() -> None:
    def build(run):
        with run.task("kernel gram", 5) as t:
            t.advance(5)
        run.log("gradient norm", 0.01, 0)

    page = _page(build)
    assert "kernel gram" in page
    assert "gradient norm" in page


def test_reports_run_level_facts() -> None:
    def build(run):
        run.note(seed=1234, dataset="breast-cancer")

    page = _page(build)
    assert "1234" in page
    assert "breast-cancer" in page


def test_says_so_when_nothing_was_tracked() -> None:
    page = _page(lambda run: None)
    assert "Nothing was tracked" in page
    assert "No series were logged" in page


# --------------------------------------------------------------------------- #
# chart edge cases, both of which are real runs
# --------------------------------------------------------------------------- #
def test_a_single_point_series_does_not_break() -> None:
    page = _page(lambda run: run.log("loss", 0.7, 0))
    assert "<svg" in page
    assert "nan" not in page.lower()


def test_a_flat_series_is_drawn_and_called_out() -> None:
    """A loss that never moved is the most diagnostic curve there is."""

    def build(run):
        for step in range(10):
            run.log("loss", 0.6931, step)

    page = _page(build)
    assert "<svg" in page
    assert "never moved" in page
    assert "nan" not in page.lower()


def test_no_series_values_render_as_nan_or_inf() -> None:
    def build(run):
        for step, value in enumerate([1e-9, 1.0, 1e7]):
            run.log("wide range", value, step)

    page = _page(build)
    assert "nan" not in page.lower()
    assert "inf" not in page.lower()


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
def test_a_real_run_produces_a_page_naming_its_work(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (16, 2))
    with qk.progress(live=False) as run:
        kernel = qk.QuantumKernel(qk.ZZFeatureMap(2), shots=64, seed=3)
        kernel(X)
    path = run.save_html(str(tmp_path / "run.html"), title="a real run")
    page = (tmp_path / "run.html").read_text(encoding="utf-8")
    assert path.endswith("run.html")
    assert "a real run" in page
    assert "kernel gram" in page


@pytest.mark.torch
def test_training_logs_its_trajectory() -> None:
    torch = pytest.importorskip("torch")
    assert torch is not None
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (20, 2))
    y = (X[:, 0] > np.pi / 2).astype(int)
    with qk.progress(live=False) as run:
        qk.VQC(n_features=2, n_classes=2, seed=0).fit(X, y, epochs=4, batch_size=10)
    assert set(run.series) == {"loss", "gradient norm", "parameter norm"}
    assert len(run.series["loss"]) == 4
    assert "gradient norm" in run.html()
