"""One HTML file that says what a run did.

A run produces a number, and a month later the number is all that is left. This
writes down the rest of it: where the time went, what the loss did, what the
library was told and what it was running on.

Deliberately a *file*, not a server. A dashboard you have to start, connect to and
keep alive is a dependency, a port and a process; a page you can open, email, and
drop next to the result in a directory is none of those and outlives all of them.
The charts are inline SVG for the same reason - nothing to fetch, nothing to pin,
nothing to break in two years.

    >>> import qmlkit as qk
    >>> with qk.progress(live=False) as run:            # doctest: +SKIP
    ...     model.fit(X, y)
    >>> run.save_html("run.html")                       # doctest: +SKIP
    'run.html'
"""

from __future__ import annotations

import html
import platform
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from qmlkit.progress import Progress

__all__ = ["render"]

_W, _H = 640, 200
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 52, 12, 14, 28


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _human(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"
    return f"{int(seconds // 3600)}h{int((seconds % 3600) // 60):02d}m"


def _tick(value: float) -> str:
    """Enough digits to distinguish the axis ends, and no more."""
    magnitude = abs(value)
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.1e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _line_chart(name: str, points: list[tuple[int, float]]) -> str:
    """A single series as inline SVG.

    Handles the two cases a naive chart gets wrong: one point (no range to scale
    against) and a flat series (zero range, which divides by zero). Both are real -
    a loss that never moved is exactly the run you most want to look at.
    """
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    lo, hi = min(ys), max(ys)
    x_lo, x_hi = min(xs), max(xs)
    flat = hi - lo < 1e-15
    span = 1.0 if flat else hi - lo
    x_span = max(x_hi - x_lo, 1)

    def px(x: int) -> float:
        return _PAD_L + (x - x_lo) / x_span * (_W - _PAD_L - _PAD_R)

    def py(y: float) -> float:
        if flat:
            return _H / 2
        return _H - _PAD_B - (y - lo) / span * (_H - _PAD_T - _PAD_B)

    path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
    last_x, last_y = points[-1]
    body = [
        f'<svg class="chart" viewBox="0 0 {_W} {_H}" role="img" '
        f'aria-label="{_esc(name)} over {len(points)} steps">',
        f'<line class="axis" x1="{_PAD_L}" y1="{_H - _PAD_B}" '
        f'x2="{_W - _PAD_R}" y2="{_H - _PAD_B}"/>',
        f'<line class="axis" x1="{_PAD_L}" y1="{_PAD_T}" x2="{_PAD_L}" y2="{_H - _PAD_B}"/>',
        f'<polyline class="series" points="{path}"/>',
        f'<circle class="dot" cx="{px(last_x):.1f}" cy="{py(last_y):.1f}" r="3"/>',
        f'<text class="tick" x="{_PAD_L - 6}" y="{_PAD_T + 4}" text-anchor="end">'
        f"{_esc(_tick(hi))}</text>",
        f'<text class="tick" x="{_PAD_L - 6}" y="{_H - _PAD_B}" text-anchor="end">'
        f"{_esc(_tick(lo))}</text>",
        f'<text class="tick" x="{_PAD_L}" y="{_H - 8}">{x_lo}</text>',
        f'<text class="tick" x="{_W - _PAD_R}" y="{_H - 8}" text-anchor="end">{x_hi}</text>',
        "</svg>",
    ]
    note = " <span class='flat'>(flat - never moved)</span>" if flat else ""
    return (
        f'<section class="panel"><h3>{_esc(name)}{note}</h3>'
        + "".join(body)
        + f'<p class="meta">{len(points)} points  ·  first {_esc(_tick(ys[0]))}  ·  '
        f"last {_esc(_tick(ys[-1]))}</p></section>"
    )


def _timing_table(run: Progress) -> str:
    if not run.records:
        return '<p class="empty">Nothing was tracked in this run.</p>'
    longest = max(r.seconds for r in run.records) or 1.0
    rows = []
    for record in run.records:
        rate = f"{record.per_item * 1000:.2f} ms" if record.per_item is not None else "-"
        width = record.seconds / longest * 100
        indent = f"padding-left:{10 + record.depth * 18}px"
        rows.append(
            f"<tr><th scope='row' style='{indent}'>{_esc(record.label)}</th>"
            f"<td class='num'>{record.items:,}</td>"
            f"<td class='num'>{record.seconds:.2f}s</td>"
            f"<td class='num'>{rate}</td>"
            f"<td class='bar'><span style='width:{width:.1f}%'></span></td></tr>"
        )
    tracked = sum(r.seconds for r in run.records if r.depth == 0)
    untracked = run.elapsed - tracked
    if untracked > 0.05 * run.elapsed and untracked > 0.1:
        share = untracked / longest * 100
        # On a first run this is usually an optional SDK imported lazily rather than
        # work the run did — `import sklearn.svm` alone is ~1.8s — so the row says so
        # instead of leaving a large unexplained number on the page.
        rows.append(
            "<tr class='untracked'><th scope='row'>(untracked)<br>"
            "<span class='hint'>setup and imports, not work the run did</span></th>"
            "<td class='num'>-</td>"
            f"<td class='num'>{untracked:.2f}s</td><td class='num'>-</td>"
            f"<td class='bar'><span style='width:{share:.1f}%'></span></td></tr>"
        )
    return (
        "<table><thead><tr><th>task</th><th class='num'>items</th>"
        "<th class='num'>time</th><th class='num'>per item</th><th></th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def _facts(run: Progress) -> str:
    facts: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run.started_at)),
        "duration": _human(run.elapsed),
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
    }
    try:
        import numpy as np

        facts["numpy"] = np.__version__
    except Exception:  # pragma: no cover - numpy is a hard dependency
        pass
    try:
        from qmlkit import __version__

        facts["qmlkit"] = __version__
    except Exception:  # pragma: no cover
        pass
    facts.update(run.meta)
    rows = "".join(
        f"<tr><th scope='row'>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in facts.items()
    )
    return f"<table class='facts'><tbody>{rows}</tbody></table>"


_CSS = """
:root{--bg:#fbfbfa;--fg:#1c1c1a;--dim:#6b6b66;--line:#e2e2dd;--card:#fff;
--ink:#3a6b5c;--warn:#8a6a1f}
@media (prefers-color-scheme:dark){:root{--bg:#16161a;--fg:#e8e8e4;--dim:#9a9a94;
--line:#2e2e34;--card:#1e1e23;--ink:#7fd1b4;--warn:#d8b45a}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:820px;margin:0 auto;padding:36px 20px 64px}
h1{font-size:21px;margin:0 0 2px;letter-spacing:-.01em}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.09em;color:var(--dim);
margin:34px 0 10px;font-weight:600}
h3{font-size:14px;margin:0 0 10px;font-weight:600}
.sub{color:var(--dim);margin:0 0 4px}
.panel{background:var(--card);border:1px solid var(--line);border-radius:9px;
padding:16px;margin-bottom:14px}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:9px;overflow:hidden}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}
thead th{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--dim)}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
th[scope=row]{font-weight:500}
.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.bar{width:28%}
.bar span{display:block;height:7px;border-radius:4px;background:var(--ink);min-width:2px}
.untracked th,.untracked td{color:var(--dim);font-style:italic}
.hint{font-style:normal;font-size:11px;font-weight:400;opacity:.8}
.untracked .bar span{background:var(--line)}
.chart{width:100%;height:auto;display:block}
.axis{stroke:var(--line);stroke-width:1}
.series{fill:none;stroke:var(--ink);stroke-width:2;stroke-linejoin:round;
stroke-linecap:round}
.dot{fill:var(--ink)}
.tick{fill:var(--dim);font-size:10px;font-family:ui-monospace,SFMono-Regular,monospace}
.meta{color:var(--dim);font-size:12px;margin:8px 0 0}
.flat{color:var(--warn);font-weight:400;font-size:12px}
.empty{color:var(--dim)}
.facts th[scope=row]{color:var(--dim);width:34%;font-weight:400}
.facts td{font-family:ui-monospace,SFMono-Regular,monospace;font-size:12.5px}
footer{color:var(--dim);font-size:12px;margin-top:34px;
border-top:1px solid var(--line);padding-top:12px}
"""


def render(run: Progress, title: str = "qmlkit run") -> str:
    """The run as one self-contained HTML page."""
    charts = "".join(_line_chart(name, points) for name, points in sorted(run.series.items()))
    series_block = (
        f"<h2>What the run did</h2>{charts}"
        if charts
        else "<h2>What the run did</h2><p class='empty'>No series were logged. "
        "A training loop logs its loss here; anything else can call "
        # `from qmlkit.progress import log`, not `qmlkit.progress.log`: importing the
        # `progress` context manager into the package namespace rebinds the attribute
        # to the function, so the dotted path raises AttributeError.
        "<code>from qmlkit.progress import log</code>, then "
        "<code>log(name, value)</code>.</p>"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>{_esc(title)}</h1>
<p class="sub">{len(run.records)} task(s) · {_human(run.elapsed)} · generated by qmlkit</p>
<h2>Where the time went</h2>
{_timing_table(run)}
{series_block}
<h2>What it was running on</h2>
{_facts(run)}
<footer>Every number here was measured during the run it describes.
Nothing on this page was estimated or rounded from a different run.</footer>
</div></body></html>
"""
