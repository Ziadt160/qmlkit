"""Write ``docs/llms.txt`` and ``docs/llms-full.txt``.

A documentation site is built for someone who will browse it. A language model
does not browse: it fetches one URL, reads what it gets, and writes code. The
`llms.txt convention <https://llmstxt.org>`_ exists for exactly that reader —
``/llms.txt`` is a short index of what is here, ``/llms-full.txt`` is the whole
thing as one file.

Both are generated rather than written, for the same reason ``tests/test_docs.py``
executes every snippet: a hand-maintained summary of an API is a second copy of
the truth, and second copies rot. The prose comes from the documentation pages
themselves, and the API list comes from the package, so neither can drift from
what the library actually does.

The output is committed, so it is fetchable from the repository as well as from
the site, and ``--check`` fails when the committed copy is stale. That check runs
in CI, which is what makes the commitment real.

    python scripts/generate_llms_txt.py            # write
    python scripts/generate_llms_txt.py --check    # fail if stale
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
SITE = "https://ziadt160.github.io/qmlkit"

#: The things a reader cannot infer from the API and will otherwise get wrong.
#: Every line here is either a constraint that propagates or a trap already paid for.
PREAMBLE = """\
> A backend-agnostic quantum machine learning library: reusable feature maps, a
> composable ansatz vocabulary, quantum kernels, PyTorch layers, and one `grad()`
> that points at any circuit and observable. The same circuit runs on the built-in
> NumPy reference, SpinQit, Qiskit or Cirq.

Install with `pip install qmlkit`. The core depends on NumPy and nothing else;
every SDK is an optional extra (`qmlkit[torch]`, `[qiskit]`, `[cirq]`, `[spinqit]`,
`[sklearn]`).

What is worth knowing before writing any of it:

- **Simulator-only for the whole 0.x line.** No hardware. Expectations and
  gradients are exact unless you pass `shots=N`, and `adjoint` is therefore the
  default gradient method — parameter-shift costs `2P` circuit evaluations for the
  same answer, so it is the teaching subject and the test-suite reference, not the
  performance default.
- **Three layers, and nothing at a higher one hides a lower one.** `VQC(...).fit(X, y)`
  for a ready-made model; `QuantumLayer(...)` as an `nn.Module` inside any
  `nn.Sequential`; or circuits and `grad()` directly. Drop a level without giving up
  what the level above was doing.
- **Estimators are scikit-learn clonable** and models are `nn.Module`s, so `Pipeline`,
  `GridSearchCV`, `cross_val_score` and ordinary torch training loops all work.
- **Run `qk.diagnose(model)` before trusting a result.** In this field a mistake
  usually returns a plausible number rather than raising: a re-uploading model whose
  trainable block commutes with its encoding reaches one Fourier frequency instead of
  the `L` it was designed for, and it trains and converges anyway. `diagnose` returns
  findings with a code, what was measured, and the edit that fixes it.
- **The names are qmlkit's, not PennyLane's or Qiskit's**, and a wrong one tells you
  the right one: `qk.AngleEmbedding` reports that it is `qk.AngleFeatureMap` here.
  Unknown registry names suggest the nearest valid one, so guessing is cheap.
- **Every extension point is a registry:** `register_ansatz`, `register_gate`,
  `register_gradient`, `register_backend`, `register_conv_filter`. Registering makes
  your thing a first-class citizen everywhere that kind of argument is taken.
- **Data re-uploading is a pattern, not a class.** It is `EncodingLayer` composed
  into the block vocabulary, with `reupload()` as a convenience over it.
"""

_NAV_PAGE = re.compile(r"^- (?:(.+?): )?([\w/.-]+\.md)$")
_NAV_SECTION = re.compile(r"^- ([^:]+):$")
_HEADING = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_SKIP_LINE = re.compile(r"^(:::|---|!!!|--8<--|\$\$|>|#|\||\s*$|```)")


def _nav() -> list[tuple[str, str, str]]:
    """``(section, title, path)`` for every page, in the order the nav lists them."""
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    block = config.split("\nnav:", 1)[1].split("\n\n", 1)[0]
    section = ""
    pages: list[tuple[str, str, str]] = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (found := _NAV_PAGE.match(stripped)) is not None:
            title, path = found.group(1), found.group(2)
            pages.append((section, title or _title(path), path))
        elif (header := _NAV_SECTION.match(stripped)) is not None:
            section = header.group(1)
    return pages


def _title(path: str) -> str:
    """The page's own H1, which is the title a reader would see."""
    found = _HEADING.search((DOCS / path).read_text(encoding="utf-8"))
    return found.group(1).strip() if found else Path(path).stem


def _summary(path: str) -> str:
    """The page's first real sentence — not a heading, admonition, table or directive.

    Read by paragraph rather than by line: markdown wraps at the column, so the first
    line of a page is usually the first *half* of its first sentence. A paragraph
    under 40 characters is a lead-in ("Needs the extra:") rather than a summary, so
    the search continues past it; and a page whose first sentence is four words
    ("Six methods, one function.") gets the next one too, up to about a line.
    """
    paragraph: list[str] = []
    for line in [*(DOCS / path).read_text(encoding="utf-8").splitlines(), ""]:
        if not _SKIP_LINE.match(line):
            paragraph.append(line.strip())
            continue
        text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", " ".join(paragraph))
        text = re.sub(r"[*`_]", "", text).strip()
        paragraph = []
        if len(text) < 40:
            continue
        summary = ""
        for sentence in re.findall(r".+?(?:[.!?](?:\s|$)|$)", text):
            summary += sentence
            if len(summary.strip()) >= 60:
                break
        return summary.strip()
    return ""


def _is_directive_stub(body: str) -> bool:
    """True when a page is nothing but headings and mkdocs directives.

    The reference pages are ``::: qmlkit.metrics`` under a heading, and the changelog
    and release pages are ``--8<--`` includes. Both are filled in at build time, so
    the raw markdown carries no prose to copy — and for the reference pages the API
    index at the end of the file already covers what mkdocstrings would have produced.
    """
    content = [
        line for line in body.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]
    return bool(content) and all(
        line.lstrip().startswith((":::", "--8<--")) for line in content
    )


def _without_directives(body: str) -> str:
    """Drop the directive lines, keep everything around them.

    A reference page is a prose intro and a heading per module, with a ``:::`` under
    each. The directives expand to nothing here, but the headings around them are a
    map of which module holds what — worth keeping, and the API index below supplies
    the detail they would have expanded to.
    """
    kept = [
        line for line in body.splitlines() if not line.lstrip().startswith((":::", "--8<--"))
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def _url(path: str) -> str:
    """Where mkdocs will serve that page, with ``use_directory_urls`` on."""
    stem = path[: -len(".md")]
    return f"{SITE}/" if stem == "index" else f"{SITE}/{stem.removesuffix('/index')}/"


def _api_index() -> list[str]:
    """Every public name with its signature and one-line summary, from the package.

    Signatures are the point. A model that can see ``grad(spec, theta, obs=None, *,
    method='auto', shots=None, ...)`` does not have to guess at the keyword it wants,
    and guessing at keywords is where most first attempts fail.
    """
    import qmlkit as qk

    try:
        import qmlkit.nn  # noqa: F401
    except ImportError:  # pragma: no cover - the generator is run where torch exists
        sys.exit(
            "generating the API index needs the torch exports, so it needs torch:\n"
            "    pip install 'qmlkit[torch]'"
        )

    lines: list[str] = []
    for name in sorted({*qk.__all__, *qk._TORCH_EXPORTS} - {"nn"}):
        obj = getattr(qk, name)
        try:
            signature = str(inspect.signature(obj))
        except (TypeError, ValueError):
            signature = ""
        doc = (inspect.getdoc(obj) or "").strip()
        summary = doc.splitlines()[0] if doc else ""
        kind = "class" if inspect.isclass(obj) else "def" if callable(obj) else ""
        lines.append(f"{kind} {name}{signature}".strip())
        if summary:
            lines.append(f"    {summary}")
    return lines


def build_index() -> str:
    """``llms.txt``: what is here and where, in one screen."""
    out = ["# qmlkit", "", PREAMBLE.rstrip(), ""]
    grouped: dict[str, list[tuple[str, str]]] = {}
    for section, title, path in _nav():
        grouped.setdefault(section or "Start here", []).append((title, path))
    for section, pages in grouped.items():
        out += [f"## {section}", ""]
        for title, path in pages:
            summary = _summary(path)
            out.append(f"- [{title}]({_url(path)})" + (f": {summary}" if summary else ""))
        out.append("")
    out += [
        "## Full text",
        "",
        f"- [Everything above as one file]({SITE}/llms-full.txt): every tutorial and "
        "guide in full, then the whole public API with signatures.",
        "",
    ]
    return "\n".join(out)


def build_full() -> str:
    """``llms-full.txt``: the prose in full, then the API, in one fetch."""
    out = ["# qmlkit", "", PREAMBLE.rstrip(), ""]
    for section, title, path in _nav():
        body = (DOCS / path).read_text(encoding="utf-8").strip()
        if _is_directive_stub(body):
            continue
        body = _without_directives(body)
        label = f"{section} / {title}" if section else title
        out += ["", "=" * 78, f"# {label}    (source: docs/{path})", "=" * 78, "", body, ""]
    out += [
        "",
        "=" * 78,
        "# API reference    (generated from the package)",
        "=" * 78,
        "",
        *_api_index(),
        "",
    ]
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed files are stale"
    )
    args = parser.parse_args()

    wanted = {DOCS / "llms.txt": build_index(), DOCS / "llms-full.txt": build_full()}
    stale = []
    for path, content in wanted.items():
        current = path.read_text(encoding="utf-8") if path.is_file() else None
        if current == content:
            continue
        if args.check:
            stale.append(path.relative_to(ROOT).as_posix())
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
            print(f"wrote {path.relative_to(ROOT).as_posix()} ({len(content):,} bytes)")

    if stale:
        print(f"stale: {', '.join(stale)}\nRegenerate with: python {Path(__file__).name}")
        return 1
    if args.check:
        print("llms.txt and llms-full.txt are current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
