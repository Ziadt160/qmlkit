"""Every Python block in the documentation is executed here.

Documentation rots silently. A tutorial that stopped working two releases ago looks
exactly like one that works, right up until somebody types it in — and the first
thing a new user does is copy the first snippet on the page. So the snippets are not
illustrations of the API, they *are* tests of it, and a rename that breaks a tutorial
breaks the build.

Blocks in one page share a namespace and run top to bottom, the way a reader would
type them. Two directives control this, both written as the first line of a block:

``# docs: skip``
    Do not run this one. For deliberate errors, shell commands shown as Python, or
    illustrative fragments that were never meant to execute.

``# docs: requires <module>``
    Run only if that module imports. Keeps torch and sklearn examples in the docs
    without making them dependencies of the docs build.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
BLOCK = re.compile(r"^```python\b[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
REQUIRES = re.compile(r"^#\s*docs:\s*requires\s+([\w.]+)\s*$", re.MULTILINE)
SKIP = re.compile(r"^#\s*docs:\s*skip\s*$", re.MULTILINE)


def _pages() -> list[Path]:
    return sorted(p for p in DOCS.rglob("*.md")) if DOCS.is_dir() else []


def _runnable_blocks(text: str) -> list[str]:
    """Code blocks that should execute, in the order a reader would meet them."""
    return [b for b in BLOCK.findall(text) if not SKIP.search(b)]


@pytest.mark.parametrize("page", _pages(), ids=lambda p: str(p.relative_to(DOCS)))
def test_documentation_snippets_run(page: Path) -> None:
    blocks = _runnable_blocks(page.read_text(encoding="utf-8"))
    if not blocks:
        pytest.skip("no runnable Python in this page")

    namespace: dict[str, object] = {"__name__": "__docs__"}
    for i, block in enumerate(blocks, start=1):
        missing = [m for m in REQUIRES.findall(block) if importlib.util.find_spec(m) is None]
        if missing:
            continue  # the page keeps working without the optional extra
        try:
            exec(compile(block, f"{page.name}#block{i}", "exec"), namespace)  # noqa: S102
        except Exception as exc:
            numbered = "\n".join(
                f"  {n:>3} | {line}" for n, line in enumerate(block.splitlines(), 1)
            )
            raise AssertionError(
                f"{page.relative_to(DOCS)} block {i} raised {type(exc).__name__}: {exc}\n{numbered}"
            ) from exc


def test_every_page_is_reachable_from_the_nav() -> None:
    """A page nobody links to is a page nobody reads, and nobody maintains."""
    config = (DOCS.parent / "mkdocs.yml").read_text(encoding="utf-8")
    nav = config.split("\nnav:", 1)[1]
    orphans = [
        str(p.relative_to(DOCS)).replace("\\", "/")
        for p in _pages()
        if str(p.relative_to(DOCS)).replace("\\", "/") not in nav
    ]
    assert not orphans, f"pages missing from mkdocs.yml nav: {orphans}"


def test_nav_lists_no_page_that_does_not_exist() -> None:
    """The other direction: a nav entry pointing at nothing breaks the build."""
    config = (DOCS.parent / "mkdocs.yml").read_text(encoding="utf-8")
    nav = config.split("\nnav:", 1)[1]
    referenced = set(re.findall(r"([\w/.-]+\.md)", nav))
    missing = sorted(name for name in referenced if not (DOCS / name).is_file())
    assert not missing, f"mkdocs.yml nav points at missing pages: {missing}"
