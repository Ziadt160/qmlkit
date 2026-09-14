"""The numbers the documentation asserts about this repository, checked against it.

Hand-counted facts drift. `docs/llms.txt` does not, because it is generated and CI
fails on a stale copy; the counts scattered through `README.md`, `AGENTS.md` and
`docs/about/validation.md` had no such gate, and three of them were wrong when this
script was written:

* ``validation.md`` claimed 1,563 tests collected; the suite collected 1,729.
* ``AGENTS.md`` said "seven" backends and ``guides/backends.md`` said "eight"; the
  registry had nine.
* ``README.md``'s ``backend_report()`` sample listed seven, missing ``aer`` and ``mps``.

Run it after anything that changes the shape of the library::

    python scripts/check_doc_numbers.py            # report, exit 1 if anything drifted
    python scripts/check_doc_numbers.py --counts   # just print the true numbers

It deliberately does **not** rewrite the prose. A number in this project carries an
argument around it — "nine backends behind one protocol" is a sentence, not a field —
and a script that edited those sentences would flatten the voice the docs are written
in. It tells you what moved; you decide what the sentence should now say.

The test-suite counts are the slow ones, so they are opt-in::

    python scripts/check_doc_numbers.py --tests    # runs pytest --collect-only
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def measured() -> dict[str, int]:
    """Everything checkable without running the suite."""
    sys.path.insert(0, str(ROOT / "src"))
    import qmlkit as qk
    from qmlkit.core.gates import list_gates

    src = list((ROOT / "src" / "qmlkit").rglob("*.py"))
    return {
        "backends": len(qk.list_backends()),
        "gates": len(list_gates()),
        "public_names": len(qk.__all__),
        "modules": len(src),
        "src_lines": sum(len(f.read_text(encoding="utf-8").splitlines()) for f in src),
        "doc_pages": len(list((ROOT / "docs").rglob("*.md"))),
        "densesim_lines": len((ROOT / "tests" / "densesim.py").read_text(encoding="utf-8").splitlines()),
        "torture_invariants": len(
            re.findall(r"^def test_", (ROOT / "tests" / "test_torture.py").read_text(encoding="utf-8"), re.M)
        ),
    }


def collected_tests() -> int:
    """How many tests the suite collects. Slow; opt-in behind ``--tests``."""
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    ).stdout
    return sum(int(n) for n in re.findall(r"^tests/\S+: (\d+)$", out, re.M))


#: doc claim -> (measured key, the regex that finds it). Each pattern captures the
#: number so a mismatch can report both sides rather than just "something is wrong".
CLAIMS: list[tuple[str, str, str]] = [
    ("AGENTS.md", "public_names", r"(\d+) names in the top-level"),
    ("AGENTS.md", "modules", r"over (\d+) modules"),
    ("docs/about/validation.md", "densesim_lines", r"`tests/densesim\.py` — (\d+) lines"),
    ("docs/about/validation.md", "torture_invariants", r"states (\d+) invariants"),
    ("docs/about/validation.md", "gates", r"all (\d+) gates"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--counts", action="store_true", help="print the true numbers and exit")
    ap.add_argument("--tests", action="store_true", help="also collect the test suite (slow)")
    args = ap.parse_args()

    truth = measured()
    if args.tests:
        truth["collected_tests"] = collected_tests()

    if args.counts:
        for key, value in sorted(truth.items()):
            print(f"{key:<20} {value:>8,}")
        return 0

    drifted = 0
    for page, key, pattern in CLAIMS:
        text = (ROOT / page).read_text(encoding="utf-8")
        found = re.search(pattern, text)
        if found is None:
            print(f"[gone]  {page}: pattern for {key!r} no longer matches -- claim moved or reworded")
            drifted += 1
            continue
        claimed = int(found.group(1).replace(",", ""))
        if claimed != truth[key]:
            print(f"[stale] {page}: says {claimed:,} {key}, measured {truth[key]:,}")
            drifted += 1

    if args.tests:
        text = (ROOT / "docs/about/validation.md").read_text(encoding="utf-8")
        found = re.search(r"of ([\d,]+) collected", text)
        if found:
            claimed = int(found.group(1).replace(",", ""))
            if claimed != truth["collected_tests"]:
                print(
                    f"[stale] docs/about/validation.md: says {claimed:,} tests collected, "
                    f"measured {truth['collected_tests']:,}"
                )
                drifted += 1

    if drifted:
        print(f"\n{drifted} claim(s) drifted. Fix the prose; do not just change the digit.")
        return 1
    print("every checked claim matches the repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
