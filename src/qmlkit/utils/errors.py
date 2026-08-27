"""Error messages written for a reader who will not go and look it up.

Most of this library's callers now are language models writing code, and they
work in a tight loop: guess an API, run it, read the traceback, try again. That
makes the exception the primary documentation — the only page that is guaranteed
to be read, at exactly the moment it is needed.

So an error about a *name* should answer three questions in one line:

1. What was wrong (``unknown gradient method 'parameter_shift'``)
2. What was probably meant (``Did you mean 'parameter-shift'?``)
3. What is actually allowed (``Valid: adjoint, backprop, ...``)

Answering only (1) costs the caller a round trip through the docs. Answering all
three usually costs them nothing: the next attempt is correct.

The near-match is deliberately more forgiving than :func:`difflib.get_close_matches`
alone. The two mistakes that dominate are separator and case drift — ``snake_case``
where the library uses ``kebab-case``, or ``Parameter-Shift`` for ``parameter-shift``
— because they are what a model produces when it half-remembers a name from a
different library. Those resolve to a single unambiguous suggestion.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

__all__ = ["UnknownName", "did_you_mean", "unknown", "wrong_size"]


class UnknownName(KeyError):
    """A :class:`KeyError` whose message survives the traceback intact.

    ``KeyError`` is alone among the builtins in rendering as ``repr(args[0])``
    rather than the message itself, so a sentence containing quoted names comes
    out escaped:

        KeyError: 'unknown gate \'cnott\'. Did you mean \'cnot\'? ...'

    Every quote in a message built to be *read* is exactly the thing that gets
    mangled. Restoring ``__str__`` fixes the rendering while leaving ``except
    KeyError`` and ``pytest.raises(KeyError)`` working, since this is still one.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _squash(text: str) -> str:
    """Collapse the differences that are never meaningful: case and separators."""
    return text.lower().replace("-", "").replace("_", "").replace(" ", "")


def did_you_mean(got: object, valid: Iterable[str], n: int = 3) -> tuple[str, ...]:
    """The closest valid spellings of ``got``, best first, possibly empty.

    A name that differs only by case or separator is treated as certain and
    returned alone; anything else falls back to fuzzy matching.

    >>> did_you_mean("parameter_shift", ["parameter-shift", "adjoint"])
    ('parameter-shift',)
    >>> did_you_mean("adjiont", ["parameter-shift", "adjoint"])
    ('adjoint',)
    >>> did_you_mean("wildly-different", ["adjoint"])
    ()
    """
    options = list(dict.fromkeys(str(v) for v in valid))
    text = str(got)
    squashed = [v for v in options if _squash(v) == _squash(text)]
    if squashed:
        return tuple(squashed[:n])
    return tuple(difflib.get_close_matches(text, options, n=n, cutoff=0.6))


def unknown(
    kind: str,
    got: object,
    valid: Iterable[str],
    *,
    hint: str | None = None,
    error: type[Exception] = ValueError,
) -> Exception:
    """Build (do not raise) the exception for an unrecognised name.

    Call it as ``raise unknown("gradient method", name, list_gradient_methods())``.
    Returning rather than raising keeps the ``raise`` visible at the call site, so
    static analysis and readers can both still see the control flow.

    ``hint`` is appended verbatim, for the cases where knowing the valid names is
    not enough to know what to do next.

    Asking for ``error=KeyError`` gets :class:`UnknownName`, which is one, but
    prints its message rather than the repr of it.
    """
    if error is KeyError:  # keep the quotes in the message readable
        error = UnknownName
    options = sorted(dict.fromkeys(str(v) for v in valid))
    near = did_you_mean(got, options)
    parts = [f"unknown {kind} {got!r}."]
    if near:
        parts.append("Did you mean " + " or ".join(repr(s) for s in near) + "?")
    if options:
        parts.append(f"Valid: {', '.join(options)}.")
    if hint:
        parts.append(hint)
    return error(" ".join(parts))


def wrong_size(
    what: str,
    expected: int,
    got: int,
    *,
    unit: str = "value",
    hint: str | None = None,
    error: type[Exception] = ValueError,
) -> Exception:
    """Build (do not raise) the exception for a length or width mismatch.

    Size errors are the other half of the repair loop, and the same rule applies:
    say what the shapes were *and* what to change. ``hint`` should name a concrete
    edit — a different constructor argument, a reducer to insert — not a restatement
    of the problem.
    """
    plural = unit if expected == 1 else f"{unit}s"
    parts = [f"{what} expects {expected} {plural}, got {got}."]
    if hint:
        parts.append(hint)
    return error(" ".join(parts))
