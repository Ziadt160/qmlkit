"""Look at the circuit — the ``qml.draw`` / ``qml.specs`` equivalent.

Plain text, no matplotlib, no optional dependency. Reads the IR, so it works for any
circuit the library can build and shows exactly what a backend will run.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.gates import get_gate
from qmlkit.core.ir import CircuitSpec, ParamLike, ParamRef

__all__ = ["draw", "specs"]

#: The diagram's glyphs, and what they degrade to when the output stream cannot encode
#: them. The ASCII forms are chosen to keep column widths identical so the fallback
#: lines up exactly like the Unicode one.
_ASCII = {
    "\u2500": "-",  # horizontal wire
    "\u2502": "|",  # vertical link between a control and its target
    "\u2020": "+",  # the dagger in S-dagger / T-dagger
    "\u03b8": "t",  # theta, in a parameter label
    "\u00b7": "*",  # the multiplication dot in a scaled parameter
    "\u2588": "#",  # the solid block in a probability histogram
}


def _stream_handles_unicode() -> bool:
    """Whether stdout can encode the glyphs the diagram uses.

    A Windows console defaults to cp1252 and encodes none of them. Checking beats
    catching, because the failure happens at *print* time — inside the caller, with a
    traceback that names the codec and not this module.
    """
    encoding = getattr(sys.stdout, "encoding", None)
    if not encoding:
        return False
    try:
        "".join(_ASCII).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def _downgrade(text: str) -> str:
    for glyph, plain in _ASCII.items():
        text = text.replace(glyph, plain)
    return text


_LABEL = {
    "cx": "X",
    "cy": "Y",
    "cz": "Z",
    "swap": "x",
    "crx": "RX",
    "cry": "RY",
    "crz": "RZ",
}


def _cell(gate: str, params: Sequence[ParamLike]) -> str:
    name = {"sdg": "S†", "tdg": "T†", "phase": "P"}.get(gate, gate.upper())
    if not params:
        return name
    p = params[0]
    if isinstance(p, ParamRef):
        body = f"θ{p.index}" if p.scale == 1.0 and p.offset == 0.0 else f"{p.scale:g}·θ{p.index}"
    else:
        body = f"{float(p):.2f}"
    return f"{name}({body})"


def draw(spec: CircuitSpec, max_width: int = 160, ascii: bool | None = None) -> str:
    """A text diagram of the circuit.

    print(qk.draw(qk.hardware_efficient(3, 1).build()))

    The diagram uses box-drawing glyphs. ``ascii=None`` (the default) checks whether
    ``sys.stdout`` can encode them and degrades to ``-``, ``|`` and ``t`` when it
    cannot: a Windows console defaults to cp1252, which encodes none of them, and
    printing the result would otherwise raise ``UnicodeEncodeError`` from inside the
    caller, with a traceback naming the codec rather than this function. Pass
    ``ascii=True`` or ``ascii=False`` to decide for yourself. Column widths are
    identical either way, so the fallback lines up exactly like the Unicode form.
    """
    n = spec.n_qubits
    # pack operations into columns so nothing overlaps on a wire
    columns: list[dict[int, str]] = []
    frontier = [0] * n
    spans: list[tuple[int, int, int]] = []  # (column, top, bottom) for two-qubit links

    for op in spec.ops:
        col = max(frontier[q] for q in op.qubits)
        while len(columns) <= col:
            columns.append({})
        g = get_gate(op.gate)
        if g.n_qubits == 1:
            columns[col][op.qubits[0]] = _cell(op.gate, op.params)
        else:
            control, target = op.qubits[0], op.qubits[1]
            columns[col][control] = "@" if op.gate != "swap" else "x"
            columns[col][target] = _cell(_LABEL.get(op.gate, op.gate), op.params)
            spans.append((col, min(op.qubits), max(op.qubits)))
        for q in range(min(op.qubits), max(op.qubits) + 1):
            frontier[q] = col + 1

    widths = [max((len(v) for v in col.values()), default=1) for col in columns] or [1]
    label_w = len(f"q{n - 1}: ")

    wires = [f"q{q}: ".ljust(label_w) for q in range(n)]
    links = {(c, q) for c, top, bottom in spans for q in range(top, bottom + 1)}

    # `cells`, not `col`: `col` is a column *index* in the packing loop above, and
    # reusing the name for the column's contents is what made this look ill-typed
    for c, cells in enumerate(columns):
        w = widths[c]
        for q in range(n):
            if q in cells:
                wires[q] += "─" + cells[q].center(w, "─") + "─"
            elif (c, q) in links:
                wires[q] += "─" + "│".center(w, "─") + "─"
            else:
                wires[q] += "─" * (w + 2)

    out = [w + "─" for w in wires]
    if max(len(line) for line in out) > max_width:
        out = [line[: max_width - 3] + "..." for line in out]
    diagram = "\n".join(out)
    plain = not _stream_handles_unicode() if ascii is None else ascii
    return _downgrade(diagram) if plain else diagram


def specs(spec: CircuitSpec) -> dict[str, object]:
    """Everything worth knowing about a circuit's cost, in one dict."""
    from qmlkit.gradients.parameter_shift import grad_circuit_cost

    out = dict(spec.resources())
    out["grad_circuits_parameter_shift"] = grad_circuit_cost(spec)
    out["grad_passes_adjoint"] = 1
    occurrences = {i: len(spec.occurrences_of(i)) for i in range(spec.n_params)}
    out["n_occurrences"] = occurrences
    tied = {i: c for i, c in occurrences.items() if c > 1}
    out["weight_tied_parameters"] = len(tied)
    return out


def draw_ansatz(ansatz: object, max_width: int = 160) -> str:
    """Convenience: draw an :class:`~qmlkit.ansatz.library.Ansatz` unbound."""
    return draw(ansatz.build(), max_width)  # type: ignore[attr-defined]


def probabilities_bar(
    probs: npt.NDArray[Any],
    n_qubits: int,
    top: int = 8,
    width: int = 30,
    ascii: bool | None = None,
) -> str:
    """A text histogram of outcome probabilities — the most likely bitstrings."""
    p = np.asarray(probs, dtype=float).ravel()
    order = np.argsort(p)[::-1][:top]
    lines = []
    for i in order:
        if p[i] < 1e-9:
            continue
        bar = "█" * max(1, int(round(p[i] * width)))
        lines.append(f"  |{format(int(i), f'0{n_qubits}b')}>  {p[i]:.4f}  {bar}")
    histogram = "\n".join(lines)
    plain = not _stream_handles_unicode() if ascii is None else ascii
    return _downgrade(histogram) if plain else histogram
