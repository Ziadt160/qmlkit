"""Reading circuits in, so people can migrate rather than only start here.

``to_qiskit``, ``to_cirq`` and ``to_spinqit`` have always existed. The reverse did
not, and one-way interop is the difference between a library someone *tries* and one
someone *adopts*: an existing project has circuits already, and a tool that cannot
read them asks for a rewrite before it has proved anything.

Three entry points, in order of how much they carry:

:func:`from_qasm`
    OpenQASM 2.0, parsed with the **standard library alone**. Qiskit, Cirq, Braket,
    t|ket> and Q# all export it, so this one function reaches every one of them
    without qmlkit taking a dependency on any.
:func:`from_qiskit`
    A ``QuantumCircuit`` object directly, including unbound ``Parameter``s, which
    QASM cannot represent.
:func:`from_pennylane`
    A tape or a QNode, for the migration this library is most often compared against.

**Qubit order is the thing to get right.** qmlkit is big-endian: qubit 0 is the most
significant bit of a basis state. Qiskit and OpenQASM are little-endian, so importing
maps their qubit ``j`` to qmlkit's ``n-1-j`` — the exact inverse of what
:meth:`~qmlkit.core.backends.qiskit_backend.QiskitBackend.to_qiskit` does on the way
out, which is what makes the round trip exact rather than merely plausible.
PennyLane is big-endian like qmlkit, so nothing is flipped there. All three are
asserted against real statevectors in ``tests/test_import.py``.

**Gates outside the supported set are refused, not approximated.** The one exception
is the ``u``/``u3`` family, which is decomposed into rotations and drops a global
phase — unobservable on its own, and observable if the circuit is later used as a
*controlled* block, so the import says so.
"""

from __future__ import annotations

import ast
import math
import re
import warnings
from collections.abc import Callable, Sequence
from typing import Any

from qmlkit.core.ir import CircuitSpec, Op, ParamRef

__all__ = [
    "from_qasm",
    "from_qiskit",
    "from_pennylane",
    "register_importer",
    "list_importers",
    "get_importer",
    "UnsupportedGate",
]


class UnsupportedGate(ValueError):
    """A source circuit used a gate qmlkit has no definition for."""


#: OpenQASM 2.0 / qelib1 names to qmlkit gates. Names not here are either refused or
#: handled by :func:`_decompose_u`.
_QASM_GATES = {
    "id": "i",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "p": "phase",
    "u1": "phase",
    "cx": "cx",
    "CX": "cx",
    "cy": "cy",
    "cz": "cz",
    "swap": "swap",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
}

#: Qiskit's ``Instruction.name`` is lowercase and mostly matches QASM.
_QISKIT_GATES = dict(_QASM_GATES)

#: PennyLane operation names.
_PENNYLANE_GATES = {
    "Identity": "i",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "X": "x",
    "Y": "y",
    "Z": "z",
    "Hadamard": "h",
    "S": "s",
    "T": "t",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "PhaseShift": "phase",
    "U1": "phase",
    "CNOT": "cx",
    "CY": "cy",
    "CZ": "cz",
    "SWAP": "swap",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
}

#: Gates that are silently dropped rather than refused: they change nothing qmlkit
#: models. ``barrier`` is a transpiler hint; ``delay`` is timing on real hardware.
_IGNORED = {"barrier", "delay", "id_placeholder"}

_GLOBAL_PHASE_NOTE = (
    "decomposed into rotations, which drops an overall phase. That is unobservable "
    "for this circuit on its own, but becomes a *relative* phase if the circuit is "
    "later used inside a controlled block."
)


# --------------------------------------------------------------------------- #
# angle expressions
# --------------------------------------------------------------------------- #
_ALLOWED_NAMES = {"pi": math.pi, "tau": math.tau, "e": math.e}


def _eval_angle(text: str) -> float:
    """Evaluate a QASM angle expression safely.

    QASM writes angles as ``pi/2``, ``-2*pi/3``, ``0.7``. Parsing them with
    :func:`eval` would execute whatever the file contained, so the expression is
    parsed to an AST and only arithmetic over numbers and ``pi``/``tau``/``e`` is
    walked. Anything else — a call, a name, an attribute — is refused.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse the angle expression {text!r}") from exc
    return _walk(tree.body, text)


def _walk(node: ast.AST, source: str) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"{source!r} is not a numeric angle")
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id not in _ALLOWED_NAMES:
            raise ValueError(
                f"angle {source!r} uses the name {node.id!r}; only "
                f"{', '.join(sorted(_ALLOWED_NAMES))} and numbers are allowed"
            )
        return _ALLOWED_NAMES[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _walk(node.operand, source)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = _walk(node.left, source), _walk(node.right, source)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return float(left**right)
    raise ValueError(f"angle {source!r} uses an operation that is not plain arithmetic")


# --------------------------------------------------------------------------- #
# the shared builder
# --------------------------------------------------------------------------- #
class _Importer:
    """Accumulates ops, applying one endianness convention and one gate table."""

    def __init__(self, n_qubits: int, table: dict[str, str], source: str, flip: bool) -> None:
        self.n_qubits = n_qubits
        self.table = table
        self.source = source
        self.flip = flip
        self.ops: list[Op] = []
        self.n_params = 0
        self.decomposed = False

    def wire(self, index: int) -> int:
        """Source qubit index to qmlkit's, honouring the endianness convention."""
        if not 0 <= index < self.n_qubits:
            raise ValueError(
                f"{self.source} refers to qubit {index}, outside the "
                f"{self.n_qubits}-qubit register"
            )
        return self.n_qubits - 1 - index if self.flip else index

    def add(self, name: str, qubits: Sequence[int], params: Sequence[Any] = ()) -> None:
        """One source instruction, translated or refused."""
        if name in _IGNORED:
            return
        wires = tuple(self.wire(q) for q in qubits)

        if name in ("u", "u3", "u2", "U"):
            self._decompose_u(name, wires, params)
            return
        if name in ("measure", "reset"):
            raise UnsupportedGate(
                f"{self.source} contains {name!r}. qmlkit has no mid-circuit measurement "
                "or reset in the 0.x line, so this circuit cannot be represented. Remove "
                "the measurements and read out with an observable instead."
            )
        if name not in self.table:
            from qmlkit.utils.errors import did_you_mean

            near = did_you_mean(name, self.table)
            hint = (
                " Did you mean " + " or ".join(repr(s) for s in near) + "?"
                if near
                else " Register it with qk.register_gate(), or transpile the circuit to "
                f"the supported basis first: {', '.join(sorted(self.table))}."
            )
            raise UnsupportedGate(f"{self.source} uses gate {name!r}, which qmlkit has no "
                                  f"mapping for.{hint}")
        self.ops.append(Op(self.table[name], wires, tuple(params)))

    def _decompose_u(self, name: str, wires: tuple[int, ...], params: Sequence[Any]) -> None:
        """``u3(t, p, l) = Rz(p) Ry(t) Rz(l)``, up to a global phase.

        Qiskit's ``u3`` carries an ``exp(i(p+l)/2)`` factor this decomposition drops.
        The import warns once rather than silently changing what a controlled copy of
        the circuit would do.
        """
        if any(isinstance(p, ParamRef) for p in params):
            raise UnsupportedGate(
                f"{self.source} has an unbound parameter inside {name!r}. The u-family is "
                "decomposed into rotations, which cannot be done symbolically here — bind "
                "the parameters first, or rebuild the circuit from rx/ry/rz."
            )
        values = [float(p) for p in params]
        if name == "u2":  # u2(phi, lam) == u3(pi/2, phi, lam)
            values = [math.pi / 2, *values]
        theta, phi, lam = (values + [0.0, 0.0, 0.0])[:3]
        self.decomposed = True
        for gate, angle in (("rz", lam), ("ry", theta), ("rz", phi)):
            if angle != 0.0:
                self.ops.append(Op(gate, wires, (angle,)))

    def finish(self) -> CircuitSpec:
        if self.decomposed:
            warnings.warn(
                f"{self.source}: a u/u3 gate was {_GLOBAL_PHASE_NOTE}",
                UserWarning,
                stacklevel=3,
            )
        return CircuitSpec(self.n_qubits, tuple(self.ops), self.n_params)


# --------------------------------------------------------------------------- #
# OpenQASM 2.0
# --------------------------------------------------------------------------- #
_QREG = re.compile(r"^\s*qreg\s+(\w+)\s*\[\s*(\d+)\s*\]\s*;")
_CREG = re.compile(r"^\s*creg\s+\w+\s*\[\s*\d+\s*\]\s*;")
_NAME = re.compile(r"^\s*(\w+)\s*")


def _split_instruction(line: str) -> tuple[str, str, str] | None:
    """``name``, the parenthesised argument text, and the qubit targets.

    Written by hand rather than as one regex because ``[^)]*`` truncates a nested
    expression like ``sin(x)`` at the first ``)``, which turns a clear refusal into a
    confusing parse error about half an expression.
    """
    body = line.strip().rstrip(";").strip()
    if not body:
        return None
    match = _NAME.match(body)
    if match is None:
        return None
    name, rest = match.group(1), body[match.end() :].lstrip()
    args = ""
    if rest.startswith("("):
        depth = 0
        for i, char in enumerate(rest):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                args, rest = rest[1:i], rest[i + 1 :].lstrip()
                break
        else:
            raise ValueError(f"unbalanced parentheses in the QASM line: {line.strip()!r}")
    return name, args, rest
_ARG = re.compile(r"(\w+)\s*\[\s*(\d+)\s*\]")


def from_qasm(text: str, little_endian: bool = True) -> CircuitSpec:
    """Parse OpenQASM 2.0 into a :class:`~qmlkit.core.ir.CircuitSpec`.

    Uses the standard library only — no Qiskit, no parser generator — so this works
    in a bare ``pip install qmlkit``. Every major SDK exports QASM 2.0, which makes
    this the widest import path the library has.

    Parameters
    ----------
    text:
        The QASM source. ``OPENQASM``/``include`` headers, comments, ``creg`` and
        ``barrier`` are accepted and ignored.
    little_endian:
        Whether the producing tool treats qubit ``0`` as the least significant bit.
        True for Qiskit and for QASM as it is normally used, which is why it is the
        default. Set False for a big-endian producer, where indices pass through.

    Raises
    ------
    UnsupportedGate
        For a gate qmlkit has no definition for, or for ``measure``/``reset``, naming
        what was found rather than dropping it.

    Notes
    -----
    Only a single quantum register is supported, which is what exported circuits
    almost always have. Classical registers are ignored, since nothing here is
    conditioned on them.
    """
    lines = _strip_comments(text)
    n_qubits, register = _find_register(lines)
    imp = _Importer(n_qubits, _QASM_GATES, "the QASM circuit", flip=little_endian)

    for line in lines:
        if not line.strip() or _QREG.match(line) or _CREG.match(line):
            continue
        if line.lstrip().startswith(("OPENQASM", "include", "gate ", "opaque ")):
            if line.lstrip().startswith(("gate ", "opaque ")):
                raise UnsupportedGate(
                    "the QASM circuit declares a custom gate, which this parser does not "
                    "expand. Flatten it with the producing tool first (in Qiskit: "
                    "`circuit.decompose()`), or register it with qk.register_gate()."
                )
            continue

        parsed = _split_instruction(line)
        if parsed is None:
            raise ValueError(f"could not parse the QASM line: {line.strip()!r}")
        name, args, targets = parsed
        if name in ("measure", "reset"):
            # checked before the register scan, so `measure q[0] -> c[0];` reports the
            # measurement rather than complaining about the classical register
            imp.add(name, [], [])
        params = [_eval_angle(a) for a in args.split(",")] if args else []
        qubits = []
        for reg, index in _ARG.findall(targets):
            if reg != register:
                raise ValueError(
                    f"the QASM circuit uses register {reg!r}; only the single register "
                    f"{register!r} is supported"
                )
            qubits.append(int(index))
        if not qubits and name not in _IGNORED:
            raise ValueError(f"could not read qubit arguments from: {line.strip()!r}")
        imp.add(name, qubits, params)
    return imp.finish()


def _strip_comments(text: str) -> list[str]:
    without_block = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return [line.split("//")[0] for line in without_block.splitlines()]


def _find_register(lines: list[str]) -> tuple[int, str]:
    registers = [(m.group(1), int(m.group(2))) for line in lines if (m := _QREG.match(line))]
    if not registers:
        raise ValueError("the QASM circuit declares no quantum register (`qreg q[n];`)")
    if len(registers) > 1:
        names = ", ".join(name for name, _ in registers)
        raise ValueError(
            f"the QASM circuit declares {len(registers)} quantum registers ({names}); "
            "only one is supported. Merge them in the producing tool first."
        )
    name, size = registers[0]
    return size, name


# --------------------------------------------------------------------------- #
# Qiskit
# --------------------------------------------------------------------------- #
def from_qiskit(circuit: Any) -> CircuitSpec:
    """Convert a Qiskit ``QuantumCircuit``, bound or parameterised.

    Unbound ``Parameter`` objects become :class:`~qmlkit.core.ir.ParamRef`\\ s, indexed
    in Qiskit's own sorted-by-name parameter order, so ``theta`` in qmlkit lines up
    with ``circuit.parameters``. That is the part QASM cannot carry, and the reason
    this exists alongside :func:`from_qasm`.

    Qiskit's qubit ``j`` becomes qmlkit's ``n-1-j``, inverting what ``to_qiskit`` does,
    so ``from_qiskit(to_qiskit(spec))`` reproduces the circuit exactly.
    """
    try:
        from qiskit.circuit import Parameter, ParameterExpression
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "from_qiskit needs Qiskit, which is an optional extra:\n"
            "    pip install 'qmlkit[qiskit]'"
        ) from exc

    n = circuit.num_qubits
    imp = _Importer(n, _QISKIT_GATES, "the Qiskit circuit", flip=True)
    index_of = {p: i for i, p in enumerate(circuit.parameters)}
    imp.n_params = len(index_of)

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = [circuit.find_bit(q).index for q in instruction.qubits]
        params: list[Any] = []
        for raw in operation.params:
            if isinstance(raw, ParameterExpression) and raw.parameters:
                symbols = list(raw.parameters)
                if len(symbols) != 1 or not isinstance(raw, Parameter):
                    raise UnsupportedGate(
                        f"the Qiskit circuit has the compound parameter expression {raw!r}. "
                        "qmlkit's ParamRef carries a scale and offset, not arbitrary "
                        "expressions — bind it, or use a plain Parameter."
                    )
                params.append(ParamRef(index_of[symbols[0]]))
            else:
                params.append(float(raw))
        imp.add(operation.name, qubits, params)
    return imp.finish()


# --------------------------------------------------------------------------- #
# PennyLane
# --------------------------------------------------------------------------- #
def from_pennylane(source: Any, *args: Any, **kwargs: Any) -> CircuitSpec:
    """Convert a PennyLane tape, QNode or quantum function.

    A QNode is called with ``*args``/``**kwargs`` to produce its tape, so the
    parameters are bound at import time — PennyLane's trainable parameters are
    positional arguments rather than named symbols, so there is nothing to carry
    across symbolically the way :func:`from_qiskit` does.

    PennyLane orders wires big-endian, the same as qmlkit, so indices pass through
    unchanged. ``tests/test_import.py`` asserts that against real statevectors rather
    than taking it on trust.
    """
    try:
        import pennylane as qml
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "from_pennylane needs PennyLane:\n    pip install pennylane"
        ) from exc

    tape = _as_tape(source, qml, args, kwargs)
    wires = list(tape.wires)
    position = {w: i for i, w in enumerate(wires)}
    imp = _Importer(len(wires), _PENNYLANE_GATES, "the PennyLane circuit", flip=False)

    for op in _flatten(tape.operations):
        imp.add(op.name, [position[w] for w in op.wires], _scalars(op))
    return imp.finish()


def _flatten(operations: Any, depth: int = 0) -> list[Any]:
    """Expand templates into the gates they are made of.

    A PennyLane tape holds ``BasicEntanglerLayers`` as *one* operation carrying a
    weight matrix, not as the rotations it stands for. Anything qmlkit has no direct
    mapping for is asked for its decomposition, recursively — which is how a template
    becomes importable rather than unsupported.
    """
    out: list[Any] = []
    for op in operations:
        if op.name in ("MidMeasureMP", "Measure"):
            raise UnsupportedGate(
                "the PennyLane circuit contains a mid-circuit measurement, which qmlkit "
                "does not model in the 0.x line."
            )
        if op.name in _PENNYLANE_GATES or op.name in _IGNORED:
            out.append(op)
            continue
        if depth >= 8:  # pragma: no cover - a decomposition that deep is pathological
            raise UnsupportedGate(
                f"the PennyLane operation {op.name!r} did not reduce to supported gates "
                "within 8 levels of decomposition"
            )
        try:
            children = op.decomposition()
        except Exception as exc:
            raise UnsupportedGate(
                f"the PennyLane circuit uses {op.name!r}, which qmlkit has no mapping for "
                f"and which does not decompose ({exc})."
            ) from None
        out.extend(_flatten(children, depth + 1))
    return out


def _scalars(op: Any) -> list[float]:
    """Operation parameters as plain floats.

    PennyLane hands back autograd/tensor scalars, and a template that slipped through
    would hand back an array — which is a clearer error here than a TypeError from
    deep inside ``float``.
    """
    import numpy as np

    values = []
    for p in op.parameters:
        arr = np.asarray(p)
        if arr.size != 1:
            raise UnsupportedGate(
                f"the PennyLane operation {op.name!r} carries an array parameter of shape "
                f"{arr.shape}, so it is a template that did not decompose."
            )
        values.append(float(arr.reshape(()).item()))
    return values


def _as_tape(source: Any, qml: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if isinstance(source, qml.tape.QuantumScript):
        return source
    if isinstance(source, qml.QNode):
        # PennyLane has moved this twice: `qml.workflow.construct_tape` is current,
        # `QNode.construct` returned the tape before that, and older versions set
        # `.tape` as a side effect. Try them in that order rather than pinning.
        construct_tape = getattr(getattr(qml, "workflow", None), "construct_tape", None)
        if construct_tape is not None:
            return construct_tape(source)(*args, **kwargs)
        built = source.construct(args, kwargs)
        return built if built is not None else source.tape
    if callable(source):
        with qml.tape.QuantumTape() as tape:
            source(*args, **kwargs)
        return tape
    raise TypeError(
        f"from_pennylane takes a QuantumTape, a QNode or a quantum function, got "
        f"{type(source).__name__}"
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_IMPORTERS: dict[str, Callable[..., CircuitSpec]] = {
    "qasm": from_qasm,
    "qiskit": from_qiskit,
    "pennylane": from_pennylane,
}


def register_importer(name: str, fn: Callable[..., CircuitSpec]) -> None:
    """Add an importer, so a new source format is reachable by name.

    The same registry pattern as ``register_gate``/``register_backend``: registering
    makes the format a first-class citizen of :func:`get_importer`.
    """
    _IMPORTERS[name] = fn


def list_importers() -> tuple[str, ...]:
    return tuple(sorted(_IMPORTERS))


def get_importer(name: str) -> Callable[..., CircuitSpec]:
    if name not in _IMPORTERS:
        from qmlkit.utils.errors import unknown

        raise unknown("importer", name, _IMPORTERS, error=KeyError)
    return _IMPORTERS[name]
