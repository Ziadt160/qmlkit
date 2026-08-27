"""A composable vocabulary for building ansätze.

Most published ansätze are a handful of primitives stacked in a pattern. Writing
them as expressions in that vocabulary — rather than as bespoke functions — means a
new one costs a line and inherits everything: correct parameter-shift rules,
resource counting, a torch layer, and a place in the registry.

    hardware_efficient = repeat(n_layers, RotationLayer(("ry", "rz")) + EntanglerLayer("cx"))

Blocks allocate their own parameters through a build context, so nothing is ever
hand-counted. :func:`share` re-uses one set of parameters across repetitions —
weight tying — which the gradient code already handles per occurrence.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager

from qmlkit.core.builder import QCircuit, entangler_pairs
from qmlkit.core.gates import get_gate
from qmlkit.core.ir import ParamRef
from qmlkit.utils.errors import unknown

__all__ = [
    "Block",
    "BuildContext",
    "RotationLayer",
    "EntanglerLayer",
    "ParametricEntangler",
    "PoolLayer",
    "EncodingLayer",
    "Sequential",
    "Repeat",
    "Share",
    "Custom",
    "repeat",
    "share",
]


class BuildContext:
    """Hands out parameter slots and tracks which qubits are still active.

    ``active`` shrinks as pooling layers halve the register (the QCNN pattern), so
    later blocks act on what is left rather than on the full width.
    """

    def __init__(self, n_qubits: int, n_inputs: int = 0) -> None:
        self.n_qubits = n_qubits
        self.n_inputs = n_inputs
        self.active: list[int] = list(range(n_qubits))
        # inputs occupy the first n_inputs indices; weights follow. That split is what
        # lets one flat vector carry both while df/dx and df/dtheta stay separable.
        self._next = n_inputs
        self._replay: list[int] | None = None
        self._replay_pos = 0
        self._log: list[int] = []

    def input_ref(self, i: int, scale: float = 1.0, offset: float = 0.0) -> ParamRef:
        """Reference encoding angle ``i``. Re-uploads reuse the same reference."""
        if not 0 <= i < self.n_inputs:
            raise IndexError(f"input {i} out of range for n_inputs={self.n_inputs}")
        return ParamRef(i, scale, offset)

    def new_param(self, scale: float = 1.0, offset: float = 0.0) -> ParamRef:
        """Allocate a parameter — or replay a shared one, when tying weights."""
        if self._replay is not None:
            index = self._replay[self._replay_pos % len(self._replay)]
            self._replay_pos += 1
        else:
            index = self._next
            self._next += 1
            self._log.append(index)
        return ParamRef(index, scale, offset)

    @property
    def n_params(self) -> int:
        """Total parameters, inputs included."""
        return self._next

    @property
    def n_weights(self) -> int:
        """Trainable parameters only."""
        return self._next - self.n_inputs

    def mark(self) -> int:
        return len(self._log)

    def since(self, mark: int) -> list[int]:
        return self._log[mark:]

    @contextmanager
    def replaying(self, indices: Sequence[int]) -> Iterator[None]:
        """Re-issue these parameter indices instead of allocating new ones."""
        prev, prev_pos = self._replay, self._replay_pos
        self._replay, self._replay_pos = list(indices), 0
        try:
            yield
        finally:
            self._replay, self._replay_pos = prev, prev_pos


class Block:
    """One piece of an ansatz. Compose with ``+``."""

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def __add__(self, other: Block) -> Sequential:
        left = self.blocks if isinstance(self, Sequential) else (self,)
        right = other.blocks if isinstance(other, Sequential) else (other,)
        return Sequential(left + right)

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Sequential(Block):
    """Blocks applied in order."""

    def __init__(self, blocks: Sequence[Block]) -> None:
        self.blocks = tuple(blocks)

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        for b in self.blocks:
            b.emit(qc, ctx)

    def __repr__(self) -> str:
        return " + ".join(repr(b) for b in self.blocks)


class RotationLayer(Block):
    """One trainable rotation per gate per active wire."""

    def __init__(self, gates: str | Sequence[str] = ("ry",), wires: Sequence[int] | None = None):
        self.gates = (gates,) if isinstance(gates, str) else tuple(gates)
        self.wires = tuple(wires) if wires is not None else None
        for g in self.gates:
            if not get_gate(g).is_parametric:
                raise ValueError(f"{g!r} takes no parameters; use EntanglerLayer for fixed gates")

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        wires = self.wires if self.wires is not None else ctx.active
        for q in wires:
            for g in self.gates:
                qc.apply(g, q, ctx.new_param())

    def __repr__(self) -> str:
        return f"RotationLayer({list(self.gates)})"


class EntanglerLayer(Block):
    """A layer of fixed two-qubit gates following a named pattern."""

    def __init__(self, gate: str = "cx", pattern: str = "chain"):
        if get_gate(gate).is_parametric:
            raise ValueError(f"{gate!r} is parameterised; use ParametricEntangler")
        self.gate = gate
        self.pattern = pattern

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        for a, b in entangler_pairs(len(wires), self.pattern):
            qc.apply(self.gate, (wires[a], wires[b]))

    def __repr__(self) -> str:
        return f"EntanglerLayer({self.gate!r}, {self.pattern!r})"


class ParametricEntangler(Block):
    """A layer of *trainable* two-qubit gates — exercises the four-term shift rule."""

    def __init__(self, gate: str = "crz", pattern: str = "ring"):
        if not get_gate(gate).is_parametric:
            raise ValueError(f"{gate!r} takes no parameters; use EntanglerLayer")
        self.gate = gate
        self.pattern = pattern

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        for a, b in entangler_pairs(len(wires), self.pattern):
            qc.apply(self.gate, (wires[a], wires[b]), ctx.new_param())

    def __repr__(self) -> str:
        return f"ParametricEntangler({self.gate!r}, {self.pattern!r})"


class PoolLayer(Block):
    """Halve the active register — the pooling half of a QCNN.

    ``keep="odd"`` retains every second wire starting from the second, matching the
    convention where the surviving qubit is the target of the preceding entangler.

    Two pooling modes, because the literature uses both:

    ``mode="discard"``
        Simply stop using the wire. Cheap, adds no parameters, and the information
        it held survives only through whatever the convolution already moved.

    ``mode="controlled"``
        Before dropping a wire, apply a trainable ``crz`` from it onto its surviving
        partner, so pooling *learns* what to carry forward. This is the simulator's
        stand-in for the measure-and-conditionally-rotate pooling of Cong, Choi &
        Lukin, which needs mid-circuit measurement and feed-forward.

    ``tied`` shares one pooling angle across the whole layer, matching the way a
    tied convolution shares one filter.
    """

    def __init__(self, keep: str = "odd", mode: str = "discard", tied: bool = True):
        if keep not in ("even", "odd"):
            raise unknown("keep", keep, ("even", "odd"))
        if mode not in ("discard", "controlled"):
            raise unknown("mode", mode, ("discard", "controlled"))
        self.keep = keep
        self.mode = mode
        self.tied = tied

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        if len(ctx.active) <= 1:
            return
        survivors = ctx.active[1::2] if self.keep == "odd" else ctx.active[0::2]
        if self.mode == "controlled":
            dropped = [w for w in ctx.active if w not in set(survivors)]
            shared = ctx.new_param() if self.tied else None
            for source, target in zip(dropped, survivors, strict=False):
                qc.crz(source, target, shared if shared is not None else ctx.new_param())
        ctx.active = survivors

    def __repr__(self) -> str:
        return f"PoolLayer({self.keep!r}, mode={self.mode!r})"


class Custom(Block):
    """Wrap an arbitrary ``fn(qc, ctx)`` as a block.

    The escape hatch: anything the vocabulary cannot express, written directly
    against the builder, still composes with everything else.
    """

    def __init__(self, fn: Callable[[QCircuit, BuildContext], None], name: str = "Custom"):
        self.fn = fn
        self.name = name

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        self.fn(qc, ctx)

    def __repr__(self) -> str:
        return f"Custom({self.name!r})"


class EncodingLayer(Block):
    """Insert a feature map as a block, so re-uploading is just composition.

    Data re-uploading is not one structure. It is *any* interleaving of an encoding
    with a trainable block, and which encoding, which block, and in what order are
    all design choices:

        repeat(3, EncodingLayer(fmap) + RotationLayer(("rz", "ry", "rz")) + EntanglerLayer())
        repeat(2, RotationLayer("ry") + EncodingLayer(fmap))          # W before S
        EncodingLayer(fmap) + repeat(4, RotationLayer("ry"))          # encode once, vary often
        EncodingLayer(zz) + RotationLayer("ry") + EncodingLayer(angle)  # two different maps

    Every repeat references the **same** input angles: re-uploading means feeding the
    same data in again, not consuming new features.

    To learn the *frequencies* rather than inherit them from the encoding, put a
    classical layer in front — ``nn.Sequential(nn.Linear(d, d), QuantumLayer(...))``.
    That is a scaling of the inputs, which torch already differentiates; it does not
    need to live inside the circuit.
    """

    def __init__(self, feature_map: object) -> None:
        self.feature_map = feature_map

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        n_angles = int(self.feature_map.n_angles)  # type: ignore[attr-defined]
        if ctx.n_inputs < n_angles:
            raise ValueError(
                f"the circuit reserves {ctx.n_inputs} input slots but this feature map "
                f"needs {n_angles}; build it with reupload() or pass n_inputs to Ansatz"
            )
        spec = self.feature_map._emit(  # type: ignore[attr-defined]
            [ctx.input_ref(i) for i in range(n_angles)]
        )
        for op in spec.ops:
            qc.apply(op.gate, op.qubits, *op.params)

    def __repr__(self) -> str:
        return f"EncodingLayer({type(self.feature_map).__name__})"


class Repeat(Block):
    """Apply a block ``times`` times, with **fresh** parameters each time."""

    def __init__(self, times: int, block: Block):
        if times < 1:
            raise ValueError("times must be at least 1")
        self.times = times
        self.block = block

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        for _ in range(self.times):
            self.block.emit(qc, ctx)

    def __repr__(self) -> str:
        return f"repeat({self.times}, {self.block!r})"


class Share(Block):
    """Apply a block ``times`` times, **re-using** the same parameters.

    Weight tying, as a QCNN's shared convolution filter does. The gradient code
    sums over each occurrence separately, which is the whole reason the IR tracks
    slots rather than just parameters.
    """

    def __init__(self, times: int, block: Block):
        if times < 1:
            raise ValueError("times must be at least 1")
        self.times = times
        self.block = block

    def emit(self, qc: QCircuit, ctx: BuildContext) -> None:
        mark = ctx.mark()
        self.block.emit(qc, ctx)
        allocated = ctx.since(mark)
        if not allocated:
            for _ in range(self.times - 1):
                self.block.emit(qc, ctx)
            return
        for _ in range(self.times - 1):
            with ctx.replaying(allocated):
                self.block.emit(qc, ctx)

    def __repr__(self) -> str:
        return f"share({self.times}, {self.block!r})"


def repeat(times: int, block: Block) -> Repeat:
    """``repeat(3, RotationLayer("ry"))`` — three layers, fresh weights each."""
    return Repeat(times, block)


def share(times: int, block: Block) -> Share:
    """``share(3, conv_block)`` — three applications of one tied weight set."""
    return Share(times, block)
