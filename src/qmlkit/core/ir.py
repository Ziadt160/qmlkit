"""The backend-neutral circuit IR.

A circuit is *data*: a list of :class:`Op`. Backends compile it; gradients read it;
resource counting and drawing read it. One representation, and every downstream
capability falls out of it.

The piece that matters most for correctness is the **slot** abstraction. A circuit
has ``n_params`` logical parameters, but those map onto *slots* — one per
(operation, parameter position). A single logical parameter may fill several slots
(weight tying, as in a QCNN's shared convolution block). The parameter-shift rule
must shift **one slot at a time** and sum the results; shifting every occurrence
together computes a different derivative entirely. Making slots explicit here is
what keeps that from being a subtle, silent bug in the gradient code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.core.gates import get_gate

__all__ = ["ParamRef", "Op", "Slot", "CircuitSpec"]


@dataclass(frozen=True)
class ParamRef:
    """A reference to logical parameter ``index``, optionally linearly rescaled.

    ``scale`` and ``offset`` let one logical parameter drive a gate angle of
    ``scale * theta[index] + offset`` without introducing a new parameter. The
    chain rule for that is handled in the gradient code.
    """

    index: int
    scale: float = 1.0
    offset: float = 0.0

    def resolve(self, theta: ArrayLike) -> float:
        values = np.asarray(theta, dtype=float)
        return self.scale * float(values[self.index]) + self.offset


ParamLike = float | int | ParamRef


@dataclass(frozen=True)
class Op:
    """One gate application."""

    gate: str
    qubits: tuple[int, ...]
    params: tuple[ParamLike, ...] = ()

    def __post_init__(self) -> None:
        g = get_gate(self.gate)
        if len(self.qubits) != g.n_qubits:
            raise ValueError(
                f"gate {self.gate!r} acts on {g.n_qubits} qubit(s), got {len(self.qubits)}"
            )
        if len(self.params) != g.n_params:
            raise ValueError(
                f"gate {self.gate!r} takes {g.n_params} parameter(s), got {len(self.params)}"
            )
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"repeated qubit in {self.gate!r} on {self.qubits}")

    @property
    def is_parametric(self) -> bool:
        return bool(self.params)


@dataclass(frozen=True)
class Slot:
    """One concrete (operation, parameter-position) angle site."""

    op_index: int
    param_pos: int
    ref: ParamRef
    gate: str


@dataclass(frozen=True)
class CircuitSpec:
    """An immutable circuit description."""

    n_qubits: int
    ops: tuple[Op, ...] = ()
    n_params: int = 0

    # ------------------------------------------------------------------ build --
    def __post_init__(self) -> None:
        if self.n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        for op in self.ops:
            for q in op.qubits:
                if not 0 <= q < self.n_qubits:
                    raise ValueError(f"qubit {q} out of range for {self.n_qubits}-qubit circuit")
        needed = 0
        for op in self.ops:
            for p in op.params:
                if isinstance(p, ParamRef):
                    needed = max(needed, p.index + 1)
        if self.n_params < needed:
            raise ValueError(
                f"circuit references parameter index {needed - 1} but n_params={self.n_params}"
            )

    def compose(self, other: CircuitSpec, param_offset: int | None = None) -> CircuitSpec:
        """Concatenate ``other`` after ``self``.

        By default the two parameter vectors are concatenated, so the composed
        circuit has ``self.n_params + other.n_params`` parameters. Pass
        ``param_offset=0`` to *share* the parameter vector instead.
        """
        shift = self.n_params if param_offset is None else param_offset
        ops = list(self.ops)
        for op in other.ops:
            ops.append(
                Op(
                    op.gate,
                    op.qubits,
                    tuple(
                        ParamRef(p.index + shift, p.scale, p.offset)
                        if isinstance(p, ParamRef)
                        else p
                        for p in op.params
                    ),
                )
            )
        return CircuitSpec(
            n_qubits=max(self.n_qubits, other.n_qubits),
            ops=tuple(ops),
            n_params=max(self.n_params, shift + other.n_params),
        )

    def __add__(self, other: CircuitSpec) -> CircuitSpec:
        return self.compose(other)

    def adjoint(self) -> CircuitSpec:
        """Reverse the circuit and invert every gate — the U†(x) of an inversion test."""
        ops: list[Op] = []
        for op in reversed(self.ops):
            g = get_gate(op.gate)
            if g.is_parametric:
                ops.append(
                    Op(
                        op.gate,
                        op.qubits,
                        tuple(
                            ParamRef(p.index, -p.scale, -p.offset)
                            if isinstance(p, ParamRef)
                            else -float(p)
                            for p in op.params
                        ),
                    )
                )
            elif g.adjoint_name is not None:
                ops.append(Op(g.adjoint_name, op.qubits))
            else:
                ops.append(op)  # self-inverse
        return CircuitSpec(self.n_qubits, tuple(ops), self.n_params)

    # ------------------------------------------------------------------ slots --
    def slots(self) -> tuple[Slot, ...]:
        """Every parameterised angle site, in circuit order."""
        out: list[Slot] = []
        for i, op in enumerate(self.ops):
            for pos, p in enumerate(op.params):
                if isinstance(p, ParamRef):
                    out.append(Slot(i, pos, p, op.gate))
        return tuple(out)

    def occurrences_of(self, param_index: int) -> tuple[Slot, ...]:
        """Slots driven by logical parameter ``param_index`` (>1 means weight tying)."""
        return tuple(s for s in self.slots() if s.ref.index == param_index)

    def gates_using(self, param_index: int) -> tuple[str, ...]:
        return tuple(dict.fromkeys(s.gate for s in self.occurrences_of(param_index)))

    def bind_slots(self, theta: ArrayLike) -> npt.NDArray[Any]:
        """Resolve the logical parameter vector into one angle per slot."""
        arr = np.asarray(theta, dtype=float).ravel()
        if arr.size != self.n_params:
            raise ValueError(f"expected {self.n_params} parameters, got {arr.size}")
        return np.array([s.ref.resolve(arr) for s in self.slots()], dtype=float)

    def with_slot_angles(self, angles: ArrayLike) -> CircuitSpec:
        """Return a fully-bound copy in which every slot takes a literal angle."""
        arr = np.asarray(angles, dtype=float).ravel()
        slots = self.slots()
        if arr.size != len(slots):
            raise ValueError(f"expected {len(slots)} slot angles, got {arr.size}")
        by_op: dict[int, dict[int, float]] = {}
        for value, s in zip(arr, slots, strict=False):
            by_op.setdefault(s.op_index, {})[s.param_pos] = float(value)
        ops: list[Op] = []
        for i, op in enumerate(self.ops):
            if i in by_op:
                overrides = by_op[i]
                ops.append(
                    Op(
                        op.gate,
                        op.qubits,
                        tuple(
                            overrides.get(pos, p if not isinstance(p, ParamRef) else 0.0)
                            for pos, p in enumerate(op.params)
                        ),
                    )
                )
            else:
                ops.append(op)
        return CircuitSpec(self.n_qubits, tuple(ops), 0)

    def bind(self, theta: ArrayLike | None = None) -> CircuitSpec:
        """Fully bind the circuit with a logical parameter vector."""
        if self.n_params == 0:
            return self
        if theta is None:
            raise ValueError(f"circuit has {self.n_params} parameters; theta is required")
        return self.with_slot_angles(self.bind_slots(theta))

    @property
    def is_bound(self) -> bool:
        return not self.slots()

    # -------------------------------------------------------------- resources --
    def depth(self) -> int:
        """Circuit depth: the longest chain of gates sharing a qubit."""
        frontier = [0] * self.n_qubits
        for op in self.ops:
            layer = max(frontier[q] for q in op.qubits) + 1
            for q in op.qubits:
                frontier[q] = layer
        return max(frontier, default=0)

    def gate_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.ops:
            counts[op.gate] = counts.get(op.gate, 0) + 1
        return dict(sorted(counts.items()))

    def resources(self) -> dict[str, object]:
        counts = self.gate_counts()
        n_2q = sum(n for g, n in counts.items() if get_gate(g).n_qubits == 2)
        return {
            "n_qubits": self.n_qubits,
            "n_params": self.n_params,
            "n_slots": len(self.slots()),
            "n_ops": len(self.ops),
            "n_1q": len(self.ops) - n_2q,
            "n_2q": n_2q,
            "depth": self.depth(),
            "gate_counts": counts,
        }

    # ---------------------------------------------------------------- niceties --
    def __iter__(self) -> Iterator[Op]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:
        return (
            f"CircuitSpec(n_qubits={self.n_qubits}, n_ops={len(self.ops)}, "
            f"n_params={self.n_params}, depth={self.depth()})"
        )


def concat(specs: Iterable[CircuitSpec]) -> CircuitSpec:
    """Compose a sequence of circuits left to right."""
    it = iter(specs)
    try:
        out = next(it)
    except StopIteration:
        raise ValueError("concat() needs at least one circuit") from None
    for s in it:
        out = out.compose(s)
    return out
