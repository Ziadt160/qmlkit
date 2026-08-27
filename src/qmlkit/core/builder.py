"""A small fluent builder for circuits.

``QCircuit`` is sugar over :class:`~qmlkit.core.ir.CircuitSpec`; anything it can
build can also be assembled by hand from ``Op`` objects. ``param()`` and
``params()`` hand out :class:`ParamRef` values, and ``share`` lets one logical
parameter drive several gates — the weight-tying case the gradient code handles
per occurrence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from qmlkit.core.gates import get_gate
from qmlkit.core.ir import CircuitSpec, Op, ParamLike, ParamRef
from qmlkit.utils.errors import unknown

__all__ = ["QCircuit", "entangler_pairs"]


def entangler_pairs(n_qubits: int, pattern: str = "chain") -> tuple[tuple[int, int], ...]:
    """Qubit pairs for a named entanglement pattern.

    On two qubits a ``"ring"`` would revisit the same pair, so it collapses to a
    single ``(0, 1)``. PennyLane's templates run their loop uniformly and emit both
    ``CNOT(0, 1)`` and ``CNOT(1, 0)`` there, which is a genuinely different circuit —
    worth knowing when porting a two-qubit ansatz between the two libraries.
    """
    if n_qubits < 2:
        return ()
    p = pattern.lower()
    if p in ("chain", "linear"):
        return tuple((i, i + 1) for i in range(n_qubits - 1))
    if p == "ring":
        if n_qubits == 2:
            return ((0, 1),)
        return tuple((i, (i + 1) % n_qubits) for i in range(n_qubits))
    if p in ("full", "all"):
        return tuple((i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits))
    if p == "alternating":
        even = tuple((i, i + 1) for i in range(0, n_qubits - 1, 2))
        odd = tuple((i, i + 1) for i in range(1, n_qubits - 1, 2))
        return even + odd
    raise unknown(
        "entanglement pattern",
        pattern,
        ("chain", "linear", "ring", "full", "all", "alternating"),
    )


class QCircuit:
    """Builds a :class:`CircuitSpec` step by step."""

    def __init__(self, n_qubits: int, n_params: int = 0) -> None:
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        self.n_qubits = n_qubits
        self._ops: list[Op] = []
        self._n_params = n_params

    # ------------------------------------------------------------ parameters --
    def param(self, scale: float = 1.0, offset: float = 0.0) -> ParamRef:
        """Allocate one new logical parameter."""
        ref = ParamRef(self._n_params, scale, offset)
        self._n_params += 1
        return ref

    def params(self, n: int) -> tuple[ParamRef, ...]:
        """Allocate ``n`` new logical parameters."""
        return tuple(self.param() for _ in range(n))

    @property
    def n_params(self) -> int:
        return self._n_params

    # ----------------------------------------------------------------- gates --
    def apply(self, gate: str, qubits: int | Sequence[int], *params: ParamLike) -> QCircuit:
        qs = (qubits,) if isinstance(qubits, int) else tuple(qubits)
        self._ops.append(Op(gate.lower(), qs, tuple(params)))
        for p in params:
            if isinstance(p, ParamRef):
                self._n_params = max(self._n_params, p.index + 1)
        return self

    # one-qubit, no parameters
    def h(self, q: int) -> QCircuit:
        return self.apply("h", q)

    def x(self, q: int) -> QCircuit:
        return self.apply("x", q)

    def y(self, q: int) -> QCircuit:
        return self.apply("y", q)

    def z(self, q: int) -> QCircuit:
        return self.apply("z", q)

    def s(self, q: int) -> QCircuit:
        return self.apply("s", q)

    def sdg(self, q: int) -> QCircuit:
        return self.apply("sdg", q)

    def t(self, q: int) -> QCircuit:
        return self.apply("t", q)

    def tdg(self, q: int) -> QCircuit:
        return self.apply("tdg", q)

    # one-qubit rotations
    def rx(self, q: int, theta: ParamLike) -> QCircuit:
        return self.apply("rx", q, theta)

    def ry(self, q: int, theta: ParamLike) -> QCircuit:
        return self.apply("ry", q, theta)

    def rz(self, q: int, theta: ParamLike) -> QCircuit:
        return self.apply("rz", q, theta)

    def phase(self, q: int, theta: ParamLike) -> QCircuit:
        return self.apply("phase", q, theta)

    # two-qubit
    def cx(self, control: int, target: int) -> QCircuit:
        return self.apply("cx", (control, target))

    def cy(self, control: int, target: int) -> QCircuit:
        return self.apply("cy", (control, target))

    def cz(self, control: int, target: int) -> QCircuit:
        return self.apply("cz", (control, target))

    def swap(self, a: int, b: int) -> QCircuit:
        return self.apply("swap", (a, b))

    def crx(self, control: int, target: int, theta: ParamLike) -> QCircuit:
        return self.apply("crx", (control, target), theta)

    def cry(self, control: int, target: int, theta: ParamLike) -> QCircuit:
        return self.apply("cry", (control, target), theta)

    def crz(self, control: int, target: int, theta: ParamLike) -> QCircuit:
        return self.apply("crz", (control, target), theta)

    # ----------------------------------------------------------------- layers --
    def rotation_layer(
        self,
        gates: Sequence[str] = ("ry",),
        wires: Iterable[int] | None = None,
        shared: ParamRef | None = None,
    ) -> QCircuit:
        """One rotation per gate per wire.

        Pass ``shared`` to tie every rotation in the layer to one logical
        parameter — the weight-tying case worth testing gradients against.
        """
        qs = list(range(self.n_qubits)) if wires is None else list(wires)
        for q in qs:
            for g in gates:
                self.apply(g.lower(), q, shared if shared is not None else self.param())
        return self

    def entangle(self, pattern: str = "chain", gate: str = "cx") -> QCircuit:
        """A layer of two-qubit gates following a named pattern."""
        if get_gate(gate).n_params:
            raise ValueError(
                f"{gate!r} is parameterised; use parametric_entangle() to allocate its angles"
            )
        for a, b in entangler_pairs(self.n_qubits, pattern):
            self.apply(gate, (a, b))
        return self

    def parametric_entangle(self, gate: str = "crz", pattern: str = "ring") -> QCircuit:
        """A layer of *trainable* two-qubit gates — exercises the four-term rule."""
        for a, b in entangler_pairs(self.n_qubits, pattern):
            self.apply(gate, (a, b), self.param())
        return self

    # ------------------------------------------------------------------ build --
    def to_spec(self) -> CircuitSpec:
        return CircuitSpec(self.n_qubits, tuple(self._ops), self._n_params)

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        return (
            f"QCircuit(n_qubits={self.n_qubits}, n_ops={len(self._ops)}, n_params={self._n_params})"
        )
