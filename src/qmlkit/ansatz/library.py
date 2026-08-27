"""The `Ansatz` type, the built-in zoo, and the registry.

An ansatz is a width plus a block. That is the whole type — which is what makes
proposing a new one a one-liner rather than a subclass:

    brick = Ansatz(6, repeat(3, RotationLayer("ry") + EntanglerLayer("cz", "alternating")))

Parameter counts are **inferred** from a dry build, never hand-counted, so a
miscount is not a failure mode. Everything downstream — gradients, resources, the
torch layer — reads the resulting IR, so a new ansatz cannot be missing a
capability it never had to opt into.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.ansatz.blocks import (
    Block,
    BuildContext,
    Custom,
    EntanglerLayer,
    ParametricEntangler,
    PoolLayer,
    RotationLayer,
    repeat,
    share,
)
from qmlkit.core.builder import QCircuit, entangler_pairs
from qmlkit.core.ir import CircuitSpec
from qmlkit.utils.errors import unknown

__all__ = [
    "Ansatz",
    "register_ansatz",
    "get_ansatz",
    "list_ansatze",
    "hardware_efficient",
    "strongly_entangling",
    "simplified_two_design",
    "tree_tensor_network",
    "mps_ansatz",
    "qcnn_ansatz",
    "qaoa_ansatz",
    "conv_block",
    "register_conv_filter",
    "list_conv_filters",
    "basic_entangler",
    "two_local",
    "random_layers",
]


class Ansatz:
    """A trainable circuit: a qubit count and a composable block."""

    def __init__(
        self, n_qubits: int, block: Block, name: str = "ansatz", n_inputs: int = 0
    ) -> None:
        if n_qubits < 1:
            raise ValueError("n_qubits must be at least 1")
        self.n_qubits = n_qubits
        self.block = block
        self.name = name
        self.n_inputs = n_inputs
        self._spec: CircuitSpec | None = None

    # ------------------------------------------------------------------ build --
    def _template(self) -> CircuitSpec:
        """The unbound circuit, built once and cached."""
        if self._spec is None:
            qc = QCircuit(self.n_qubits)
            ctx = BuildContext(self.n_qubits, self.n_inputs)
            self.block.emit(qc, ctx)
            self._spec = qc.to_spec()
        return self._spec

    def build(self, theta: ArrayLike | None = None) -> CircuitSpec:
        """The circuit, bound to ``theta`` if given.

        ``theta`` is the **full** parameter vector: any reserved input slots first,
        then the weights. When ``n_inputs`` is 0 that is just the weights, but a
        re-uploading model reserves input slots — use :meth:`bind` there, which takes
        data and weights separately.
        """
        spec = self._template()
        if theta is None:
            return spec
        arr = np.asarray(theta, dtype=float).ravel()
        if self.n_inputs and arr.size == self.n_weights:
            raise ValueError(
                f"{type(self).__name__} reserves {self.n_inputs} input slots, so build() "
                f"expects {self.n_params} values (inputs first, then {self.n_weights} "
                "weights). Use .bind(x, weights) to pass data and weights separately."
            )
        return spec.bind(arr)

    def bind(self, x: ArrayLike, weights: ArrayLike) -> CircuitSpec:
        """Bind data and weights separately, in that order."""
        angles = self.angles(x) if hasattr(self, "angles") else np.asarray(x, dtype=float)
        return self.build(np.concatenate([np.ravel(angles), np.ravel(weights)]))

    def __call__(self, theta: ArrayLike | None = None) -> CircuitSpec:
        return self.build(theta)

    # ------------------------------------------------------------------ shape --
    @property
    def n_params(self) -> int:
        """Every parameter, inputs included. Inferred — never hand-counted."""
        return self._template().n_params

    @property
    def n_weights(self) -> int:
        """Trainable parameters only, excluding reserved input slots."""
        return self.n_params - self.n_inputs

    @property
    def param_shape(self) -> tuple[int, ...]:
        return (self.n_weights,)

    def init(
        self, method: str = "small", seed: int | None = None, scale: float = 0.1
    ) -> npt.NDArray[Any]:
        """Initial parameters.

        ``small`` (default) keeps angles near zero, which keeps the circuit near
        identity and the gradient away from the barren-plateau regime. ``uniform``
        samples the full range — the standard way to *land* on a plateau, useful
        when that is what you are studying. ``zeros`` is exactly identity.
        """
        rng = np.random.default_rng(seed)
        n = self.n_weights
        if method == "small":
            return rng.normal(0.0, scale, n)
        if method == "uniform":
            return rng.uniform(-np.pi, np.pi, n)
        if method == "zeros":
            return np.zeros(n)
        raise unknown("init method", method, ("small", "uniform", "zeros"))

    # -------------------------------------------------------------- resources --
    def resources(self) -> dict[str, object]:
        """Gate counts, depth, and the *real* gradient cost."""
        from qmlkit.gradients.parameter_shift import grad_circuit_cost

        spec = self._template()
        out = dict(spec.resources())
        out["grad_circuits"] = grad_circuit_cost(spec)
        return out

    def __repr__(self) -> str:
        return f"Ansatz({self.name!r}, n_qubits={self.n_qubits}, n_params={self.n_params})"


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
AnsatzFactory = Callable[..., Ansatz]
_REGISTRY: dict[str, AnsatzFactory] = {}


def register_ansatz(
    name: str, factory: AnsatzFactory | None = None
) -> Callable[[AnsatzFactory], AnsatzFactory] | AnsatzFactory:
    """Register an ansatz factory. Usable as a decorator or a direct call.

    def brick_wall(n_qubits, n_layers=3):
    return Ansatz(n_qubits, repeat(n_layers, RotationLayer("ry")
                                             + EntanglerLayer("cz", "alternating")))
    """

    def _register(f: AnsatzFactory) -> AnsatzFactory:
        if name in _REGISTRY:
            raise ValueError(f"ansatz {name!r} is already registered")
        _REGISTRY[name] = f
        return f

    return _register if factory is None else _register(factory)


def get_ansatz(name: str, **kwargs: object) -> Ansatz:
    """Build a registered ansatz by name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        raise unknown(
            "ansatz",
            name,
            list_ansatze(),
            hint="Add your own with register_ansatz(name, factory).",
            error=KeyError,
        ) from None
    return factory(**kwargs)


def list_ansatze() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# --------------------------------------------------------------------------- #
# the built-in zoo — every one is a short expression in the vocabulary
# --------------------------------------------------------------------------- #
def hardware_efficient(
    n_qubits: int,
    n_layers: int = 2,
    rotations: Sequence[str] = ("ry", "rz"),
    entangler: str = "cx",
    pattern: str = "chain",
) -> Ansatz:
    """Rotations then entanglers, repeated. General-purpose, barren-plateau prone."""
    return Ansatz(
        n_qubits,
        repeat(n_layers, RotationLayer(rotations) + EntanglerLayer(entangler, pattern)),
        "hardware_efficient",
    )


def strongly_entangling(n_qubits: int, n_layers: int = 2) -> Ansatz:
    """Three rotations per wire, plus a ring of CX per layer."""
    return Ansatz(
        n_qubits,
        repeat(n_layers, RotationLayer(("rz", "ry", "rz")) + EntanglerLayer("cx", "ring")),
        "strongly_entangling",
    )


def simplified_two_design(n_qubits: int, n_layers: int = 2) -> Ansatz:
    """The standard reference ansatz in barren-plateau studies."""
    return Ansatz(
        n_qubits,
        RotationLayer("ry")
        + repeat(n_layers, EntanglerLayer("cz", "alternating") + RotationLayer("ry")),
        "simplified_two_design",
    )


def tree_tensor_network(
    n_qubits: int,
    filter: str | tuple[ConvFilter, int] = "ry_cx",  # noqa: A002 - the domain word
    tied: bool = False,
) -> Ansatz:
    """Log-depth merge tree — shallow, and resistant to barren plateaus.

    Each merge is a two-qubit ``filter`` from the same registry a QCNN convolves
    with, so the tensor at every node is as general as you choose to pay for.
    """
    fn, n_params = _resolve_filter(filter)

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        nodes = list(ctx.active)
        shared = tuple(ctx.new_param() for _ in range(n_params)) if tied else None
        while len(nodes) > 1:
            nxt = []
            for i in range(0, len(nodes) - 1, 2):
                a, b = nodes[i], nodes[i + 1]
                params = shared or tuple(ctx.new_param() for _ in range(n_params))
                fn(qc, a, b, params)
                nxt.append(b)
            if len(nodes) % 2:
                nxt.append(nodes[-1])
            nodes = nxt
        ctx.active = nodes

    return Ansatz(n_qubits, Custom(build, "ttn"), "tree_tensor_network")


def mps_ansatz(
    n_qubits: int,
    filter: str | tuple[ConvFilter, int] = "ry_cx",  # noqa: A002 - the domain word
    tied: bool = False,
) -> Ansatz:
    """A staircase of two-qubit blocks — a bond-dimension-2 matrix product state.

    The block is the same two-qubit *filter* a QCNN convolves with, so it comes from
    the same registry: ``"su4"`` gives a genuine bond-dimension-2 MPS with arbitrary
    tensors, ``"ry_cx"`` the cheap real-valued one. ``tied=True`` reuses one tensor
    down the whole chain, which is the translation-invariant MPS.
    """
    fn, n_params = _resolve_filter(filter)

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        shared = tuple(ctx.new_param() for _ in range(n_params)) if tied else None
        for i in range(len(wires) - 1):
            params = shared or tuple(ctx.new_param() for _ in range(n_params))
            fn(qc, wires[i], wires[i + 1], params)

    return Ansatz(n_qubits, Custom(build, "mps"), "mps")


# --------------------------------------------------------------------------- #
# QCNN convolution filters
#
# A QCNN is a *pattern* — convolve, pool, repeat — not one circuit. The literature
# differs mainly in which two-qubit filter slides across the register: Cong, Choi &
# Lukin (2019) use a general SU(4); Hur, Kim & Park (2022) benchmark eight cheaper
# ones. So the filter is a value you pass in, not a subclass you pick, and adding
# your own is a function rather than a fork.
#
# A filter is ``fn(qc, a, b, params)`` plus the number of parameters it consumes.
# --------------------------------------------------------------------------- #
ConvFilter = Callable[[QCircuit, int, int, "Sequence[Any]"], None]

_FILTERS: dict[str, tuple[ConvFilter, int]] = {}


def register_conv_filter(name: str, fn: ConvFilter, n_params: int) -> None:
    """Make a two-qubit filter reachable by name from :func:`conv_block`."""
    if name in _FILTERS:
        raise ValueError(f"conv filter {name!r} is already registered")
    _FILTERS[name] = (fn, n_params)


def list_conv_filters() -> tuple[str, ...]:
    return tuple(sorted(_FILTERS))


def _f_ry_cx(qc: QCircuit, a: int, b: int, p: Sequence[Any]) -> None:
    """The cheapest useful filter: one rotation each, one entangler."""
    qc.ry(a, p[0])
    qc.ry(b, p[1])
    qc.cx(a, b)


def _f_real(qc: QCircuit, a: int, b: int, p: Sequence[Any]) -> None:
    """Real-amplitude block — rotations either side of the entangler."""
    qc.ry(a, p[0])
    qc.ry(b, p[1])
    qc.cx(a, b)
    qc.ry(a, p[2])
    qc.ry(b, p[3])


def _f_zz(qc: QCircuit, a: int, b: int, p: Sequence[Any]) -> None:
    """Single-qubit Ry rotations plus a genuine ZZ interaction.

    The single-qubit part is deliberately ``ry`` rather than ``rz``: an all-``rz``
    filter is diagonal, so from ``|0...0>`` it does *nothing at all* — measured, not
    guessed, and the reason ``test_no_shipped_filter_is_inert`` exists.
    """
    qc.ry(a, p[0])
    qc.ry(b, p[1])
    qc.cx(a, b)
    qc.rz(b, p[2])
    qc.cx(a, b)


def _u3(qc: QCircuit, wire: int, p: Sequence[Any]) -> None:
    """An arbitrary single-qubit unitary, as Rz-Ry-Rz."""
    qc.rz(wire, p[0])
    qc.ry(wire, p[1])
    qc.rz(wire, p[2])


def _f_su4(qc: QCircuit, a: int, b: int, p: Sequence[Any]) -> None:
    """A general two-qubit unitary — the Vatan-Williams form, 3 CNOTs, 15 angles.

    This is what Cong, Choi & Lukin's QCNN uses. It can express *any* two-qubit
    gate, which is the most expressive filter possible and also the most expensive.
    """
    _u3(qc, a, p[0:3])
    _u3(qc, b, p[3:6])
    qc.cx(b, a)
    qc.rz(a, p[6])
    qc.ry(b, p[7])
    qc.cx(a, b)
    qc.ry(b, p[8])
    qc.cx(b, a)
    _u3(qc, a, p[9:12])
    _u3(qc, b, p[12:15])


register_conv_filter("ry_cx", _f_ry_cx, 2)
register_conv_filter("real", _f_real, 4)
register_conv_filter("zz", _f_zz, 3)
register_conv_filter("su4", _f_su4, 15)


def _resolve_filter(spec: str | tuple[ConvFilter, int]) -> tuple[ConvFilter, int]:
    if isinstance(spec, str):
        try:
            return _FILTERS[spec]
        except KeyError:
            raise unknown(
                "conv filter",
                spec,
                list_conv_filters(),
                hint="Add your own with register_conv_filter(name, factory).",
                error=KeyError,
            ) from None
    fn, n_params = spec
    return fn, int(n_params)


def conv_block(
    pattern: str = "chain",
    tied: bool = True,
    filter: str | tuple[ConvFilter, int] = "ry_cx",  # noqa: A002 - the domain word
) -> Block:
    """A QCNN convolution layer: slide one two-qubit ``filter`` across ``pattern``.

    ``tied=True`` allocates **one** filter and reuses it at every pair — the genuine
    convolutional structure, and the reason the gradient code sums over occurrences.
    ``tied=False`` gives each pair its own weights.

    ``filter`` is a registered name (:func:`list_conv_filters`) or a
    ``(fn, n_params)`` pair, where ``fn(qc, a, b, params)`` writes the filter. That
    is the whole extension point: a filter from a paper we have never heard of is a
    function you pass in, not a class you subclass.
    """
    fn, n_params = _resolve_filter(filter)
    label = filter if isinstance(filter, str) else getattr(filter[0], "__name__", "custom")

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        pairs = entangler_pairs(len(wires), pattern)
        if not pairs:
            return
        shared = tuple(ctx.new_param() for _ in range(n_params)) if tied else None
        for a, b in pairs:
            params = (
                shared if shared is not None else tuple(ctx.new_param() for _ in range(n_params))
            )
            fn(qc, wires[a], wires[b], params)

    return Custom(build, f"conv_{label}{'_tied' if tied else ''}")


def qcnn_ansatz(
    n_qubits: int,
    tie_weights: bool = True,
    filter: str | tuple[ConvFilter, int] = "ry_cx",  # noqa: A002 - the domain word
    pattern: str = "chain",
    pool: str = "discard",
    keep: str = "odd",
) -> Ansatz:
    """Convolution + pooling, halving the register until one qubit is left.

    There is no single "the QCNN": papers differ in the two-qubit filter and in how
    pooling discards a wire. Rather than shipping one class per paper, this is the
    shared skeleton with both choices exposed — so reproducing a particular variant
    is a keyword, and inventing one is a function.

    ``tie_weights=True`` shares one filter across all applications in a layer — the
    genuine convolutional structure, and the case whose gradient needs a sum over
    occurrences.
    """
    import math

    n_layers = max(1, int(math.ceil(math.log2(n_qubits))))
    layer = conv_block(pattern=pattern, tied=tie_weights, filter=filter) + PoolLayer(
        keep, mode=pool, tied=tie_weights
    )
    return Ansatz(n_qubits, repeat(n_layers, layer), "qcnn")


def qaoa_ansatz(
    n_qubits: int,
    edges: Sequence[tuple[int, int]] | None = None,
    p: int = 1,
    mixer: str = "x",
) -> Ansatz:
    """Cost and mixer layers — only ``2p`` parameters, whatever the width.

    Both angles in a round are shared across all their gates, which is what keeps
    the parameter count at ``2p`` rather than growing with the graph.
    """
    graph = list(edges) if edges is not None else list(entangler_pairs(n_qubits, "chain"))
    if mixer not in ("x", "y", "xy"):
        raise unknown("mixer", mixer, ("x", "y", "xy"))

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        for q in ctx.active:
            qc.h(q)
        for _ in range(p):
            gamma = ctx.new_param()  # one cost angle for the whole round
            for a, b in graph:
                qc.cx(a, b)
                qc.apply("rz", b, gamma)
                qc.cx(a, b)
            beta = ctx.new_param()  # one mixer angle for the whole round
            if mixer in ("x", "xy"):
                for q in ctx.active:
                    qc.apply("rx", q, beta)
            if mixer in ("y", "xy"):
                for q in ctx.active:
                    qc.apply("ry", q, beta)

    return Ansatz(n_qubits, Custom(build, "qaoa"), "qaoa")


# Registered after definition rather than by decorator: the decorator returns a
# union that mypy cannot see through, and keeping these functions plainly typed is
# worth more than the sugar.
_BUILTINS: list[tuple[str, AnsatzFactory]] = [
    ("hardware_efficient", hardware_efficient),
    ("strongly_entangling", strongly_entangling),
    ("simplified_two_design", simplified_two_design),
    ("tree_tensor_network", tree_tensor_network),
    ("mps", mps_ansatz),
    ("qcnn", qcnn_ansatz),
    ("qaoa", qaoa_ansatz),
]
for _name, _factory in _BUILTINS:
    register_ansatz(_name, _factory)


def basic_entangler(n_qubits: int, n_layers: int = 2, rotation: str = "rx") -> Ansatz:
    """One rotation per wire plus a ring of CNOTs — the minimal useful template."""
    return Ansatz(
        n_qubits,
        repeat(n_layers, RotationLayer((rotation,)) + EntanglerLayer("cx", "ring")),
        "basic_entangler",
    )


def two_local(
    n_qubits: int,
    n_layers: int = 2,
    rotations: Sequence[str] = ("ry",),
    entangler: str = "cx",
    pattern: str = "full",
) -> Ansatz:
    """A configurable rotation/entangler alternation, ending on a rotation layer."""
    return Ansatz(
        n_qubits,
        repeat(n_layers, RotationLayer(rotations) + EntanglerLayer(entangler, pattern))
        + RotationLayer(rotations),
        "two_local",
    )


def random_layers(
    n_qubits: int, n_layers: int = 2, ratio_imprimitive: float = 0.3, seed: int | None = None
) -> Ansatz:
    """Randomly placed rotations and CNOTs — the baseline a new ansatz must beat."""
    rng = np.random.default_rng(seed)
    plan: list[tuple[str, tuple[int, ...]]] = []
    for _ in range(n_layers):
        for q in range(n_qubits):
            plan.append((str(rng.choice(["rx", "ry", "rz"])), (q,)))
            if n_qubits > 1 and rng.random() < ratio_imprimitive:
                other = int(rng.choice([w for w in range(n_qubits) if w != q]))
                plan.append(("cx", (q, other)))

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        for gate, wires in plan:
            if gate == "cx":
                qc.cx(*wires)
            else:
                qc.apply(gate, wires[0], ctx.new_param())

    return Ansatz(n_qubits, Custom(build, "random"), "random_layers")


_LATE: list[tuple[str, AnsatzFactory]] = [
    ("basic_entangler", basic_entangler),
    ("two_local", two_local),
    ("random_layers", random_layers),
]
for _name, _factory in _LATE:
    register_ansatz(_name, _factory)

# re-exported so `from qmlkit.ansatz import ...` reaches the vocabulary too
_VOCAB = (RotationLayer, EntanglerLayer, ParametricEntangler, PoolLayer, Custom, repeat, share)
