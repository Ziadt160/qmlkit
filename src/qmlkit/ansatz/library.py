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

import numpy as np
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
    ) -> np.ndarray:
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
        raise ValueError(f"unknown init method {method!r}; expected small, uniform or zeros")

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
        raise KeyError(
            f"unknown ansatz {name!r}; registered: {', '.join(list_ansatze())}"
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


def tree_tensor_network(n_qubits: int) -> Ansatz:
    """Log-depth merge tree — shallow, and resistant to barren plateaus."""

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        nodes = list(ctx.active)
        while len(nodes) > 1:
            nxt = []
            for i in range(0, len(nodes) - 1, 2):
                a, b = nodes[i], nodes[i + 1]
                qc.ry(a, ctx.new_param())
                qc.ry(b, ctx.new_param())
                qc.cx(a, b)
                nxt.append(b)
            if len(nodes) % 2:
                nxt.append(nodes[-1])
            nodes = nxt
        ctx.active = nodes

    return Ansatz(n_qubits, Custom(build, "ttn"), "tree_tensor_network")


def mps_ansatz(n_qubits: int) -> Ansatz:
    """A staircase of two-qubit blocks — a bond-dimension-2 matrix product state."""

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        for i in range(len(wires) - 1):
            qc.ry(wires[i], ctx.new_param())
            qc.ry(wires[i + 1], ctx.new_param())
            qc.cx(wires[i], wires[i + 1])

    return Ansatz(n_qubits, Custom(build, "mps"), "mps")


def conv_block(pattern: str = "chain", tied: bool = True) -> Block:
    """A QCNN convolution layer.

    ``tied=True`` allocates **one** two-qubit filter and slides it across every
    pair — the genuine convolutional structure, and the reason the gradient code
    sums over occurrences. ``tied=False`` gives each pair its own weights.
    """

    def build(qc: QCircuit, ctx: BuildContext) -> None:
        wires = ctx.active
        pairs = entangler_pairs(len(wires), pattern)
        if not pairs:
            return
        shared = (ctx.new_param(), ctx.new_param()) if tied else None
        for a, b in pairs:
            ta, tb = shared if shared is not None else (ctx.new_param(), ctx.new_param())
            qc.ry(wires[a], ta)
            qc.ry(wires[b], tb)
            qc.cx(wires[a], wires[b])

    return Custom(build, f"conv{'_tied' if tied else ''}")


def qcnn_ansatz(n_qubits: int, tie_weights: bool = True) -> Ansatz:
    """Convolution + pooling, halving the register until one qubit is left.

    ``tie_weights=True`` shares one filter across all applications in a layer — the
    genuine convolutional structure, and the case whose gradient needs a sum over
    occurrences.
    """
    import math

    n_layers = max(1, int(math.ceil(math.log2(n_qubits))))
    layer = conv_block(tied=tie_weights) + PoolLayer("odd")
    return Ansatz(n_qubits, repeat(n_layers, layer), "qcnn")


def qaoa_ansatz(
    n_qubits: int, edges: Sequence[tuple[int, int]] | None = None, p: int = 1
) -> Ansatz:
    """Cost and mixer layers — only ``2p`` parameters, whatever the width.

    Both angles in a round are shared across all their gates, which is what keeps
    the parameter count at ``2p`` rather than growing with the graph.
    """
    graph = list(edges) if edges is not None else list(entangler_pairs(n_qubits, "chain"))

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
            for q in ctx.active:
                qc.apply("rx", q, beta)

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
