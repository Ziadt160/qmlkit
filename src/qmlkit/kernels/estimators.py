r"""The three ways to read a kernel off a circuit.

A quantum kernel is an overlap: :math:`k(x, x') = |\langle \phi(x')|\phi(x)\rangle|^2`.
Three estimators get at it, and they are not interchangeable:

============  ==========  ====================  =========================================
Estimator     Qubits      Depth                 Gives you
============  ==========  ====================  =========================================
inversion     ``n``       ``2 x`` feature map   ``|<.>|^2`` — the magnitude
swap test     ``2n + 1``  1 map + n CSWAPs      ``|<.>|^2``, from an ancilla
Hadamard      ``n + 1``   controlled map        ``Re<.>`` — **signed**
============  ==========  ====================  =========================================

The inversion (compute-uncompute) test is the default: fewest qubits, no ancilla,
no controlled gates. The swap test earns its extra register when you already hold
two states and cannot rebuild one. The Hadamard test is the only one that keeps the
*sign* of the inner product — magnitude estimators map ``+1/2`` and ``-1/2`` to the
same number.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from qmlkit.core.builder import QCircuit
from qmlkit.core.execute import BackendLike, probabilities, run_counts
from qmlkit.core.ir import CircuitSpec, Op
from qmlkit.encoding.feature_maps import FeatureMap

__all__ = [
    "fidelity_kernel",
    "swap_test_kernel",
    "hadamard_test",
    "swap_readout",
    "swap_probability",
    "inversion_circuit",
]


def inversion_circuit(fmap: FeatureMap, x: Sequence[float], xp: Sequence[float]) -> CircuitSpec:
    """``U(x)`` followed by ``U(x')†``. P(all zeros) *is* the kernel."""
    return fmap.build(x).compose(fmap.adjoint(xp), param_offset=0)


def fidelity_kernel(
    fmap: FeatureMap,
    x: Sequence[float],
    xp: Sequence[float],
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> float:
    """``k(x, x')`` by the compute-uncompute test — the default estimator.

    With ``shots=None`` this reads the exact all-zeros probability. With a shot
    budget it counts the all-zeros outcomes, which is what a device would do.
    """
    spec = inversion_circuit(fmap, x, xp)
    if shots is None:
        return float(probabilities(spec, backend=backend)[0])
    counts = run_counts(spec, shots=shots, backend=backend, seed=seed)
    zeros = "0" * spec.n_qubits
    return counts.get(zeros, 0) / shots


def swap_readout(p_ancilla_zero: float) -> float:
    """Invert the swap-test readout: ``k = 2 P(anc=0) - 1``."""
    return 2.0 * float(p_ancilla_zero) - 1.0


def swap_probability(k: float) -> float:
    """Forward direction: ``P(anc=0) = (1 + k) / 2``. Orthogonal states give a fair coin."""
    return (1.0 + float(k)) / 2.0


def swap_test_kernel(
    fmap: FeatureMap,
    x: Sequence[float],
    xp: Sequence[float],
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> float:
    """``k(x, x')`` by the swap test — two registers plus one ancilla.

    Costs ``2n + 1`` qubits against the inversion test's ``n``, and needs a CSWAP
    per qubit pair. Worth it only when you genuinely hold two states already.
    """
    a, b = fmap.build(x), fmap.build(xp)
    n = fmap.n_qubits
    ancilla = 2 * n

    ops: list[Op] = []
    ops += [Op(op.gate, tuple(q for q in op.qubits), op.params) for op in a.ops]
    ops += [Op(op.gate, tuple(q + n for q in op.qubits), op.params) for op in b.ops]
    ops.append(Op("h", (ancilla,)))
    for i in range(n):
        ops += _controlled_swap(ancilla, i, i + n)
    ops.append(Op("h", (ancilla,)))

    spec = CircuitSpec(2 * n + 1, tuple(ops), 0)
    if shots is None:
        probs = probabilities(spec, backend=backend)
        # ancilla is the least significant bit (qubit 0 is most significant)
        p0 = float(probs[::2].sum())
    else:
        counts = run_counts(spec, shots=shots, backend=backend, seed=seed)
        p0 = sum(v for k, v in counts.items() if k[ancilla] == "0") / shots
    return swap_readout(p0)


def _controlled_swap(control: int, a: int, b: int) -> list[Op]:
    """CSWAP from CX and a Toffoli built out of the registered gate set."""
    return [Op("cx", (b, a)), *_toffoli(control, a, b), Op("cx", (b, a))]


def _toffoli(c1: int, c2: int, target: int) -> list[Op]:
    """Standard H/T/CX decomposition — keeps CSWAP inside the portable gate set."""
    return [
        Op("h", (target,)),
        Op("cx", (c2, target)),
        Op("tdg", (target,)),
        Op("cx", (c1, target)),
        Op("t", (target,)),
        Op("cx", (c2, target)),
        Op("tdg", (target,)),
        Op("cx", (c1, target)),
        Op("t", (c2,)),
        Op("t", (target,)),
        Op("h", (target,)),
        Op("cx", (c1, c2)),
        Op("t", (c1,)),
        Op("tdg", (c2,)),
        Op("cx", (c1, c2)),
    ]


def hadamard_test(
    fmap: FeatureMap,
    x: Sequence[float],
    xp: Sequence[float],
    part: str = "real",
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> float:
    """``Re<phi(x')|phi(x)>`` (or the imaginary part) — the **signed** inner product.

    The only estimator that distinguishes ``+1/2`` from ``-1/2``; the magnitude ones
    map both to ``1/4``. The price is an ancilla and a *controlled* feature map,
    which roughly doubles depth and wants all-to-all connectivity to the ancilla —
    which is why it is rarely the right choice on hardware.
    """
    if part not in ("real", "imag"):
        raise ValueError(f"part must be 'real' or 'imag', got {part!r}")
    n = fmap.n_qubits
    ancilla = n

    qc = QCircuit(n + 1)
    qc.h(ancilla)
    if part == "imag":
        qc.sdg(ancilla)
    spec = qc.to_spec()

    body = fmap.build(x).compose(fmap.adjoint(xp), param_offset=0)
    ops = list(spec.ops)
    for op in body.ops:
        ops.extend(_controlled(op, ancilla))
    ops.append(Op("h", (ancilla,)))

    full = CircuitSpec(n + 1, tuple(ops), 0)
    if shots is None:
        probs = probabilities(full, backend=backend)
        p0 = float(probs[::2].sum())
    else:
        counts = run_counts(full, shots=shots, backend=backend, seed=seed)
        p0 = sum(v for k, v in counts.items() if k[ancilla] == "0") / shots
    return 2.0 * p0 - 1.0


def _controlled(op: Op, control: int) -> list[Op]:
    """Lift one operation to its controlled version."""
    simple = {"x": "cx", "y": "cy", "z": "cz", "rx": "crx", "ry": "cry", "rz": "crz"}
    if op.gate in simple and len(op.qubits) == 1:
        return [Op(simple[op.gate], (control, op.qubits[0]), op.params)]
    if op.gate == "h":
        # H = Ry(pi/4) X Ry(-pi/4) up to phase; controlled-H from CY-free primitives
        t = op.qubits[0]
        return [
            Op("ry", (t,), (np.pi / 4,)),
            Op("cx", (control, t)),
            Op("ry", (t,), (-np.pi / 4,)),
        ]
    if op.gate == "cx":
        return _toffoli(control, op.qubits[0], op.qubits[1])
    if op.gate in ("s", "sdg", "t", "tdg", "phase"):
        angle = {"s": np.pi / 2, "sdg": -np.pi / 2, "t": np.pi / 4, "tdg": -np.pi / 4}
        theta = angle[op.gate] if op.gate != "phase" else float(op.params[0])
        return [Op("crz", (control, op.qubits[0]), (theta,)), Op("phase", (control,), (theta / 2,))]
    raise NotImplementedError(
        f"the Hadamard test has no controlled form for {op.gate!r}; use the inversion "
        "or swap estimator, which need no controlled gates"
    )
