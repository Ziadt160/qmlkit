"""Fusion loses below 12 qubits because BUILDING the block matrix costs more than
the gates it replaces -- 84% of the fused time at 6 qubits is the build.

So the question becomes: what if the build is cached? Two things are measured here.

1. The ceiling. Build every block once, then time only the applies. No caching
   scheme can beat this, so if the ceiling is not worth having, fusion is finished
   at these widths and the honest thing is to say so.
2. The achievable part. In a training loop the angles change every step, so only
   blocks whose gates carry NO parameters can be cached across calls. How much of a
   real ansatz is that, and what does caching just those buy?
"""

import sys
import time

sys.path.insert(
    0,
    r"C:\Quantum Machine Learning Module\.claude\worktrees"
    r"\qmlkit-pennylane-marketing-83f602\qmlkit\src",
)

import numpy as np

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import _apply
from qmlkit.core.gates import gate_matrix

from proto_fusion import best, block_matrix, plan, run_unfused


def run_prebuilt(spec, prebuilt):
    """The ceiling: every block matrix already in hand."""
    state = np.zeros((2,) * spec.n_qubits, dtype=complex)
    state[(0,) * spec.n_qubits] = 1.0
    for wires, matrix in prebuilt:
        state = _apply(state, matrix, wires)
    return state.reshape(-1)


def run_partial(spec, blocks, cache):
    """Constant blocks come from the cache; parameterised ones are rebuilt."""
    state = np.zeros((2,) * spec.n_qubits, dtype=complex)
    state[(0,) * spec.n_qubits] = 1.0
    for i, (wires, ops) in enumerate(blocks):
        if i in cache:
            state = _apply(state, cache[i], wires)
        elif len(ops) == 1:
            op = ops[0]
            state = _apply(
                state, np.asarray(gate_matrix(op.gate, op.params), dtype=complex), op.qubits
            )
        else:
            state = _apply(state, block_matrix(wires, ops), wires)
    return state.reshape(-1)


print("1. THE CEILING -- every block matrix prebuilt, only applies timed\n")
print(f"{'qubits':>7}{'gates':>7}" + "".join(f"{f'k<={k}':>12}" for k in (2, 3, 4, 5)))
print("-" * 56)
for n in (4, 6, 8, 10, 12, 14):
    ansatz = qk.hardware_efficient(n, 3)
    spec = ansatz.build().bind(ansatz.init(seed=0))
    base = best(lambda: run_unfused(spec))
    reference = run_unfused(spec)
    row = f"{n:>7}{len(spec.ops):>7}"
    for k in (2, 3, 4, 5):
        blocks = plan(spec.ops, k)
        prebuilt = [(w, block_matrix(w, o)) for w, o in blocks]
        assert np.max(np.abs(run_prebuilt(spec, prebuilt) - reference)) < 1e-12
        t = best(lambda p=prebuilt: run_prebuilt(spec, p))
        row += f"{base / t:>11.2f}x"
    print(row)

print("\n2. HOW MUCH OF A REAL ANSATZ IS CONSTANT (cacheable across steps)\n")
print(f"{'ansatz':<28}{'gates':>7}{'constant':>10}{'share':>8}")
print("-" * 53)
for label, factory in [
    ("hardware_efficient(6,3)", lambda: qk.hardware_efficient(6, 3)),
    ("hardware_efficient(10,3)", lambda: qk.hardware_efficient(10, 3)),
    ("basic_entangler(6,3)", lambda: qk.get_ansatz("basic_entangler")(6, 3)),
]:
    try:
        ansatz = factory()
    except Exception as exc:  # pragma: no cover - ansatz name differences
        print(f"{label:<28}  skipped ({type(exc).__name__})")
        continue
    spec = ansatz.build().bind(ansatz.init(seed=0))
    const = sum(1 for op in spec.ops if not op.params)
    print(f"{label:<28}{len(spec.ops):>7}{const:>10}{const / len(spec.ops) * 100:>7.0f}%")

print("\n3. ACHIEVABLE -- caching only the all-constant blocks\n")
print(f"{'qubits':>7}" + "".join(f"{f'k<={k}':>12}" for k in (2, 3, 4)))
print("-" * 44)
for n in (6, 8, 10, 12):
    ansatz = qk.hardware_efficient(n, 3)
    spec = ansatz.build().bind(ansatz.init(seed=0))
    base = best(lambda: run_unfused(spec))
    reference = run_unfused(spec)
    row = f"{n:>7}"
    for k in (2, 3, 4):
        blocks = plan(spec.ops, k)
        cache = {
            i: block_matrix(w, o)
            for i, (w, o) in enumerate(blocks)
            if len(o) > 1 and all(not op.params for op in o)
        }
        assert np.max(np.abs(run_partial(spec, blocks, cache) - reference)) < 1e-12
        t = best(lambda b=blocks, c=cache: run_partial(spec, b, c))
        row += f"{base / t:>11.2f}x  ({len(cache)} cached)"
    print(row)
