"""Gate fusion prototype: does it actually pay, and where?

The literature says merge 3-5 qubits. The width measurement said a 4-qubit gate
costs 1.3x a 1-qubit gate below 12 qubits. Both suggest fusion wins. Neither
accounts for the cost of *building* the fused matrix, which is a 2^(2k) array --
for k=4 that is 256 complex numbers, four times the size of a 6-qubit state.

So this measures the whole thing end to end, including the build, and checks the
answer against the unfused path before reporting any speedup.
"""

import sys
import time
from pathlib import Path

# resolve qmlkit from this checkout, not from whatever an editable install points at
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import _apply
from qmlkit.core.gates import gate_matrix


def plan(ops, max_qubits):
    """Greedy consecutive grouping. Order is preserved, so this is always correct."""
    blocks, current, wires = [], [], set()
    for op in ops:
        candidate = wires | set(op.qubits)
        if current and len(candidate) > max_qubits:
            blocks.append((tuple(sorted(wires)), current))
            current, wires = [op], set(op.qubits)
        else:
            current.append(op)
            wires = candidate
    if current:
        blocks.append((tuple(sorted(wires)), current))
    return blocks


def block_matrix(wires, ops):
    """Compose the block's gates into one 2^k x 2^k unitary.

    Built by applying each gate to the *output* legs of an identity, in circuit
    order -- the same contraction the state path uses, so a convention error here
    would have to be a convention error there too.
    """
    k = len(wires)
    index = {q: i for i, q in enumerate(wires)}
    u = np.eye(2**k, dtype=complex).reshape((2,) * (2 * k))
    for op in ops:
        local = tuple(index[q] for q in op.qubits)
        u = _apply(u, np.asarray(gate_matrix(op.gate, op.params), dtype=complex), local)
    return u.reshape(2**k, 2**k)


def run_unfused(spec):
    state = np.zeros((2,) * spec.n_qubits, dtype=complex)
    state[(0,) * spec.n_qubits] = 1.0
    for op in spec.ops:
        state = _apply(state, np.asarray(gate_matrix(op.gate, op.params), dtype=complex), op.qubits)
    return state.reshape(-1)


def run_fused(spec, blocks):
    state = np.zeros((2,) * spec.n_qubits, dtype=complex)
    state[(0,) * spec.n_qubits] = 1.0
    for wires, ops in blocks:
        if len(ops) == 1:
            op = ops[0]
            state = _apply(
                state, np.asarray(gate_matrix(op.gate, op.params), dtype=complex), op.qubits
            )
        else:
            state = _apply(state, block_matrix(wires, ops), wires)
    return state.reshape(-1)


def best(fn, repeats=15):
    fn()
    return min(_t(fn) for _ in range(repeats))


def _t(fn):
    s = time.perf_counter()
    fn()
    return time.perf_counter() - s


print("greedy consecutive fusion, hardware-efficient ansatz, 3 layers\n")
print(f"{'qubits':>7}{'gates':>7}  |" + "".join(f"{f'k<={k}':>18}" for k in (2, 3, 4, 5)))
print(f"{'':>7}{'':>7}  |" + "".join(f"{'blocks':>8}{'speedup':>10}" for _ in (2, 3, 4, 5)))
print("-" * 90)

for n in (4, 6, 8, 10, 12, 14, 16):
    ansatz = qk.hardware_efficient(n, 3)
    spec = ansatz.build().bind(ansatz.init(seed=0))
    base = best(lambda: run_unfused(spec))
    reference = run_unfused(spec)
    row = f"{n:>7}{len(spec.ops):>7}  |"
    for k in (2, 3, 4, 5):
        blocks = plan(spec.ops, k)
        got = run_fused(spec, blocks)
        err = float(np.max(np.abs(got - reference)))
        if err > 1e-12:
            row += f"{'WRONG':>18}"
            continue
        t = best(lambda b=blocks: run_fused(spec, b))
        row += f"{len(blocks):>8}{base / t:>9.2f}x"
    print(row)

print("\nwhere the fused time goes at 6 and 14 qubits (k<=4):")
for n in (6, 14):
    ansatz = qk.hardware_efficient(n, 3)
    spec = ansatz.build().bind(ansatz.init(seed=0))
    blocks = plan(spec.ops, 4)
    build = best(lambda: [block_matrix(w, o) for w, o in blocks if len(o) > 1])
    total = best(lambda: run_fused(spec, blocks))
    print(
        f"  n={n:>2}: {len(blocks)} blocks, build {build * 1e6:>7.1f} us"
        f"  total {total * 1e6:>7.1f} us  -> build is {build / total * 100:.0f}% of it"
    )
