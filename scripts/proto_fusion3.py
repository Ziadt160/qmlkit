"""The gap is the build, and the build is per-call overhead again.

Composing m gates into a block matrix currently costs m separate _apply calls on a
2^(2k) array -- so the same ~10us NumPy call cost that fusion exists to avoid is
being paid inside the thing that avoids it.

But composing m gates IS a tensor-network contraction, and einsum expresses one of
those in a SINGLE call. The subscripts depend only on the block's structure, not on
its angles, so they can be computed once per circuit and reused every step while the
angles change underneath.

If that works, the build stops scaling with m and fusion reaches the ceiling.
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

from proto_fusion import best, block_matrix, plan, run_unfused

LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def einsum_plan(wires, ops):
    """Subscripts that contract this block's gates into one matrix, in one call.

    Each wire carries a running index label. A gate consumes its wires' current
    labels as inputs and issues fresh ones as outputs -- which is precisely the
    circuit read as a tensor network, left to right.
    """
    index = {q: i for i, q in enumerate(wires)}
    nxt = len(wires)
    current = list(range(len(wires)))  # label per wire, starts at the input legs
    operands = []
    for op in ops:
        local = [index[q] for q in op.qubits]
        outs = []
        for _ in local:
            outs.append(nxt)
            nxt += 1
        ins = [current[i] for i in local]
        operands.append(outs + ins)
        for slot, label in zip(local, outs):
            current[slot] = label
    output = current + list(range(len(wires)))
    subs = ",".join("".join(LETTERS[i] for i in o) for o in operands)
    return f"{subs}->{''.join(LETTERS[i] for i in output)}", nxt


def block_matrix_einsum(wires, ops, subscripts):
    k = len(wires)
    mats = [
        np.asarray(gate_matrix(op.gate, op.params), dtype=complex).reshape((2,) * (2 * len(op.qubits)))
        for op in ops
    ]
    return np.einsum(subscripts, *mats, optimize=True).reshape(2**k, 2**k)


def run_fused_einsum(spec, blocks, plans):
    state = np.zeros((2,) * spec.n_qubits, dtype=complex)
    state[(0,) * spec.n_qubits] = 1.0
    for (wires, ops), subs in zip(blocks, plans):
        if len(ops) == 1:
            op = ops[0]
            state = _apply(
                state, np.asarray(gate_matrix(op.gate, op.params), dtype=complex), op.qubits
            )
        else:
            state = _apply(state, block_matrix_einsum(wires, ops, subs), wires)
    return state.reshape(-1)


def _run_seq(spec, blocks):
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


print("does an einsum build beat the sequential build, and does fusion then pay?\n")
print(f"{'qubits':>7}{'gates':>7}   " + "".join(f"{f'k<={k}':>22}" for k in (3, 4, 5)))
print(f"{'':>7}{'':>7}   " + "".join(f"{'seq':>10}{'einsum':>12}" for _ in (3, 4, 5)))
print("-" * 80)

for n in (4, 6, 8, 10, 12, 14):
    ansatz = qk.hardware_efficient(n, 3)
    spec = ansatz.build().bind(ansatz.init(seed=0))
    base = best(lambda: run_unfused(spec))
    reference = run_unfused(spec)
    row = f"{n:>7}{len(spec.ops):>7}   "
    for k in (3, 4, 5):
        blocks = plan(spec.ops, k)
        plans = [einsum_plan(w, o)[0] if len(o) > 1 else None for w, o in blocks]

        seq = best(lambda b=blocks: _run_seq(spec, b))
        got = run_fused_einsum(spec, blocks, plans)
        err = float(np.max(np.abs(got - reference)))
        if err > 1e-12:
            row += f"{base / seq:>9.2f}x{'WRONG':>12}"
            continue
        ein = best(lambda b=blocks, p=plans: run_fused_einsum(spec, b, p))
        row += f"{base / seq:>9.2f}x{base / ein:>11.2f}x"
    print(row)
