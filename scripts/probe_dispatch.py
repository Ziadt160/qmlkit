"""Is the NumPy backend dispatch-bound? Measure, do not infer.

Splits one statevector() call into (a) what the arithmetic costs and (b) what is
left, which is Python and NumPy dispatch. If (b) dominates, threads cannot help
and fusion/compilation can.
"""

import time

import numpy as np

import qmlkit as qk
from qmlkit.core.gates import gate_matrix

print(f"qmlkit from: {qk.__file__}")
print(f"numpy {np.__version__}\n")


def best(fn, repeats=7):
    fn()
    return min(_time(fn) for _ in range(repeats))


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


backend = qk.get_backend("numpy")

print(f"{'circuit':<22}{'ops':>6}{'total ms':>11}{'us/op':>9}{'arith ms':>11}{'dispatch':>10}")
print("-" * 69)

for n_qubits, n_layers in [(4, 2), (6, 3), (8, 3), (10, 3), (12, 3), (14, 3)]:
    ansatz = qk.hardware_efficient(n_qubits, n_layers)
    theta = ansatz.init(seed=0)
    spec = ansatz.build().bind(theta)
    n_ops = len(spec.ops)

    total = best(lambda: backend.statevector(spec))

    # the same arithmetic, with no per-op Python: one tensordot+moveaxis per op,
    # on a preallocated state, matrices already built.
    mats = [np.asarray(gate_matrix(op.gate, op.params), dtype=complex) for op in spec.ops]
    wires = [tuple(op.qubits) for op in spec.ops]
    state0 = np.zeros((2,) * n_qubits, dtype=complex)
    state0.flat[0] = 1.0

    def arithmetic_only():
        state = state0
        for matrix, qubits in zip(mats, wires):
            k = len(qubits)
            op = matrix.reshape((2,) * (2 * k))
            state = np.tensordot(op, state, axes=(list(range(k, 2 * k)), list(qubits)))
            state = np.moveaxis(state, list(range(k)), list(qubits))
        return state

    arith = best(arithmetic_only)
    dispatch_share = (total - arith) / total * 100

    print(
        f"{f'HEA {n_qubits}q x{n_layers}':<22}{n_ops:>6}{total * 1000:>11.3f}"
        f"{total / n_ops * 1e6:>9.1f}{arith * 1000:>11.3f}{dispatch_share:>9.0f}%"
    )

# How much of the "arithmetic" is itself NumPy call overhead rather than FLOPs?
print("\nNumPy call overhead floor (empty-ish tensordot on a tiny array):")
tiny = np.zeros((2,) * 6, dtype=complex)
tiny.flat[0] = 1.0
gate = np.eye(2, dtype=complex).reshape(2, 2)
t = best(lambda: np.moveaxis(np.tensordot(gate, tiny, axes=([1], [0])), [0], [0]), repeats=50)
print(f"  one tensordot+moveaxis on 6 qubits: {t * 1e6:.2f} us")
print(f"  -> a 54-gate circuit cannot beat  : {t * 54 * 1e6:.1f} us on dispatch alone")
