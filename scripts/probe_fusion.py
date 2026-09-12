"""Two ratios decide whether gate fusion is worth building.

1. Does a WIDER gate cost more? If a 4-qubit gate costs about what a 1-qubit gate
   costs, then merging k gates into one wider one is nearly free, and fusion buys
   close to a k-fold cut in NumPy calls.
2. What does one full 2^n matvec cost, vs the gate-by-gate loop? That is the floor
   any compiled path could aim at.
"""

import time

import numpy as np

REPEATS = 200


def best(fn, repeats=REPEATS):
    fn()
    return min(_t(fn) for _ in range(repeats))


def _t(fn):
    s = time.perf_counter()
    fn()
    return time.perf_counter() - s


def apply(matrix, state, qubits):
    k = len(qubits)
    op = matrix.reshape((2,) * (2 * k))
    out = np.tensordot(op, state, axes=(list(range(k, 2 * k)), list(qubits)))
    return np.moveaxis(out, list(range(k)), list(qubits))


print("1. cost per gate application, by gate WIDTH\n")
print(f"{'n_qubits':>9}{'1q':>10}{'2q':>10}{'3q':>10}{'4q':>10}   (microseconds)")
print("-" * 60)

for n in (6, 8, 10, 12, 14):
    state = np.zeros((2,) * n, dtype=complex)
    state.flat[0] = 1.0
    row = []
    for k in (1, 2, 3, 4):
        rng = np.random.default_rng(0)
        m = rng.normal(size=(2**k, 2**k)) + 1j * rng.normal(size=(2**k, 2**k))
        wires = tuple(range(k))
        row.append(best(lambda m=m, wires=wires: apply(m, state, wires)) * 1e6)
    print(f"{n:>9}" + "".join(f"{v:>10.1f}" for v in row))

print("\n2. one full 2^n matvec (what a compiled/fused path could approach)\n")
print(f"{'n_qubits':>9}{'matvec us':>12}{'51 x 1q us':>13}{'ratio':>9}")
print("-" * 45)

for n in (6, 8, 10, 12):
    dim = 2**n
    rng = np.random.default_rng(0)
    U = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    v = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    mv = best(lambda: U @ v) * 1e6

    state = np.zeros((2,) * n, dtype=complex)
    state.flat[0] = 1.0
    g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))

    def loop51():
        s = state
        for i in range(51):
            s = apply(g, s, (i % n,))
        return s

    lp = best(loop51, repeats=20) * 1e6
    print(f"{n:>9}{mv:>12.1f}{lp:>13.1f}{lp / mv:>9.1f}x")

print("\n3. how many NumPy calls does BLAS threading even reach?")
print("   (a 2^n matvec is one BLAS call; a gate loop is 2 calls per gate)")
for n in (6, 10, 14, 18):
    print(f"   n={n:>2}: statevector is {2**n:>7} complex = {2**n * 16 / 1024:>8.1f} KiB")
