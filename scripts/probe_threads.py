"""Where does splitting one gate across threads start to pay?

The standard HPC statevector trick: a single-qubit gate on wire k partitions the
2^n amplitudes into 2^(n-1) independent pairs, so the update is embarrassingly
parallel over the leading axis. Qiskit Aer ships exactly this and disables it below
14 qubits by default (`statevector_parallel_threshold`). This measures where the
crossover actually is on this machine, for this kernel.

NumPy releases the GIL inside these ops, so a ThreadPoolExecutor really does get
parallelism here -- this is not a GIL artefact.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

CORES = os.cpu_count() or 1
print(f"logical cores: {CORES}\n")

GATE = np.array([[0.6 + 0.0j, 0.8 + 0.0j], [0.8 + 0.0j, -0.6 + 0.0j]], dtype=complex)


def apply_range(state, m, lo, hi):
    """The pair update for rows [lo, hi) of the leading axis. Writes in place."""
    a = state[lo:hi, 0, :]
    b = state[lo:hi, 1, :]
    top = m[0, 0] * a + m[0, 1] * b
    bot = m[1, 0] * a + m[1, 1] * b
    state[lo:hi, 0, :] = top
    state[lo:hi, 1, :] = bot


def best(fn, repeats):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


print("one single-qubit gate, split across threads over the leading axis")
print("(times in microseconds; speedup is vs 1 thread)\n")
header = f"{'qubits':>7}{'MiB':>9}{'1 thr':>10}"
for t in (2, 4, 8):
    header += f"{f'{t} thr':>10}{'':>7}"
print(header)
print(f"{'':>7}{'':>9}{'':>10}" + "".join(f"{'us':>10}{'x':>7}" for _ in (2, 4, 8)))
print("-" * 78)

for n in (8, 10, 12, 14, 16, 18, 20, 22):
    # view as (2^a, 2, 2^b) with the target in the middle: split the leading axis
    target = n // 2
    lead = 2**target
    tail = 2 ** (n - target - 1)
    state = np.zeros((lead, 2, tail), dtype=complex)
    state.flat[0] = 1.0
    mib = state.nbytes / 1024 / 1024
    repeats = 50 if n <= 16 else 8

    single = best(lambda: apply_range(state, GATE, 0, lead), repeats)
    row = f"{n:>7}{mib:>9.2f}{single * 1e6:>10.1f}"

    for n_threads in (2, 4, 8):
        if lead < n_threads:
            row += f"{'-':>10}{'-':>7}"
            continue
        edges = [lead * i // n_threads for i in range(n_threads + 1)]
        chunks = list(zip(edges[:-1], edges[1:]))
        pool = ThreadPoolExecutor(max_workers=n_threads)

        def threaded():
            list(pool.map(lambda c: apply_range(state, GATE, c[0], c[1]), chunks))

        multi = best(threaded, repeats)
        pool.shutdown()
        row += f"{multi * 1e6:>10.1f}{single / multi:>7.2f}"
    print(row)

print("\nthread-pool dispatch floor (submit 8 no-op tasks and join):")
pool = ThreadPoolExecutor(max_workers=8)
floor = best(lambda: list(pool.map(lambda _: None, range(8))), 200)
pool.shutdown()
print(f"  {floor * 1e6:.1f} us  <- any gate cheaper than this cannot be threaded at a profit")
print(f"  a 6-qubit gate costs ~11 us, so the floor is {floor * 1e6 / 11:.1f}x the whole gate")
