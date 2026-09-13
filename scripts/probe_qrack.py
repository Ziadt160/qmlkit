"""Would Qrack actually be faster for the circuits this library runs?

Qrack is a serious C++/OpenCL simulator that beats everything at 30+ qubits. The
question here is narrower: at 4-12 qubits, through its Python binding, running the
shape of circuit a QML training loop runs, does it win?

Times one hardware-efficient ansatz end to end -- the Python-visible cost, which is
what a training loop actually pays, not the C++ kernel time.
"""

import time

import sys

# explicit, because an editable install resolves qmlkit to a different worktree
sys.path.insert(
    0,
    r"C:\Quantum Machine Learning Module\.claude\worktrees"
    r"\qmlkit-pennylane-marketing-83f602\qmlkit\src",
)
sys.path.insert(
    1,
    r"C:\Users\pc\AppData\Local\Temp\claude"
    r"\C--Quantum-Machine-Learning-Module--claude-worktrees-qmlkit-pennylane-marketing-83f602"
    r"\030d1a83-50ab-491b-997d-f11158dcf009\scratchpad\pqk",
)

import numpy as np
from pyqrack import QrackSimulator

import qmlkit as qk

print(f"qmlkit  : {qk.__file__}")
print(f"pyqrack : {QrackSimulator.__module__}\n")

for name in ("r", "u", "mcz", "mcx"):
    print(f"  has {name}: {hasattr(QrackSimulator, name)}")

PAULI_Y = 2  # Pauli enum: I=0, X=1, Y=2, Z=3


def best(fn, repeats=10):
    fn()
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


print(f"\n{'circuit':<16}{'gates':>7}{'qmlkit':>12}{'qrack':>12}{'qrack/qmlkit':>15}")
print("-" * 64)

for n_qubits, n_layers in [(4, 2), (6, 3), (8, 3), (10, 3), (12, 3)]:
    ansatz = qk.hardware_efficient(n_qubits, n_layers)
    theta = ansatz.init(seed=0)
    spec = ansatz.build().bind(theta)
    backend = qk.get_backend("numpy")
    n_gates = len(spec.ops)

    ours = best(lambda: backend.statevector(spec))

    # the same shape of circuit, driven through Qrack's Python API
    angles = np.asarray(theta, dtype=float)

    def qrack_run():
        sim = QrackSimulator(n_qubits, is_gpu=False)
        k = 0
        for _ in range(n_layers):
            for q in range(n_qubits):
                sim.r(PAULI_Y, float(angles[k % len(angles)]), q)
                k += 1
            for q in range(n_qubits - 1):
                sim.mcz([q], q + 1)
        out = sim.out_ket()
        sim.reset_all()
        return out

    theirs = best(qrack_run)
    print(
        f"{f'HEA {n_qubits}q x{n_layers}':<16}{n_gates:>7}{ours * 1000:>11.3f}ms"
        f"{theirs * 1000:>11.3f}ms{theirs / ours:>14.2f}x"
    )

# where does the Qrack time actually go?
print("\nbreaking the 6-qubit Qrack call down:")
sim = QrackSimulator(6, is_gpu=False)
print(f"  construct simulator : {best(lambda: QrackSimulator(6, is_gpu=False), 20) * 1e6:>9.1f} us")
print(f"  one r() gate        : {best(lambda: sim.r(PAULI_Y, 0.3, 0), 200) * 1e6:>9.1f} us")
print(f"  one mcz()           : {best(lambda: sim.mcz([0], 1), 200) * 1e6:>9.1f} us")
print(f"  out_ket()           : {best(lambda: sim.out_ket(), 50) * 1e6:>9.1f} us")
print("\n  (qmlkit's whole 6-qubit, 51-gate circuit costs ~880 us)")
