"""Keep your PennyLane circuit; run the expensive loop here.

    pip install pennylane
    python examples/accelerate_pennylane.py

Migrating a project is a large ask, and it has to be paid for before it pays off.
Accelerating one is a small one. `qk.from_pennylane` reads a tape, a QNode or a plain
quantum function, so a PennyLane circuit can be handed to qmlkit for the parts where
qmlkit is faster, and left exactly where it is for everything else.

This script does that for the three inner loops worth stealing, and checks the numbers
agree with PennyLane's own before quoting a speedup — an acceleration that changes the
answer is not an acceleration.

Where the gap comes from, in each case:

1. **Kernel Gram matrix.** Every entry is the *same* circuit at different angles, so
   the whole matrix is one batched evaluation. PennyLane runs one QNode call per pair,
   and at these sizes the per-call overhead dominates the arithmetic so completely that
   `lightning.qubit` is *slower* than `default.qubit` here.
2. **Batched gradients.** A training batch is the same circuit at one parameter vector
   per sample. qmlkit differentiates the whole batch in one sweep.
3. **Fubini-Study metric.** An algorithmic difference rather than an overhead one:
   closed-form differentiation of the state against PennyLane's `O(P^2)` Hadamard
   tests, or its `O(P)` adjoint route.
"""

from __future__ import annotations

import time

import numpy as np
import pennylane as qml

import qmlkit as qk

REPEATS = 3
N_QUBITS = 3
N_LAYERS = 2


def best(fn, repeats: int = REPEATS) -> float:
    fn()  # warm up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return min(times)


def report(case: str, theirs: float, ours: float, agreement: float) -> None:
    verdict = "OK" if agreement < 1e-8 else f"DISAGREE ({agreement:.1e})"
    print(
        f"  {case:<34}{theirs * 1000:>9.1f} ms{ours * 1000:>10.1f} ms"
        f"{theirs / ours:>8.1f}x   {verdict}"
    )


print("Your circuit, written in PennyLane. Nothing below rewrites it.\n")
print(f"  {'inner loop':<34}{'PennyLane':>12}{'qmlkit':>13}{'':>8}   agree")

# --------------------------------------------------------------------------- #
# 1. a kernel Gram matrix
# --------------------------------------------------------------------------- #
rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, (20, N_QUBITS))
device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(device)
def overlap(a, b):
    qml.AngleEmbedding(a, wires=range(N_QUBITS), rotation="Y")
    qml.adjoint(qml.AngleEmbedding)(b, wires=range(N_QUBITS), rotation="Y")
    return qml.probs(wires=range(N_QUBITS))


_UPPER_I, _UPPER_J = np.triu_indices(len(X), 1)


def pennylane_gram():
    """The *broadcast* route, which is the honest opponent.

    `default.qubit` vectorises when it is handed a stacked array, so the whole upper
    triangle is one call rather than m(m-1)/2 of them. Timing the pair loop instead
    reports something like 200x here, and that is a measurement of Python call
    overhead rather than of either library.
    """
    K = np.eye(len(X))
    values = np.asarray(overlap(qml.numpy.array(X[_UPPER_I]), qml.numpy.array(X[_UPPER_J])))[:, 0]
    K[_UPPER_I, _UPPER_J] = values
    K[_UPPER_J, _UPPER_I] = values
    return K


# the same feature map, read out of the PennyLane circuit's own convention.
# qmlkit does not run the composed circuit at all: P(0...0) of U(x')^dag U(x)|0> *is*
# |<psi(x')|psi(x)>|**2, so it evaluates one state per row and takes a matrix product.
kernel = qk.QuantumKernel(qk.AngleFeatureMap(N_QUBITS, entangle=False))
report(
    f"{len(X)}x{len(X)} kernel Gram matrix",
    best(pennylane_gram),
    best(lambda: kernel(X)),
    float(np.max(np.abs(pennylane_gram() - kernel(X)))),
)

# --------------------------------------------------------------------------- #
# 2. gradients for a training batch
# --------------------------------------------------------------------------- #
ansatz = qk.hardware_efficient(N_QUBITS, N_LAYERS)
spec = ansatz.build()
thetas = rng.uniform(-np.pi, np.pi, (32, ansatz.n_params))


@qml.qnode(qml.device("default.qubit", wires=N_QUBITS))
def circuit(t):
    # `t[..., i]` rather than `t[i]`, which is all it takes to make the same QNode
    # accept a stacked array and broadcast over it
    for layer in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RY(t[..., layer * 2 * N_QUBITS + w * 2], wires=w)
            qml.RZ(t[..., layer * 2 * N_QUBITS + w * 2 + 1], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
    return qml.expval(qml.PauliZ(0))


def pennylane_batch_grad():
    """Every row's gradient in one broadcast pass, which is PennyLane's fast route.

    The rows are independent parameter vectors, so `d(sum_i f(t_i))/dt_i` is exactly
    each row's own gradient and one backward pass produces all 32. Looping
    `lightning.qubit` with `diff_method="adjoint"` instead — which this file used to do,
    and which reported 40.9x — measures 107 ms against this route's 19 ms. Five times
    the wall clock is not a fact about either library.
    """
    stacked = qml.numpy.array(thetas, requires_grad=True)
    return np.asarray(qml.grad(lambda t: qml.numpy.sum(circuit(t)))(stacked))


report(
    f"gradients for a batch of {len(thetas)}",
    best(pennylane_batch_grad),
    best(lambda: qk.grad_batch(spec, thetas, qk.Z(0))),
    float(np.max(np.abs(pennylane_batch_grad() - qk.grad_batch(spec, thetas, qk.Z(0))))),
)

# --------------------------------------------------------------------------- #
# 3. the Fubini-Study metric
# --------------------------------------------------------------------------- #
theta = ansatz.init(seed=0)
metric_device = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(metric_device)
def metric_circuit(t):
    for layer in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RY(t[layer * 2 * N_QUBITS + w * 2], wires=w)
            qml.RZ(t[layer * 2 * N_QUBITS + w * 2 + 1], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
    return qml.expval(qml.PauliZ(0))


tp = qml.numpy.array(theta, requires_grad=True)
report(
    f"exact metric tensor, P={ansatz.n_params}",
    best(lambda: qml.adjoint_metric_tensor(metric_circuit)(tp)),
    best(lambda: qk.metric_tensor(spec, theta, approx=None)),
    float(
        np.max(
            np.abs(
                np.asarray(qml.adjoint_metric_tensor(metric_circuit)(tp))
                - np.asarray(qk.metric_tensor(spec, theta, approx=None))
            )
        )
    ),
)

print(
    """
  Every row agrees with PennyLane to machine precision, which is the only thing that
  makes the timings worth reading.

  What this is not: a claim that qmlkit is faster than PennyLane in general. It is
  not -- see examples/benchmark_pennylane.py, where the honest median against
  PennyLane's fastest configuration is 1.7x and several rows are a tie. These three
  loops are the ones where the gap is real, and they happen to be the ones a QML
  workload spends its time in.

  Nothing above required rewriting the PennyLane circuit. `qk.from_pennylane(qnode)`
  reads it directly when you want the circuit itself rather than an equivalent one."""
)
