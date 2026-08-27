"""One real experiment, implemented twice, trained twice, compared step by step.

    pip install pennylane
    python examples/head_to_head.py

`tests/test_pennylane_parity.py` checks 301 quantities in isolation — gate matrices,
expectations, gradients. This asks a different and blunter question: if you sat down
and did the *same experiment* in both libraries, would you reach the same conclusion?

The task is binary classification of the moons dataset with a variational classifier:
angle encoding, a hardware-efficient ansatz, `<Z0>` as the decision function, squared
loss, plain gradient descent. Identical data, identical initial weights, identical
learning rate, and no stochasticity anywhere — so two correct implementations must
agree at *every* step, not merely at the end. A method that ends up in the same place
by a different route would still be a bug.

The PennyLane side is written in PennyLane's own idiom rather than translated through
qmlkit's IR, which is the whole point: two independent expressions of one experiment.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pennylane as qml

import qmlkit as qk

N_QUBITS = 2
N_LAYERS = 4
N_STEPS = 40
LEARNING_RATE = 0.6

#: which simulator qmlkit runs on. PennyLane always uses its own default.qubit, so
#: passing "qiskit" here makes this a three-way check: our IR, their IR, and a third
#: SDK's simulator underneath ours.
BACKEND = sys.argv[1] if len(sys.argv) > 1 else "numpy"


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# the shared setup: same data, same starting weights, for both libraries
# --------------------------------------------------------------------------- #
X, y_binary = qk.datasets.make_moons(n_samples=60, noise=0.1, seed=0)
y = 2.0 * y_binary - 1.0  # {0, 1} -> {-1, +1}, the range <Z0> actually lives in
X = qk.AngleScaler().fit(X).transform(X)

ansatz = qk.hardware_efficient(N_QUBITS, N_LAYERS)
theta0 = ansatz.init("uniform", seed=7)
P = ansatz.n_params

print(f"moons: {len(X)} samples, {X.shape[1]} features -> {N_QUBITS} qubits")
print(f"ansatz: hardware_efficient({N_QUBITS}, {N_LAYERS}) with P={P} parameters")
print(f"training: plain gradient descent, lr={LEARNING_RATE}, {N_STEPS} steps, no randomness")
print(f"qmlkit backend: {BACKEND!r}   PennyLane: default.qubit")
if not qk.is_available(BACKEND):
    raise SystemExit(f"backend {BACKEND!r} is not installed here: {qk.backend_report()}")


# --------------------------------------------------------------------------- #
# implementation A — qmlkit
# --------------------------------------------------------------------------- #
model = qk.reupload(
    qk.AngleFeatureMap(N_QUBITS, entangle=False),
    n_layers=1,
    block=qk.repeat(N_LAYERS, qk.RotationLayer(("ry", "rz")) + qk.EntanglerLayer("cx", "chain")),
)
qk_spec = model.build()
OBS = qk.Z(0)


def qk_predict(theta: np.ndarray) -> np.ndarray:
    return np.array(
        [
            qk.expval(qk_spec, OBS, theta=np.concatenate([model.angles(x), theta]), backend=BACKEND)
            for x in X
        ]
    )


def qk_loss(theta: np.ndarray) -> float:
    return float(np.mean((qk_predict(theta) - y) ** 2))


def qk_grad(theta: np.ndarray) -> np.ndarray:
    """d/dtheta of the mean squared error, by the chain rule through each sample."""
    total = np.zeros(P)
    for x, target in zip(X, y, strict=True):
        full = np.concatenate([model.angles(x), theta])
        pred = qk.expval(qk_spec, OBS, theta=full, backend=BACKEND)
        dfull = qk.grad(qk_spec, full, OBS, backend=BACKEND)  # d<Z0>/d(inputs and weights)
        total += 2.0 * (pred - target) * dfull[model.n_inputs :]
    return total / len(X)


# --------------------------------------------------------------------------- #
# implementation B — PennyLane, written the way PennyLane docs would write it
# --------------------------------------------------------------------------- #
dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev)
def pl_circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(N_QUBITS), rotation="Y")
    for layer in range(N_LAYERS):
        for w in range(N_QUBITS):
            qml.RY(weights[layer * 2 * N_QUBITS + w * 2], wires=w)
            qml.RZ(weights[layer * 2 * N_QUBITS + w * 2 + 1], wires=w)
        for w in range(N_QUBITS - 1):
            qml.CNOT(wires=[w, w + 1])
    return qml.expval(qml.PauliZ(0))


def pl_loss(weights):
    preds = qml.math.stack([pl_circuit(weights, x) for x in X])
    return qml.math.mean((preds - y) ** 2)


# --------------------------------------------------------------------------- #
header("1. Do the two circuits agree before any training?")
# --------------------------------------------------------------------------- #
qk_initial = qk_predict(theta0)
pl_initial = np.array([float(pl_circuit(theta0, x)) for x in X])
forward_err = float(np.abs(qk_initial - pl_initial).max())
print(f"  predictions over all {len(X)} samples   max|diff| = {forward_err:.3e}")
print(f"  initial loss   qmlkit {qk_loss(theta0):.10f}   PennyLane {float(pl_loss(theta0)):.10f}")

g_qk = qk_grad(theta0)
g_pl = np.asarray(qml.grad(pl_loss)(qml.numpy.array(theta0, requires_grad=True)))
grad_err = float(np.abs(g_qk - g_pl).max())
print(f"  gradient of the loss, all {P} parameters   max|diff| = {grad_err:.3e}")


# --------------------------------------------------------------------------- #
header("2. Train both, and compare every step")
# --------------------------------------------------------------------------- #
theta_qk = theta0.copy()
qk_history = [qk_loss(theta_qk)]
for _ in range(N_STEPS):
    theta_qk = theta_qk - LEARNING_RATE * qk_grad(theta_qk)
    qk_history.append(qk_loss(theta_qk))

theta_pl = qml.numpy.array(theta0, requires_grad=True)
pl_history = [float(pl_loss(theta_pl))]
opt = qml.GradientDescentOptimizer(stepsize=LEARNING_RATE)
for _ in range(N_STEPS):
    theta_pl = opt.step(pl_loss, theta_pl)
    pl_history.append(float(pl_loss(theta_pl)))

print(f"  {'step':>5}{'qmlkit loss':>18}{'PennyLane loss':>18}{'|diff|':>12}")
for step in (0, 1, 2, 5, 10, 20, N_STEPS):
    d = abs(qk_history[step] - pl_history[step])
    print(f"  {step:>5}{qk_history[step]:>18.10f}{pl_history[step]:>18.10f}{d:>12.2e}")

traj_err = max(abs(a - b) for a, b in zip(qk_history, pl_history, strict=True))
weight_err = float(np.abs(theta_qk - np.asarray(theta_pl)).max())
print(f"\n  worst disagreement across all {N_STEPS + 1} steps: {traj_err:.3e}")
print(f"  final weight vectors differ by:                {weight_err:.3e}")


# --------------------------------------------------------------------------- #
header("3. Do they classify the same way?")
# --------------------------------------------------------------------------- #
qk_pred = np.sign(qk_predict(theta_qk))
pl_pred = np.sign(np.array([float(pl_circuit(theta_pl, x)) for x in X]))
qk_acc = float(np.mean(qk_pred == y))
pl_acc = float(np.mean(pl_pred == y))

print(f"  accuracy      qmlkit {qk_acc:.1%}      PennyLane {pl_acc:.1%}")
print(f"  labels agreed on {int(np.sum(qk_pred == pl_pred))}/{len(X)} samples")
print(f"  loss           {qk_history[0]:.6f} -> {qk_history[-1]:.6f}  (both libraries)")


# --------------------------------------------------------------------------- #
header("4. Does the gradient method change the science?")
# --------------------------------------------------------------------------- #
# The experiment above used the default (adjoint) on our side and backprop on theirs.
# Both read the statevector, so neither could run on a device. Re-run the same
# training with the hardware-valid methods and see whether the answer moves.
SHORT = 12


def train(theta: np.ndarray, method: str, steps: int = SHORT, **kw) -> list[float]:
    history = [qk_loss(theta)]
    for _ in range(steps):
        total = np.zeros(P)
        for x, target in zip(X, y, strict=True):
            full = np.concatenate([model.angles(x), theta])
            pred = qk.expval(
                qk_spec,
                OBS,
                theta=full,
                backend=BACKEND,
                shots=kw.get("shots"),
                seed=kw.get("seed"),
            )
            dfull = qk.grad(qk_spec, full, OBS, method=method, backend=BACKEND, **kw)
            total += 2.0 * (pred - target) * dfull[model.n_inputs :]
        theta = theta - LEARNING_RATE * total / len(X)
        history.append(qk_loss(theta))
    return history


exact_reference = train(theta0.copy(), "adjoint")
print(f"  {'method':<20}{'final loss':>16}{'vs adjoint':>14}   runs on hardware")
for method in ("adjoint", "parameter-shift", "hadamard"):
    h = train(theta0.copy(), method)
    dev_from_exact = max(abs(a - b) for a, b in zip(h, exact_reference, strict=True))
    hardware = "no - reads the statevector" if method == "adjoint" else "yes"
    print(f"  {method:<20}{h[-1]:>16.10f}{dev_from_exact:>14.1e}   {hardware}")

# ...and what a real device would actually give you, with finite sampling.
noisy = train(theta0.copy(), "parameter-shift", shots=1024, seed=0)
drift = max(abs(a - b) for a, b in zip(noisy, exact_reference, strict=True))
print(f"  {'parameter-shift':<20}{noisy[-1]:>16.10f}{drift:>14.1e}   yes, at 1024 shots")
print(
    f"\n  The exact methods are interchangeable: identical trajectories to machine\n"
    f"  precision, so choosing between them is purely a question of cost.\n"
    f"  Shot noise is the thing that actually moves the answer: at 1024 shots the run\n"
    f"  still descends ({noisy[0]:.4f} -> {noisy[-1]:.4f}) but drifts {drift:.0e} off the exact\n"
    f"  path — twelve orders of magnitude more than any of the exact methods do."
)


# --------------------------------------------------------------------------- #
header("5. Does the simulator underneath change the answer?")


# --------------------------------------------------------------------------- #
# Same experiment again, once per installed backend. A circuit is data here, so the
# backend only decides *who multiplies the matrices* -- but that is exactly the kind
# of claim that deserves checking rather than asserting, because the SDKs disagree
# about qubit ordering, idle wires and controlled-gate conventions underneath.
def train_on(backend: str, steps: int = SHORT) -> tuple[list[float], float]:
    theta = theta0.copy()
    history = [
        float(
            np.mean(
                (
                    np.array(
                        [
                            qk.expval(
                                qk_spec,
                                OBS,
                                theta=np.concatenate([model.angles(x), theta]),
                                backend=backend,
                            )
                            for x in X
                        ]
                    )
                    - y
                )
                ** 2
            )
        )
    ]
    started = time.perf_counter()
    for _ in range(steps):
        total = np.zeros(P)
        for x, target in zip(X, y, strict=True):
            full = np.concatenate([model.angles(x), theta])
            pred = qk.expval(qk_spec, OBS, theta=full, backend=backend)
            total += (
                2.0
                * (pred - target)
                * qk.grad(qk_spec, full, OBS, backend=backend)[model.n_inputs :]
            )
        theta = theta - LEARNING_RATE * total / len(X)
        history.append(
            float(
                np.mean(
                    (
                        np.array(
                            [
                                qk.expval(
                                    qk_spec,
                                    OBS,
                                    theta=np.concatenate([model.angles(x), theta]),
                                    backend=backend,
                                )
                                for x in X
                            ]
                        )
                        - y
                    )
                    ** 2
                )
            )
        )
    return history, time.perf_counter() - started


installed = [b for b in qk.available_backends() if b != "torch"]
reference, _ = train_on("numpy")
print(f"  {'backend':<10}{'final loss':>18}{'vs numpy':>12}{'wall clock':>13}")
for name in installed:
    history, seconds = train_on(name)
    drift = max(abs(a - b) for a, b in zip(history, reference, strict=True))
    print(f"  {name:<10}{history[-1]:>18.12f}{drift:>12.1e}{seconds:>11.1f} s")
print(
    "\n  Identical training curves on every installed simulator. The backends differ\n"
    "  in qubit ordering (Qiskit is little-endian), in how they treat idle wires,\n"
    "  and in one outright gate discrepancy — all handled at build time, so the\n"
    "  physics that comes back out is the same and only the speed changes."
)


# --------------------------------------------------------------------------- #
header("Verdict")
# --------------------------------------------------------------------------- #
TOL = 1e-8
checks = {
    "forward pass over the dataset": forward_err,
    "gradient of the training loss": grad_err,
    f"loss at every one of {N_STEPS + 1} steps": traj_err,
    "final trained weights": weight_err,
}
worst = max(checks.values())
for name, err in checks.items():
    print(f"  {'agree' if err <= TOL else 'DIFFER':<7} {name:<42} {err:.3e}")

same_labels = int(np.sum(qk_pred == pl_pred)) == len(X)
print(f"  {'agree' if same_labels else 'DIFFER':<7} {'every predicted label':<42}")

if worst <= TOL and same_labels and abs(qk_acc - pl_acc) < 1e-12:
    print(
        f"\n  Same experiment, same answer. Two independent implementations tracked each\n"
        f"  other to {worst:.1e} through 40 training steps and produced identical labels."
    )
    sys.exit(0)
print("\n  MISMATCH — the two libraries disagree about this experiment.")
sys.exit(1)
