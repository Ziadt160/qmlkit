"""Cross-validate qmlkit against PennyLane.

    pip install pennylane
    python examples/compare_pennylane.py

Two independent implementations agreeing to machine precision is much stronger
evidence than either one's own test suite. This compares expectation values,
gradients from every method both libraries offer, feature-map kernels, and the
Fourier spectrum of a re-uploading model.

Any disagreement above 1e-10 is reported as a FAIL and the script exits non-zero.

This is the readable version, meant to be run and looked at. The exhaustive one lives
in ``tests/test_pennylane_parity.py`` -- 301 cases including randomised circuit
fuzzing -- so it guards every future change instead of only today's.
"""

from __future__ import annotations

import sys

import numpy as np
import pennylane as qml

import qmlkit as qk

TOL = 1e-10
results: list[tuple[str, float, bool]] = []


def check(name: str, a, b, tol: float = TOL) -> None:
    """Record a comparison and print it."""
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    ok = err <= tol
    results.append((name, err, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} max|diff| = {err:.3e}")


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
header("1. Expectation values")
# --------------------------------------------------------------------------- #
dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def pl_layered(theta):
    """Ry/Rz on every wire, then a CX ring — matched to hardware_efficient."""
    for layer in range(2):
        for w in range(3):
            qml.RY(theta[layer * 6 + w * 2], wires=w)
            qml.RZ(theta[layer * 6 + w * 2 + 1], wires=w)
        for w in range(3):
            qml.CNOT(wires=[w, (w + 1) % 3])
    return qml.expval(qml.PauliZ(0) + 0.5 * qml.PauliZ(0) @ qml.PauliZ(2) + 0.3 * qml.PauliX(1))


qk_ansatz = qk.hardware_efficient(3, n_layers=2, rotations=("ry", "rz"), pattern="ring")
obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2) + 0.3 * qk.X(1)
rng = np.random.default_rng(0)
theta = rng.uniform(-np.pi, np.pi, qk_ansatz.n_params)

check(
    "layered ansatz, 3-term observable",
    qk.expval(qk_ansatz.build(), obs, theta=theta),
    pl_layered(theta),
)


@qml.qnode(dev)
def pl_state(theta):
    qml.RY(theta[0], wires=0)
    qml.RZ(theta[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CRZ(theta[2], wires=[0, 1])
    qml.Hadamard(wires=2)
    qml.CY(wires=[1, 2])
    return qml.state()


qc = qk.QCircuit(3)
qc.ry(0, qk.ParamRef(0)).rz(1, qk.ParamRef(1)).cx(0, 1)
qc.crz(0, 1, qk.ParamRef(2)).h(2).cy(1, 2)
mixed = qc.to_spec()
t3 = np.array([0.7, -1.1, 2.0])
check("statevector, mixed gate set (incl. CRZ, CY)", qk.statevector(mixed.bind(t3)), pl_state(t3))


# --------------------------------------------------------------------------- #
header("2. Gradients — every method, both libraries")
# --------------------------------------------------------------------------- #
# PennyLane only differentiates arrays it marks trainable
theta_pl = qml.numpy.array(theta, requires_grad=True)
pl_backprop = np.asarray(qml.grad(pl_layered)(theta_pl))
print(f"  PennyLane backprop reference computed for P={len(theta)}\n")

# the exact methods must agree to machine precision; finite differences are
# *defined* to be biased, so it gets a tolerance matching its O(eps^2) error
APPROXIMATE = {"finite-diff": 1e-6, "spsa": None}
for method in qk.list_gradient_methods():
    if method == "spsa":
        continue  # stochastic: compared separately, on its mean
    g = qk.grad(qk_ansatz.build(), theta, obs, method=method)
    check(
        f"qmlkit {method:<16} vs PennyLane backprop",
        g,
        pl_backprop,
        tol=APPROXIMATE.get(method, TOL),
    )

for pl_method in ("parameter-shift", "adjoint", "backprop"):
    node = qml.QNode(pl_layered.func, dev, diff_method=pl_method)
    check(
        f"PennyLane {pl_method:<15} vs qmlkit adjoint",
        np.asarray(qml.grad(node)(theta_pl)),
        qk.grad(qk_ansatz.build(), theta, obs, method="adjoint"),
    )

# SPSA is stochastic in both libraries: check it is unbiased, not identical
spsa_avg = np.mean(
    [
        qk.grad(qk_ansatz.build(), theta, obs, method="spsa", seed=s, n_avg=60, c=0.01)
        for s in range(80)
    ],
    axis=0,
)
check("qmlkit SPSA (4800 samples) vs PennyLane backprop", spsa_avg, pl_backprop, tol=0.05)


# --------------------------------------------------------------------------- #
header("3. Feature maps and kernels")
# --------------------------------------------------------------------------- #
x1 = np.array([0.4, 1.3])
x2 = np.array([1.9, 0.6])


@qml.qnode(qml.device("default.qubit", wires=2))
def pl_angle_kernel(a, b):
    qml.AngleEmbedding(a, wires=range(2), rotation="Y")
    qml.adjoint(qml.AngleEmbedding)(b, wires=range(2), rotation="Y")
    return qml.probs(wires=range(2))


qk_angle = qk.AngleFeatureMap(2, entangle=False)
check(
    "angle-map fidelity kernel",
    qk.fidelity_kernel(qk_angle, x1, x2),
    pl_angle_kernel(x1, x2)[0],
)


@qml.qnode(qml.device("default.qubit", wires=2))
def pl_iqp_kernel(a, b):
    qml.IQPEmbedding(a, wires=range(2), n_repeats=2)
    qml.adjoint(qml.IQPEmbedding)(b, wires=range(2), n_repeats=2)
    return qml.probs(wires=range(2))


# A real convention difference, not a bug. PennyLane's IQPEmbedding emits
# RZ(x_i) and MultiRZ(x_i * x_j); qmlkit follows the Qiskit / lecture convention
# and emits Rz(2 * phi). Halving the data map lines them up exactly.
qk_iqp = qk.PauliFeatureMap(
    2,
    paulis=("Z", "ZZ"),
    reps=2,
    data_map=lambda x, idx: float(np.prod([x[i] for i in idx])) / 2,
)
check("IQP / ZZ-map fidelity kernel", qk.fidelity_kernel(qk_iqp, x1, x2), pl_iqp_kernel(x1, x2)[0])


# --------------------------------------------------------------------------- #
header("4. Quantum information")


# --------------------------------------------------------------------------- #
@qml.qnode(qml.device("default.qubit", wires=2))
def pl_bell():
    qml.Hadamard(0)
    qml.CNOT(wires=[0, 1])
    return qml.state()


bell_qc = qk.QCircuit(2)
bell_qc.h(0).cx(0, 1)
bell = bell_qc.to_spec()

check("Bell state", qk.statevector(bell), pl_bell())
pl_dm = np.outer(pl_bell(), np.conj(pl_bell()))
check("reduced density matrix", qk.reduced_dm(bell, [0]), qml.math.reduce_dm(pl_dm, [0]))
check(
    "von Neumann entropy",
    qk.vn_entropy(bell, [0], base=2),
    qml.math.vn_entropy(pl_dm, indices=[0], base=2),
)
check("purity", qk.purity(bell, [0]), qml.math.purity(pl_dm, indices=[0]))


# --------------------------------------------------------------------------- #
header("5. Fourier spectrum of a re-uploading model")
# --------------------------------------------------------------------------- #
n_layers = 3


@qml.qnode(qml.device("default.qubit", wires=1))
def pl_reupload(x, w):
    for layer in range(n_layers):
        qml.RY(x, wires=0)
        qml.RZ(w[layer * 3], wires=0)
        qml.RY(w[layer * 3 + 1], wires=0)
        qml.RZ(w[layer * 3 + 2], wires=0)
    return qml.expval(qml.PauliZ(0))


model = qk.reupload(
    qk.AngleFeatureMap(1, entangle=False),
    n_layers=n_layers,
    rotations=("rz", "ry", "rz"),
    entangler=None,
)
weights = rng.uniform(-np.pi, np.pi, model.n_weights)
bound = model.build()

xs = np.linspace(0, 2 * np.pi, 32, endpoint=False)
qk_values = np.array(
    [qk.expval(bound, qk.Z(0), theta=np.concatenate([model.angles([x]), weights])) for x in xs]
)
pl_values = np.array([float(pl_reupload(x, weights)) for x in xs])
check("re-uploading model, swept over its input", qk_values, pl_values)

qk_spectrum = qk.fourier.spectrum(
    lambda x: qk.expval(bound, qk.Z(0), theta=np.concatenate([model.angles([x]), weights])),
    n_layers + 3,
)
pl_coeffs = qml.fourier.coefficients(lambda x: pl_reupload(x, weights), 1, n_layers)
pl_spectrum = {
    k: float(abs(pl_coeffs[k]) + abs(pl_coeffs[-k])) if k else float(abs(pl_coeffs[0]))
    for k in range(n_layers + 1)
}
print(f"\n  qmlkit    frequencies {sorted(qk_spectrum)}")
print(f"  PennyLane frequencies {sorted(k for k, v in pl_spectrum.items() if v > 1e-8)}")
check(
    "Fourier amplitudes",
    [qk_spectrum.get(k, 0.0) for k in range(n_layers + 1)],
    [pl_spectrum.get(k, 0.0) for k in range(n_layers + 1)],
)


# --------------------------------------------------------------------------- #
header("Summary")
# --------------------------------------------------------------------------- #
passed = sum(1 for _, _, ok in results if ok)
worst = max(results, key=lambda r: r[1])
print(f"  {passed}/{len(results)} comparisons agree")
print(f"  largest disagreement: {worst[1]:.3e}  ({worst[0]})")
if passed == len(results):
    print("\n  qmlkit and PennyLane agree on every comparison.")
    sys.exit(0)
print("\n  MISMATCHES:")
for name, err, ok in results:
    if not ok:
        print(f"    {name}: {err:.3e}")
sys.exit(1)
