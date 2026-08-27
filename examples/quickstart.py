"""qmlkit quickstart — every layer of the library, end to end.

    python examples/quickstart.py

Runs on the built-in NumPy backend, so it needs nothing but qmlkit itself. The
sklearn and torch sections skip themselves if those extras are missing.
"""

from __future__ import annotations

import importlib.util
import warnings

import numpy as np

import qmlkit as qk

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def header(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
header("1. Circuits are data")
# --------------------------------------------------------------------------- #
qc = qk.QCircuit(3)
qc.rotation_layer(("ry", "rz")).entangle("ring")
spec = qc.to_spec()

print(qk.draw(spec))
print(f"\nparameters {spec.n_params} · depth {spec.depth()} · gates {spec.gate_counts()}")
print(f"backends available here: {qk.available_backends()}")


# --------------------------------------------------------------------------- #
header("2. Expectations are exact unless you ask for shots")
# --------------------------------------------------------------------------- #
angle = qk.angle_encode([0.7])
print(f"<Z> of Ry(0.7)|0>   = {qk.expval(angle, qk.Z(0)):.15f}")
print(f"cos(0.7)            = {np.cos(0.7):.15f}")

for shots in (100, 10_000, 1_000_000):
    value, err = qk.expectation(angle, qk.Z(0), shots=shots, seed=0, return_std=True)
    print(f"  shots {shots:>9,}: {value:+.5f} +- {err:.5f}")
print(f"shots for a precision of 0.001: {qk.shots_for_precision(0.001):,}")


# --------------------------------------------------------------------------- #
header("3. Six ways to compute the same gradient")
# --------------------------------------------------------------------------- #
ansatz = qk.hardware_efficient(3, n_layers=2)
theta = ansatz.init(seed=0)
obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
reference = qk.grad(ansatz.build(), theta, obs, method="adjoint")

# Exactness is a property of the algorithm, not of one measurement: finite
# differences can land inside 1e-9 at a lucky eps and still be biased by construction.
KIND = {
    "adjoint": "exact",
    "backprop": "exact",
    "hadamard": "exact",
    "parameter-shift": "exact",
    "finite-diff": "biased, O(h^2)",
    "spsa": "stochastic",
}
print(f"{'method':<18}{'circuits':>10}   {'max error vs adjoint':>22}   {'kind':<16}")
for method in qk.list_gradient_methods():
    kwargs = {"seed": 0, "n_avg": 50} if method == "spsa" else {}
    try:
        g = qk.grad(ansatz.build(), theta, obs, method=method, **kwargs)
    except qk.BackendNotAvailable as exc:
        # registered but not installed -- show what a bare install would show
        print(f"{method:<18}{'-':>10}   {exc.args[0].splitlines()[0]:>22}")
        continue
    cost = qk.gradient_cost(ansatz.build(), method)
    err = float(np.abs(g - reference).max())
    print(f"{method:<18}{str(cost):>10}   {err:>22.2e}   {KIND.get(method, 'custom'):<16}")

print(
    "\nthe four exact methods are four independent routes to one number, so"
    " disagreement between them is a bug, not a tolerance argument."
    "\nhadamard gets there in half parameter-shift's circuits, paying one"
    " ancilla that must reach every wire the generator touches."
)

print("\nadjoint is the default because its cost does not grow with the parameter count:")
for layers in (2, 6, 12):
    a = qk.hardware_efficient(5, layers)
    print(
        f"  P={a.n_params:>3}  adjoint 1 pass   parameter-shift "
        f"{qk.gradient_cost(a.build(), 'parameter-shift'):>4} circuits"
    )


# --------------------------------------------------------------------------- #
header("4. Encoding: every way data gets into a circuit")
# --------------------------------------------------------------------------- #
print(f"basis_encode([1,0,1])     -> {qk.run_counts(qk.basis_encode([1, 0, 1]), 512, seed=0)}")
amp = qk.statevector(qk.amplitude_encode([1, 2, 3, 4]))
target = np.array([1, 2, 3, 4]) / np.linalg.norm([1, 2, 3, 4])
print(f"amplitude_encode(1,2,3,4) -> {np.round(amp.real, 4)}")
print(f"                   target -> {np.round(target, 4)}")

fm = qk.ZZFeatureMap(2, reps=2)
print(f"\nZZFeatureMap(2, reps=2): {fm.resources()}")


# --------------------------------------------------------------------------- #
header("5. Re-uploading is a pattern, not a structure")
# --------------------------------------------------------------------------- #
for label, model in [
    ("default (S W)", qk.reupload(qk.AngleFeatureMap(2), n_layers=3)),
    ("order='WS'", qk.reupload(qk.AngleFeatureMap(2), n_layers=3, order="WS")),
    ("shared weights", qk.reupload(qk.AngleFeatureMap(2), n_layers=3, share_weights=True)),
    ("ZZ feature map", qk.reupload(qk.ZZFeatureMap(2, reps=1), n_layers=2)),
]:
    print(
        f"  {label:<18} inputs {model.n_inputs}  "
        f"weights {model.n_weights:>3}  depth {model.resources()['depth']:>3}"
    )

print("\nL uploads reach frequencies 0..L — measured, not asserted:")
for layers in (1, 2, 3):
    m = qk.reupload(qk.AngleFeatureMap(1, entangle=False), n_layers=layers, entangler=None)
    w = m.init(seed=0)
    bound = m.build()

    def f(x, bound=bound, m=m, w=w):
        return qk.expval(bound, qk.Z(0), theta=np.concatenate([m.angles([x]), w]))

    present = sorted(qk.fourier.spectrum(f, layers + 3))
    print(f"  L={layers}: {present}")


# --------------------------------------------------------------------------- #
header("6. Inventing an ansatz costs one line")
# --------------------------------------------------------------------------- #
brick = qk.Ansatz(
    4, qk.repeat(2, qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "alternating")), "brick_wall"
)
print(qk.draw(brick.build()))
print(f"\n{brick}\n")
print(qk.metrics.AnsatzReport(brick, n_samples=300))

print("\ncompare candidates:")
print(f"{'ansatz':<24}{'params':>7}{'depth':>7}{'2q':>5}{'expressibility':>16}{'entangling':>12}")
for name in ("hardware_efficient", "strongly_entangling", "tree_tensor_network", "mps"):
    r = qk.metrics.AnsatzReport(qk.get_ansatz(name, n_qubits=4), n_samples=250).results
    print(
        f"{name:<24}{r['n_params']:>7}{r['depth']:>7}{r['n_2q']:>5}"
        f"{r['expressibility']:>16.4f}{r['entangling_capability']:>12.4f}"
    )


# --------------------------------------------------------------------------- #
header("7. Quantum kernels")
# --------------------------------------------------------------------------- #
X, y = qk.datasets.ad_hoc_data(n_samples=40, n_features=2, gap=0.4, seed=0)
X_train, X_test, y_train, y_test = qk.datasets.train_test_split(X, y, 0.3, seed=0)

kernel = qk.QuantumKernel(qk.ZZFeatureMap(2, reps=2))
K = kernel(X_train)
print(
    f"Gram matrix {K.shape}: symmetric={np.allclose(K, K.T)} unit-diagonal="
    f"{np.allclose(np.diag(K), 1)} PSD={qk.is_psd(K)}"
)
print(
    f"target alignment {qk.target_alignment(K, y_train):+.4f} · circuits run {kernel.n_evaluations}"
)

if HAS_SKLEARN:
    clf = qk.QSVC(qk.ZZFeatureMap(2, reps=2)).fit(X_train, y_train)
    tr, te = clf.score(X_train, y_train), clf.score(X_test, y_test)
    print(f"\nQSVC on ad_hoc_data: train {tr:.0%}  test {te:.0%}")
    from sklearn.svm import SVC

    for k in ("rbf", "linear"):
        s = SVC(kernel=k).fit(X_train, y_train)
        print(
            f"classical {k:<7}         train {s.score(X_train, y_train):.0%}  "
            f"test {s.score(X_test, y_test):.0%}"
        )
    print("(the dataset is built to be separable by a quantum kernel and not a classical one)")

print("\nglobal fidelity kernels concentrate as the register widens; projected ones resist it:")
rng = np.random.default_rng(0)
for n in (2, 4, 6, 8):
    Xn = rng.uniform(0, np.pi, (8, n))
    f = qk.ZZFeatureMap(n, reps=2)
    off = lambda M: float(M[~np.eye(len(M), dtype=bool)].std())  # noqa: E731
    print(
        f"  n={n}: fidelity spread {off(qk.QuantumKernel(f)(Xn)):.5f}   "
        f"projected {off(qk.projected_kernel_matrix(f, Xn)):.5f}"
    )


# --------------------------------------------------------------------------- #
header("8. Optimisers built for circuits")
# --------------------------------------------------------------------------- #
a = qk.hardware_efficient(3, 2)
spec_o, start = a.build(), a.init("uniform", seed=1)
cost = qk.Z(0) + qk.Z(1) + qk.Z(2)  # minimum is -3


def loss(t: np.ndarray) -> float:
    return qk.expval(spec_o, cost, theta=t)


_, roto = qk.minimize_rotosolve(loss, start, n_sweeps=8)
_, qng = qk.minimize_qng(spec_o, start, cost, n_steps=25, lr=0.15)
plain = start.copy()
for _ in range(25):
    plain = plain - 0.15 * qk.grad(spec_o, plain, cost)
_, spsa = qk.minimize_spsa(loss, start, n_iterations=200, seed=0)

print(f"start                       {roto[0]:+.5f}")
print(f"Rotosolve  (8 sweeps)       {roto[-1]:+.5f}   no learning rate at all")
print(f"QNG        (25 steps)       {qng[-1]:+.5f}   follows the Fubini-Study geometry")
print(f"plain GD   (25 steps)       {loss(plain):+.5f}")
print(f"SPSA       (200 iters)      {spsa[-1]:+.5f}   2 circuits per step, any P")
print("theoretical minimum         -3.00000")


# --------------------------------------------------------------------------- #
header("9. Torch: the two-line path, and the layer underneath")
# --------------------------------------------------------------------------- #
if not HAS_TORCH:
    print("(torch not installed — skipping)")
else:
    import torch
    from torch import nn

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    Xc = rng.normal(size=(60, 4))
    yc = (Xc[:, 0] * Xc[:, 1] > 0).astype(int)  # XOR-like, not linearly separable

    model = qk.VQC(n_features=4, n_classes=2, seed=0).fit(Xc, yc, epochs=25)
    print(f"qk.VQC(4, 2).fit(X, y)  ->  accuracy {model.score(Xc, yc):.1%}")
    print(f"  loss {model.history_[0]:.4f} -> {model.history_[-1]:.4f}")

    layer = qk.QuantumLayer(
        qk.ZZFeatureMap(3, reps=1), qk.hardware_efficient(3, 2), [qk.Z(0), qk.Z(1)], init_seed=0
    ).double()
    net = nn.Sequential(nn.Linear(6, 3), nn.Tanh(), layer, nn.Linear(2, 2)).double()
    xb = torch.randn(8, 6, dtype=torch.float64)
    nn.CrossEntropyLoss()(net(xb), torch.randint(0, 2, (8,))).backward()
    print("\nnn.Sequential(Linear, QuantumLayer, Linear):")
    print(f"  pre-net gradient norm {float(net[0].weight.grad.norm()):.6f}  <- it trains")
    print(f"  quantum gradient norm {float(layer.theta.grad.norm()):.6f}")


# --------------------------------------------------------------------------- #
header("10. Generative: learning a distribution")
# --------------------------------------------------------------------------- #
target = qk.datasets.bars_and_stripes(2)
print(f"bars-and-stripes: {len(target)} of {2**4} bitstrings are valid")
qcbm = qk.generative.QCBM(4, n_layers=3, seed=0)
before = qcbm.exact_distance(target)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    qcbm.fit(target, n_iterations=80, n_samples=256, seed=0)
print(f"QCBM total-variation distance to target: {before:.4f} -> {qcbm.exact_distance(target):.4f}")

memory = qk.generative.QuantumHopfield().store(
    {"A": [1, 0, 0, 1], "B": [1, 1, 0, 0], "C": [0, 1, 1, 0]}
)
cue = [1.0, 0.2, 0.0, 0.9]
print(f"\nHopfield recall of {cue}: {memory.recall(cue)}")
print(f"  overlaps { {k: round(v, 3) for k, v in memory.overlaps(cue).items()} }")

print("\n" + "=" * 74)
print("Done. Every number above was computed, not quoted.")
print("=" * 74)
