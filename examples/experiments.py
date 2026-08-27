"""Three real experiments, end to end.

    pip install "qmlkit[torch,sklearn]"
    python examples/experiments.py

Not toy demonstrations built to flatter the library — three problems people actually
run, with an independent check on each answer and a classical reference point beside
it, because an accuracy reported without context is not a result.

1. **H2 ground-state energy.** The Hamiltonian is computed here from STO-3G
   integrals, not imported from a chemistry package, and the answer is checked
   against dense diagonalisation at every bond length.
2. **QCNN on MNIST.** Real MNIST, 0 vs 1, PCA to 8 qubits. Three convolution
   filters, to show what the filter choice is actually worth.
3. **VQC on breast cancer.** 30 clinical features, PCA to 4, with the metric that
   matters clinically — recall on the malignant class — reported separately from
   accuracy.

Runtime is roughly 20 minutes; the QCNN is most of it.
"""

from __future__ import annotations

import time

import qmlkit as qk


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
header("1. H2 ground-state energy")
# --------------------------------------------------------------------------- #
from qmlkit.algorithms import (  # noqa: E402
    CHEMICAL_ACCURACY,
    VQE,
    AdaptVQE,
    chemistry_operator_pool,
    default_operator_pool,
    exact_ground_energy,
    h2_hamiltonian,
)
from qmlkit.algorithms.adapt import _commutator  # noqa: E402

hamiltonian, info = h2_hamiltonian(0.735)
print(f"  built from STO-3G integrals: {info['n_terms']} Pauli terms on {info['n_qubits']} qubits")
print(f"  nuclear repulsion {info['nuclear_repulsion']:.6f} Ha")
print(f"  Hartree-Fock reference |{''.join(map(str, info['hartree_fock_occupation']))}>")
print(f"  chemical accuracy = 1 kcal/mol = {CHEMICAL_ACCURACY * 1000:.3f} mHa")

exact = exact_ground_energy(hamiltonian, 4)
print(f"\n  exact (dense diagonalisation) {exact:.8f} Ha")
print("  literature FCI/STO-3G          -1.1373   Ha")


def hartree_fock(inner: qk.Ansatz) -> qk.Ansatz:
    """Prepend the HF occupation, since an Ansatz otherwise starts from |0000>."""

    def build(circuit, context):  # type: ignore[no-untyped-def]
        for wire, occupied in enumerate(info["hartree_fock_occupation"]):
            if occupied:
                circuit.x(wire)
        inner.block.emit(circuit, context)

    return qk.Ansatz(4, qk.Custom(build, "hf"), f"hf_{inner.name}")


print(f"\n  {'ansatz':<34}{'P':>4}{'E (Ha)':>14}{'error (mHa)':>13}  chemical?")
for label, ansatz in [
    ("hardware_efficient(4, 2) + HF", hartree_fock(qk.hardware_efficient(4, 2))),
    ("hardware_efficient(4, 4) + HF", hartree_fock(qk.hardware_efficient(4, 4))),
    (
        "strongly_entangling(4, 3) + HF",
        hartree_fock(qk.get_ansatz("strongly_entangling", n_qubits=4, n_layers=3)),
    ),
]:
    result = VQE(hamiltonian, ansatz=ansatz).run(seed=0, n_sweeps=80)
    error = (result.energy - exact) * 1000
    verdict = "yes" if abs(error) < CHEMICAL_ACCURACY * 1000 else "no"
    print(f"  {label:<34}{ansatz.n_params:>4}{result.energy:>14.8f}{error:>13.5f}  {verdict}")

# The generic ADAPT pool is *wrong here*, and it is worth seeing why rather than
# being told: a molecular Hamiltonian conserves particle number, so a generator that
# does not has exactly zero gradient at the Hartree-Fock state.
reference = qk.QCircuit(4)
reference.x(0).x(1)
hf_state = reference.to_spec()
generic = [
    abs(qk.expval(hf_state, _commutator(hamiltonian, op)))
    for op in default_operator_pool(4)
    if _commutator(hamiltonian, op).terms
]
print(f"\n  largest gradient in the generic pool:   {max(generic, default=0.0):.2e}  (all zero)")

pool = chemistry_operator_pool(4)
best = max(abs(qk.expval(hf_state, _commutator(hamiltonian, op))) for op in pool)
print(f"  largest gradient in the chemistry pool: {best:.4f}")

adapt = AdaptVQE(hamiltonian, 4, pool=pool, reference=[0, 1]).run(
    max_operators=3, gradient_tol=1e-4, n_steps=250, lr=0.3
)
print(f"\n  ADAPT-VQE selected {adapt.n_operators} operator: {[str(o) for o in adapt.operators]}")
print(f"  E = {adapt.energy:.8f} Ha, error {(adapt.energy - exact) * 1000:.5f} mHa")

print(
    f"\n  dissociation curve\n  {'R (A)':>7}{'exact':>14}{'ADAPT':>14}{'error (mHa)':>13}{'ops':>5}"
)
for r in (0.4, 0.5, 0.6, 0.735, 0.9, 1.1, 1.4, 1.8, 2.4, 3.0):
    h_r, _ = h2_hamiltonian(r)
    exact_r = exact_ground_energy(h_r, 4)
    got = AdaptVQE(h_r, 4, pool=pool, reference=[0, 1]).run(
        max_operators=3, gradient_tol=1e-4, n_steps=250, lr=0.3
    )
    print(
        f"  {r:>7.3f}{exact_r:>14.8f}{got.energy:>14.8f}"
        f"{(got.energy - exact_r) * 1000:>13.5f}{got.n_operators:>5}"
    )


# --------------------------------------------------------------------------- #
header("2. QCNN on MNIST (0 vs 1)")
# --------------------------------------------------------------------------- #
import torch  # noqa: E402
from sklearn.datasets import fetch_openml, load_breast_cancer  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from torch import nn  # noqa: E402

from qmlkit.nn.advanced import QCNNLayer  # noqa: E402

N_QUBITS = 8
pixels, labels = fetch_openml(
    "mnist_784", version=1, return_X_y=True, as_frame=False, parser="liac-arff"
)
labels = labels.astype(int)
keep = (labels == 0) | (labels == 1)
pixels, labels = pixels[keep][:1200], labels[keep][:1200]
x_train, x_test, y_train, y_test = train_test_split(
    pixels, labels, test_size=0.3, random_state=0, stratify=labels
)
pca = PCA(n_components=N_QUBITS, random_state=0).fit(x_train)
z_train, z_test = pca.transform(x_train), pca.transform(x_test)
scaler = qk.AngleScaler().fit(z_train)
z_train, z_test = scaler.transform(z_train), scaler.transform(z_test)
print(
    f"  {len(pixels)} images, 784 pixels -> {N_QUBITS} PCA components "
    f"({pca.explained_variance_ratio_.sum():.1%} of the variance)"
)
print(f"  train {z_train.shape}, test {z_test.shape}")

N_TRAIN, EPOCHS = 400, 60
inputs = torch.tensor(z_train[:N_TRAIN])
targets = torch.tensor(y_train[:N_TRAIN], dtype=torch.long)
held_out = torch.tensor(z_test)


def train_qcnn(filter_name: str, epochs: int = EPOCHS, seed: int = 0):
    torch.manual_seed(seed)
    layer = QCNNLayer(N_QUBITS, filter=filter_name, tie_weights=True, init_seed=seed).double()
    model = nn.Sequential(layer, nn.Linear(1, 2).double())
    optimiser = torch.optim.Adam(model.parameters(), lr=0.15)
    criterion = nn.CrossEntropyLoss()
    started = time.perf_counter()
    history = []
    for _ in range(epochs):
        optimiser.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimiser.step()
        history.append(float(loss.detach()))
    with torch.no_grad():
        train_accuracy = (model(inputs).argmax(1).numpy() == y_train[:N_TRAIN]).mean()
        test_accuracy = (model(held_out).argmax(1).numpy() == y_test).mean()
    return layer, history, train_accuracy, test_accuracy, time.perf_counter() - started


print(f"\n  {'filter':<9}{'weights':>9}{'depth':>7}{'loss':>19}{'train':>8}{'test':>8}{'time':>8}")
for name in ("ry_cx", "real", "su4"):
    layer, history, train_accuracy, test_accuracy, seconds = train_qcnn(name)
    depth = qk.qcnn_ansatz(N_QUBITS, filter=name).resources()["depth"]
    print(
        f"  {name:<9}{layer.n_weights:>9}{depth:>7}"
        f"{history[0]:>10.4f}->{history[-1]:.4f}{train_accuracy:>8.1%}{test_accuracy:>8.1%}{seconds:>7.0f}s"
    )

baseline = LogisticRegression(max_iter=2000).fit(z_train[:N_TRAIN], y_train[:N_TRAIN])
context = baseline.score(z_test, y_test)
print(f"\n  for context, logistic regression on the same 8 features: {context:.1%}")


# --------------------------------------------------------------------------- #
header("3. VQC on breast cancer")
# --------------------------------------------------------------------------- #
data = load_breast_cancer()
print(
    f"  {data.data.shape[0]} samples, {data.data.shape[1]} clinical features, "
    f"classes {list(data.target_names)}"
)
bx_train, bx_test, by_train, by_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
)
standard = StandardScaler().fit(bx_train)

for n_qubits in (4, 6):
    reducer = PCA(n_components=n_qubits, random_state=0).fit(standard.transform(bx_train))
    f_train = reducer.transform(standard.transform(bx_train))
    f_test = reducer.transform(standard.transform(bx_test))
    angles = qk.AngleScaler().fit(f_train)
    f_train, f_test = angles.transform(f_train), angles.transform(f_test)

    started = time.perf_counter()
    model = qk.VQC(n_features=n_qubits, n_classes=2, n_layers=3, seed=0).fit(
        f_train, by_train, epochs=30
    )
    predictions = model.predict(f_test)
    caught = int(((predictions == 0) & (by_test == 0)).sum())
    missed = int(((predictions == 1) & (by_test == 0)).sum())
    reference = LogisticRegression(max_iter=2000).fit(f_train, by_train)

    print(
        f"\n  VQC on {n_qubits} qubits | PCA keeps {reducer.explained_variance_ratio_.sum():.1%} "
        f"| {time.perf_counter() - started:.0f}s"
    )
    print(f"    loss {model.history_[0]:.4f} -> {model.history_[-1]:.4f}")
    print(
        f"    train {model.score(f_train, by_train):.1%}   test {model.score(f_test, by_test):.1%}"
    )
    print(
        f"    malignant recall {caught}/{caught + missed} = {caught / (caught + missed):.1%}"
        "   <- the error that matters clinically"
    )
    print(f"    logistic regression, same features: {reference.score(f_test, by_test):.1%}")

print(f"\n{'=' * 78}\nEvery number above was computed by this script.\n{'=' * 78}")
