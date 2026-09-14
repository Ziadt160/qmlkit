"""End-to-end use case: does the analysis actually improve a model, or only describe it?

The test that matters is not "does the tool emit a finding" -- it is "does following
the finding's advice make the model better". Every section below measures before and
after, so a wrong recommendation shows up as a number that did not move.
"""

from __future__ import annotations

import numpy as np
import torch

import qmlkit as qk
from qmlkit.ansatz import Ansatz, EncodingLayer, EntanglerLayer, RotationLayer, repeat
from qmlkit.datasets import make_moons, train_test_split
from qmlkit.metrics import effective_dimension, fisher_information
from qmlkit.nn.models import VQC

RULE = "=" * 78
N_QUBITS = 4


def banner(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def variational(n_layers: int, rotations: tuple[str, ...] = ("ry", "rz")) -> Ansatz:
    """A plain trainable block -- VQC's own feature map does the encoding in front."""
    block = RotationLayer(rotations) + EntanglerLayer("cz", "ring")
    return Ansatz(N_QUBITS, repeat(n_layers, block), name=f"{'+'.join(rotations)}_{n_layers}L")


def accuracy(ansatz: Ansatz, observables, Xtr, ytr, Xte, yte, epochs: int = 40, seed: int = 0):
    """Train the library's own VQC on this ansatz and score it. The integration test."""
    torch.manual_seed(seed)
    model = VQC(
        n_features=Xtr.shape[1],
        n_qubits=N_QUBITS,
        feature_map=qk.AngleFeatureMap(N_QUBITS, rotation="ry"),
        ansatz=ansatz,
        observables=list(observables),
        seed=seed,
    )
    model.fit(Xtr, ytr, epochs=epochs, lr=0.1)
    return float(np.mean(model.predict(Xte) == yte))


X, y = make_moons(n_samples=160, noise=0.15, seed=0)
X = np.hstack([X, X])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, seed=0)
print(
    f"moons: {Xtr.shape[0]} train / {Xte.shape[0]} test, {X.shape[1]} features, {N_QUBITS} qubits"
)

# --------------------------------------------------------------------------- #
banner("STEP 1  A deep ansatz with a single-qubit readout -- what does landscape() say?")
deep = variational(6)
report = qk.landscape(deep, qk.Z(0), n_samples=80, minima=False)
print(report.gradients)
print()
for f in report.findings:
    print(f"  [{f.severity}] {f.code}")
    print(f"      {f.message[:200]}")
    print(f"      FIX: {f.fix[:140]}")
print("\n  -- the existing diagnose() on the same ansatz, for comparison --")
found = list(qk.diagnose(deep))
print("  " + ("\n  ".join(f"[{f.severity}] {f.code}" for f in found) if found else "(nothing)"))

# --------------------------------------------------------------------------- #
banner("STEP 2  Does the recommended fix actually raise accuracy?")
one_wire = [qk.Z(0)]
all_wires = [qk.Z(i) for i in range(N_QUBITS)]
print(f"{'configuration':<34} {'silent':>9} {'typical var':>13} {'test acc':>9}")
for label, ans, obs in [
    ("6 layers, readout Z(0)", deep, one_wire),
    ("6 layers, readout every wire", deep, all_wires),
    ("3 layers, readout every wire", variational(3), all_wires),
]:
    stats = qk.gradient_stats(ans, sum(obs), n_samples=80)
    acc = accuracy(ans, obs, Xtr, ytr, Xte, yte)
    print(
        f"{label:<34} {stats.silent.size:>4}/{stats.n_params:<4} "
        f"{stats.typical_variance:>13.3e} {acc:>8.1%}"
    )

# --------------------------------------------------------------------------- #
banner("STEP 3  Sizing the ansatz: does the QFIM rank predict where accuracy plateaus?")
scan = qk.overparametrisation(variational, range(1, 7))
print(scan)
print()
print(f"{'layers':>7} {'params':>7} {'rank':>6} {'verdict':>9} {'test acc':>9}")
for n_layers, params, rank in zip(scan.layers, scan.n_params, scan.ranks, strict=True):
    acc = accuracy(variational(n_layers), all_wires, Xtr, ytr, Xte, yte)
    print(f"{n_layers:>7} {params:>7} {rank:>6} {scan.verdict(params):>9} {acc:>8.1%}")

# --------------------------------------------------------------------------- #
banner("STEP 4  Local minima at a real data point (re-uploading ansatz)")
fmap = qk.AngleFeatureMap(N_QUBITS, rotation="ry")
block = EncodingLayer(fmap) + RotationLayer(("ry", "rz")) + EntanglerLayer("cz", "ring")
reupload = Ansatz(N_QUBITS, repeat(3, block), name="reupload_3L")
print(f"{reupload.name}: {reupload.n_inputs} input slots, {reupload.n_weights} weights")
try:
    qk.minima_scan(reupload, sum(all_wires), n_starts=4, n_steps=50)
except ValueError as exc:
    print(f"\n  without a data point it refuses, correctly:\n    {str(exc)[:150]}...")
print()
print(qk.minima_scan(reupload, sum(all_wires), x=Xtr[0], n_starts=8, n_steps=1800, seed=0))

# --------------------------------------------------------------------------- #
banner("STEP 5  The encoding decision, priced")
print(qk.loading_cost(X.shape[1], "angle"))
print()
print(qk.loading_cost(1024, "amplitude"))
print()
print(qk.qram_cost(1024))

# --------------------------------------------------------------------------- #
banner("STEP 6  Fisher information, now that it reads the data")

weights = reupload.init("uniform", seed=0)
fim = fisher_information(reupload, Xtr, weights, sum(all_wires))
one_row = fisher_information(reupload, Xtr[:1], weights, sum(all_wires))
print(f"  weights                      {reupload.n_weights}")
print(f"  FIM rank from 1 row          {np.linalg.matrix_rank(one_row)}")
print(f"  FIM rank from {len(Xtr)} rows        {np.linalg.matrix_rank(fim)}")
print(f"  effective dimension          {effective_dimension(fim):.2f} of {reupload.n_weights}")
print(
    f"  symmetric / PSD              {np.allclose(fim, fim.T)} / "
    f"{bool(np.all(np.linalg.eigvalsh(fim) > -1e-9))}"
)
