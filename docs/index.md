# qmlkit

A quantum machine learning library where **a circuit is data, not a backend object**.

That one decision is why a single gradient implementation serves five backends, why
inventing an ansatz takes one line and inherits correct gradients for free, and why
weight tying is expressible at all. Everything downstream — differentiation, resource
counting, drawing, translation to SpinQit or Qiskit or Cirq — reads the same structure.

```python
import numpy as np
import qmlkit as qk

ansatz = qk.hardware_efficient(3, n_layers=2)
theta = ansatz.init(seed=0)
spec = ansatz.build()
observable = qk.Z(0) + 0.5 * qk.ZZ(0, 2)

print(f"<O>       = {qk.expval(spec, observable, theta=theta):+.6f}")
print(f"gradient  = {np.round(qk.grad(spec, theta, observable)[:4], 4)} ...")
print(f"cost      = 1 pass (adjoint) vs {qk.gradient_cost(spec, 'parameter-shift')} circuits (parameter-shift)")
```

## Where to start

<div class="grid cards" markdown>

- :material-play: **[Tutorials](tutorials/index.md)**

    Eight pages, start to finish. Every snippet is executed by the test suite, so
    none of them can quietly stop working.

- :material-book-open-variant: **[Guides](guides/index.md)**

    Why parameter-shift is subtler than it looks, which gradient to reach for, and
    how to add your own gate, ansatz, backend or estimator.

- :material-api: **[Reference](reference/index.md)**

    Generated from the docstrings, so it cannot drift from the code.

- :material-check-decagram: **[Validation](about/validation.md)**

    301 cross-validation cases against PennyLane, and the four genuine convention
    differences that surfaced.

</div>

## What is actually here

| | |
|---|---|
| **Backends** | NumPy (reference), SpinQit, Qiskit, Cirq, Torch — one circuit, one answer, checked against each other |
| **Gradients** | adjoint, backprop, Hadamard-test, parameter-shift, SPSA, finite differences, behind one `grad()` |
| **Encoding** | angle, amplitude, basis, Hamiltonian, and Pauli feature maps, with input gradients |
| **Ansätze** | a composable block vocabulary plus ten templates written in it |
| **Kernels** | three overlap estimators, PSD repair, `QSVC`/`QSVR`, trainable and projected kernels |
| **PyTorch** | `QuantumLayer`, `VQC`, `VQRegressor`, QCNN/QLSTM/MPS, dressed networks |
| **Analysis** | expressibility, Meyer–Wallach entanglement, barren-plateau scans, Fourier spectra, Fubini–Study geometry |

## Three layers, and you pick where to stand

Nothing at a higher layer hides a lower one.

```python
# docs: requires torch
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)
X = rng.normal(size=(40, 4))
y = (X[:, 0] * X[:, 1] > 0).astype(int)

# 1. a ready-made model
model = qk.VQC(n_features=4, n_classes=2, seed=0).fit(X, y, epochs=5)

# 2. a torch layer, in any nn.Module you like
layer = qk.QuantumLayer(qk.ZZFeatureMap(3), qk.hardware_efficient(3, 2), [qk.Z(0)])

# 3. circuits and gradients directly
g = qk.grad(qk.hardware_efficient(3, 2).build(), qk.hardware_efficient(3, 2).init(seed=0))
```

## Scope

**Simulator-only for the whole 0.x line.** That is a design constraint, not a
missing feature: it makes `adjoint` the right default gradient, makes shot noise
opt-in rather than unavoidable, and keeps the focus on properties of the *model* —
trainability, expressibility, concentration — rather than properties of a device.
Parameter-shift is still first-class, because it is the rule that stays valid when
you do move to hardware.

Apache-2.0. [Source on GitHub](https://github.com/Ziadt160/qmlkit).
