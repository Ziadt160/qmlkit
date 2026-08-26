# 6. Quantum kernels

The other way to use a feature map. No variational parameters, no training loop, no
barren plateaus — just an inner product between encoded states, handed to a classical
solver that is convex and has a unique optimum.

The price is that you pay it in circuits: `m(m−1)/2` of them for an `m`-sample
training set, and that quadratic is what limits the approach long before qubit counts
do.

## The kernel is an overlap

```python
import numpy as np
import qmlkit as qk

fmap = qk.ZZFeatureMap(2, reps=2)
x1, x2 = np.array([0.4, 1.3]), np.array([1.9, 0.6])

print(f"k(x, x)  = {qk.fidelity_kernel(fmap, x1, x1):.10f}")
print(f"k(x, x') = {qk.fidelity_kernel(fmap, x1, x2):.10f}")
```

```text
k(x, x)  = 1.0000000000
k(x, x') = 0.7055431570
```

`fidelity_kernel` is compute-uncompute: run `U(x)`, then `U(x')†`, and read the
probability of measuring all zeros. That probability *is* `|⟨φ(x')|φ(x)⟩|²`. It falls
straight out of `spec.adjoint()` and needs no ancilla.

Three estimators, same quantity:

```python
import numpy as np
import qmlkit as qk

fmap = qk.ZZFeatureMap(2, reps=2)
x1, x2 = np.array([0.4, 1.3]), np.array([1.9, 0.6])

print(f"fidelity (compute-uncompute) {qk.fidelity_kernel(fmap, x1, x2):+.10f}")
print(f"swap test                    {qk.swap_test_kernel(fmap, x1, x2):+.10f}")
print(f"hadamard test (signed)       {qk.hadamard_test(fmap, x1, x2):+.10f}")
```

```text
fidelity (compute-uncompute) +0.7055431570
swap test                    +0.7055431570
hadamard test (signed)       -0.2029143060
```

The Hadamard test is the odd one out on purpose: it estimates the **signed** inner
product `Re⟨φ(x')|φ(x)⟩`, not its square. Use it when the sign carries information;
use `fidelity_kernel` otherwise, since it needs the fewest qubits and no ancilla.

## A Gram matrix

```python
import numpy as np
import qmlkit as qk

X, y = qk.datasets.ad_hoc_data(n_samples=40, n_features=2, gap=0.4, seed=0)
X_train, X_test, y_train, y_test = qk.datasets.train_test_split(X, y, 0.3, seed=0)

kernel = qk.QuantumKernel(qk.ZZFeatureMap(2, reps=2))
K = kernel(X_train)

print(f"shape {K.shape}  symmetric {np.allclose(K, K.T)}  unit diagonal {np.allclose(np.diag(K), 1)}")
print(f"positive semi-definite: {qk.is_psd(K)}")
print(f"circuits run: {kernel.n_evaluations}  (m(m-1)/2 = {len(X_train) * (len(X_train) - 1) // 2})")
print(f"target alignment: {qk.target_alignment(K, y_train):+.4f}")
```

```text
shape (28, 28)  symmetric True  unit diagonal True
positive semi-definite: True
circuits run: 378  (m(m-1)/2 = 378)
target alignment: +0.3360
```

`QuantumKernel` caches symmetrically, so `k(a,b)` and `k(b,a)` share one entry and a
training Gram matrix costs exactly `m(m−1)/2` circuits — not `m²`.

**Target alignment** measures how well the kernel's geometry matches the labels,
before fitting anything. It is the cheapest signal you have about whether a feature
map suits a dataset.

## Classification

```python
# docs: requires sklearn
import qmlkit as qk
from sklearn.svm import SVC

X, y = qk.datasets.ad_hoc_data(n_samples=40, n_features=2, gap=0.4, seed=0)
X_train, X_test, y_train, y_test = qk.datasets.train_test_split(X, y, 0.3, seed=0)

clf = qk.QSVC(qk.ZZFeatureMap(2, reps=2)).fit(X_train, y_train)
print(f"QSVC        train {clf.score(X_train, y_train):.0%}  test {clf.score(X_test, y_test):.0%}")
for kind in ("rbf", "linear"):
    s = SVC(kernel=kind).fit(X_train, y_train)
    print(f"SVC {kind:<7} train {s.score(X_train, y_train):.0%}  test {s.score(X_test, y_test):.0%}")
```

```text
QSVC        train 100%  test 100%
SVC rbf     train 68%  test 67%
SVC linear  train 57%  test 67%
```

!!! warning "Read that honestly"
    `ad_hoc_data` is **constructed** to be separable by a ZZ kernel and not by a
    classical one. It demonstrates that the machinery works; it is not evidence of
    quantum advantage on real data. A dataset built to favour your method is a
    sanity check, not a result.

## Shot noise breaks positive semi-definiteness

A Gram matrix estimated from finite samples can leave the PSD cone, and an SVM solver
handed a non-PSD kernel does not necessarily fail loudly — it can just return
something wrong.

```python
import numpy as np
import qmlkit as qk

X, y = qk.datasets.ad_hoc_data(n_samples=40, n_features=2, gap=0.4, seed=0)
X_train, *_ = qk.datasets.train_test_split(X, y, 0.3, seed=0)

K = qk.QuantumKernel(qk.ZZFeatureMap(2, reps=2), shots=512, seed=0)(X_train[:10])
print(f"sampled Gram is PSD: {qk.is_psd(K)}   min eigenvalue {qk.min_eigenvalue(K):+.5f}")

for name, repair in (
    ("threshold", qk.threshold_matrix),
    ("displace", qk.displace_matrix),
    ("flip", qk.flip_matrix),
):
    R = repair(K)
    print(f"  {name:<10} PSD {qk.is_psd(R)}   min eig {qk.min_eigenvalue(R):+.5f}"
          f"   ‖K−R‖_F {np.linalg.norm(K - R):.5f}")
```

```text
sampled Gram is PSD: False   min eigenvalue -0.06497
  threshold  PSD True   min eig +0.00000   ‖K−R‖_F 0.06497
  displace   PSD True   min eig +0.00000   ‖K−R‖_F 0.20546
  flip       PSD True   min eig +0.06497   ‖K−R‖_F 0.12994
```

`threshold` clips negative eigenvalues to zero and is the closest PSD matrix in
Frobenius norm — usually the right default. `displace` shifts the whole spectrum,
which preserves eigenvectors but distorts more. `flip` takes absolute values, keeping
the spectral magnitude at the cost of moving further.

## Concentration is the real limit

As the register widens, fidelities between distinct points all collapse toward the
same tiny number. The kernel stops distinguishing anything, and no amount of shots
recovers it — the signal is gone, not merely noisy.

```python
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)

def spread(M):
    return float(M[~np.eye(len(M), dtype=bool)].std())

for n in (2, 4, 6, 8):
    Xn = rng.uniform(0, np.pi, (8, n))
    fmap = qk.ZZFeatureMap(n, reps=2)
    print(f"n={n}: fidelity spread {spread(qk.QuantumKernel(fmap)(Xn)):.5f}"
          f"   projected {spread(qk.projected_kernel_matrix(fmap, Xn)):.5f}")
```

```text
n=2: fidelity spread 0.17154   projected 0.14718
n=4: fidelity spread 0.07615   projected 0.10260
n=6: fidelity spread 0.02117   projected 0.06558
n=8: fidelity spread 0.01654   projected 0.06393
```

The fidelity kernel's off-diagonal spread falls by a factor of ten from 2 to 8
qubits. The **projected** kernel — which compares one-qubit reduced density matrices
instead of the global overlap — falls by only about half, and by 6 qubits it carries
three times more signal than the fidelity kernel does. It is not immune to
concentration; it just degrades far more slowly. `concentration_report` measures this
for your own feature map.

## Training the embedding

You can optimise the *feature map* before fitting any classifier, by maximising
target alignment:

```python
import numpy as np
import qmlkit as qk

X, y = qk.datasets.ad_hoc_data(n_samples=24, n_features=2, gap=0.4, seed=0)

def factory(w):
    """A ZZ map whose data scaling is the thing being learned."""
    return qk.PauliFeatureMap(
        2, paulis=("Z", "ZZ"), reps=2,
        data_map=lambda x, idx: float(np.prod([x[i] for i in idx])) * float(w[0]),
    )

trainable = qk.TrainableKernel(factory, n_params=1)
trainable.fit(X, y, n_iterations=60, theta0=np.array([0.4]), seed=0)
print(f"alignment {trainable.history_[0]:+.4f} -> {trainable.history_[-1]:+.4f}")
print(f"scale     0.4 -> {trainable.params_[0]:.3f}")
```

```text
alignment +0.0460 -> +0.1406
scale     0.4 -> 0.549
```

Three times the alignment, before any classifier is fitted. Two honest caveats: the
landscape is bumpy — a scale of 1.0 scores 0.3664 on this data, so SPSA has found a
local optimum, not the best one — and `fit` returns the **final** iterate rather than
the best seen, so a short run can end below where it started. `history_` is the full
alignment trajectory, and its last entry always corresponds to `params_`.

---

**Next:** [Re-uploading and Fourier](07-reuploading.md) — why depth buys you
frequencies, measured rather than asserted.
