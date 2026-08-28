# Study 2 — Is a quantum kernel worth trying at all?

A quantum kernel is the most appealing thing in this field: no training loop, no
barren plateau, a Gram matrix you hand to any kernel method. It is also the easiest
place to spend a month on a matrix that could never have separated anything.

Two numbers decide it, and both are available **before** the kernel is fitted to
anything.

## The data and the bar

```python
import numpy as np
import qmlkit as qk

X, y = qk.datasets.make_circles(n_samples=80, seed=0)
table = qk.baseline(X, y, cv=3, seed=0,
                    include=["majority", "rbf-kernel-ridge", "nearest-centroid"])
print(table)
```

`rbf-kernel-ridge` is the row that matters. It is the *identical algorithm* to a
quantum kernel method — a closed-form kernel ridge solve — differing only in which
kernel fills the Gram matrix. Any gap between it and a quantum kernel is attributable
to the kernel and to nothing else, which is what makes it the honest foil.

## Question one: has the kernel concentrated?

A fidelity kernel's off-diagonal entries shrink like `2^-n`. Once the spread is at
that scale, every pair of inputs looks equally similar and no model built on the
matrix can separate them — the Gram matrix still exists, the SVM still fits, and the
accuracy is chance.

```python
feature_map = qk.ZZFeatureMap(2, reps=2)
kernel = qk.QuantumKernel(feature_map)
gram = kernel(X[:60])

report = qk.concentration_report(gram, n_qubits=2)
print(f"off-diagonal spread {report['off_diagonal_std']:.4f}")
print(f"predicted at 2 qubits {report['predicted_spread']:.4f}")
print(f"positive semi-definite: {report['is_psd']}")
```

At two qubits there is plenty of spread. The check earns its place as the width grows:
`shots_to_resolve` says roughly `4^n` shots are needed to see a `2^-n` signal above
sampling noise, so a 10-qubit fidelity kernel needs about a million shots *per entry*
before the number means anything.

## Question two: is the geometry even different?

Concentration says the kernel can still distinguish things. It does not say it
distinguishes anything the classical kernel could not. That is the geometric
difference:

```python
classical = np.exp(-0.5 * ((X[:60, None, :] - X[None, :60, :]) ** 2).sum(-1))
g = qk.geometric_difference(classical, gram)
print(f"g(K_classical, K_quantum) = {g:.1f}")
print("large -> a geometry the RBF kernel cannot reach" if g > 10
      else "small -> the classical kernel already spans this")
```

Huang et al.'s statistic: large means the two kernels induce genuinely different
geometries, so a separation is at least *possible*. Small means it is not, whatever
the accuracy table later says.

## Both at once

`diagnose` takes a Gram matrix directly:

```python
print(qk.diagnose(gram, n_qubits=2))
```

## The verdict

```python
K_train = kernel(X[:60])
svc = qk.QSVC(feature_map).fit(X[:60], y[:60])
scores = qk.evaluate.classification(y[60:], svc.predict(X[60:]))
print(f"QSVC balanced accuracy {scores['balanced_accuracy']:.3f}")
print(f"best classical         {table.best_classical.mean:.3f}  ({table.best_classical.name})")
print(f"circuits run: {kernel.n_evaluations:,}")
```

The two checks are worth running in that order because they can disagree, and the
disagreement is the finding. On the credit-risk data in
[`examples/credit_risk.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/credit_risk.py)
they do exactly that: geometric difference **66.9** — a geometry the RBF kernel cannot
reach — against a concentration report saying the spread is *already* at the `2^-n`
scale at four qubits. The reachable geometry is being squeezed out as fast as it
appears, and widening the register improves the first number while making the second
worse.

That tension is publishable, and it cost two function calls rather than a fortnight of
fitting SVMs.
