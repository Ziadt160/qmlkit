# Study 3 — Regression, and two metrics that disagree

Classification hides a bad model behind accuracy. Regression hides one behind `r2`,
which is scaled by the variance of whatever you happened to sample — so the same model
scores differently on a narrow test set and a wide one, and neither number is wrong.

`qk.evaluate.regression` returns all of them, and says when `r2` has stopped meaning
anything.

## A target with structure a linear model cannot reach

```python
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)
X = rng.uniform(-1.0, 1.0, size=(160, 3))
y = np.sin(2.0 * X[:, 0]) + 0.4 * X[:, 1] ** 2 + 0.1 * rng.normal(size=160)

train, test = np.arange(120), np.arange(120, 160)
print(f"target range [{y.min():.2f}, {y.max():.2f}], variance {y.var():.3f}")
```

## The bar first

```python
table = qk.baseline(X, y, cv=3, seed=0, include=["mean", "linear", "rbf-kernel-ridge"])
print(table)
```

`mean` scores `r2 = 0` by construction — it is the definition of the zero point, not a
model. `linear` is the one that matters here: the target is deliberately non-linear, so
the gap between `linear` and `rbf-kernel-ridge` is how much non-linear structure is
actually available to be captured. If that gap is small, no model of any kind is going
to look impressive, and it is better to know before training one.

## The quantum regressor

```python
# docs: requires torch
import torch

pipeline = qk.FeaturePipeline(n_qubits=3).fit(X[train])
Xtr, Xte = pipeline.transform(X[train]), pipeline.transform(X[test])

torch.manual_seed(0)
model = qk.VQRegressor(n_features=3, n_qubits=3, n_layers=2, seed=0)
model.fit(Xtr, y[train], epochs=30, lr=0.08, batch_size=256)

scores = qk.evaluate.regression(y[test], model.predict(Xte))
print(scores)
```

Seven metrics, and they answer different questions:

- **`r2`** — how much of the variance is explained, relative to predicting the mean.
  Comparable across models on *this* data and not across datasets.
- **`rmse`** and **`mae`** — error in the target's own units. `rmse` punishes large
  misses quadratically; `mae` does not. When they disagree, the residuals are skewed.
- **`median_absolute_error`** — the typical miss, immune to a handful of outliers.
- **`max_error`** — the worst single case, which is the only one that matters if the
  prediction feeds a decision with a floor under it.
- **`explained_variance`** — `r2` without the bias term, so the gap between them is
  exactly the model's systematic offset.

`mape` is omitted here rather than returned as infinity, and the note says why: it is
undefined wherever the target is zero, and this target crosses zero.

## Where the quantum model's ceiling comes from

A variational circuit with angle encoding is a **Fourier series in the input**, and its
reachable frequencies are set by how many times the data is uploaded — not by how many
weights it has:

```python
print(qk.fourier.reachable_frequencies(1))   # one upload
print(qk.fourier.reachable_frequencies(3))   # three uploads
```

The target contains `sin(2x)`, so a single-upload model has no frequency-2 component to
fit it with, however many layers are stacked on top. That is a representational limit,
not an optimisation one, and no learning rate will move it.

Re-uploading is the fix — with the trap the library warns about:

```python
fmap = qk.AngleFeatureMap(3, entangle=False)
good = qk.reupload(fmap, n_layers=3, block=qk.RotationLayer(("rz", "ry", "rz")))
print(f"{good.n_params} weights, reaching frequencies 0..3")
```

If the trainable block used `ry` — the same generator the encoding uses — the uploads
would merge into a single rotation, the model would reach one frequency, and every
weight would become a phase. It would still train, still converge, and still report a
loss. `reupload()` warns at construction, and `qk.diagnose` catches it on a model
composed by hand:

```python
collapsed = qk.Ansatz(3, qk.repeat(3, qk.EncodingLayer(fmap) + qk.RotationLayer("ry")),
                      name="ry-reupload", n_inputs=3)
print(qk.diagnose(collapsed))
```

## The verdict

```python
# docs: requires torch
print(f"quantum r2 {scores['r2']:.3f}  vs  best classical "
      f"{table.best_classical.mean:.3f} ({table.best_classical.name})")
```

Verify the spectrum of any re-uploading model you build with
`qmlkit.fourier.spectrum` rather than trusting the upload count — the frequencies are
reachable, not guaranteed.
