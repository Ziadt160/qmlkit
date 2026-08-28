# Study 1 — Imbalanced classification, and the metric that lies

The full version of this runs on the
[Kaggle credit-risk table](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) —
32,581 loan applications, 21.8% of which defaulted — in
[`examples/credit_risk.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/credit_risk.py).
This page is the same twelve steps compressed onto data the page can generate, so
every number below is produced by the code beside it.

## The data, and what its labels will break

```python
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)
X = rng.normal(size=(400, 4))
score = 1.6 * X[:, 0] - 1.1 * X[:, 1] + 0.5 * X[:, 2] + 0.6 * rng.normal(size=400)
y = (score > np.quantile(score, 0.78)).astype(int)   # 22% positive, like the real thing

print(qk.imbalance.imbalance_report(y))
```

Two findings, and both name the call that fixes them. Everything this study does about
the skew comes from that report and nowhere else:

```python
train, test = qk.imbalance.stratified_split(y, test_size=0.3, seed=0)
print(f"{train.size} train / {test.size} test, {int(y[test].sum())} positives held out")
```

A random split of a 78/22 problem leaves the minority class thin or absent in test
often enough to make a single test score meaningless. Stratified splitting removes
that as a source of variance before it becomes one.

## The bar, before any quantum code

```python
table = qk.baseline(X, y, cv=3, seed=0,
                    include=["majority", "logistic", "random-forest", "rbf-kernel-ridge"])
print(table)
```

Read the *failures* as well as the winner. On the real credit table, `svc-rbf` and
`mlp` sit at the 0.500 floor because the raw columns span six orders of magnitude —
the table diagnosed the preprocessing before any model was tuned.

## The naive model, and the metric that admits it

```python
# docs: requires torch
import torch

pipeline = qk.FeaturePipeline(n_qubits=4).fit(X[train])
Xtr, Xte = pipeline.transform(X[train]), pipeline.transform(X[test])

torch.manual_seed(0)
naive = qk.VQC(n_features=4, n_classes=2, n_qubits=4, n_layers=2, seed=0)
naive.fit(Xtr, y[train], epochs=20, lr=0.08, batch_size=256)
naive_scores = qk.evaluate.classification(y[test], naive.predict(Xte))
print(naive_scores)
```

`Scores` returns every metric at once precisely so the disagreement between them stays
visible, and prints a note when accuracy is overstating the model. `primary` is
`balanced_accuracy` rather than `accuracy` here, chosen from the class distribution
rather than by the author.

## One keyword, taken from step one

```python
# docs: requires torch
torch.manual_seed(0)
weighted = qk.VQC(n_features=4, n_classes=2, n_qubits=4, n_layers=2,
                  class_weight="balanced", seed=0)
weighted.fit(Xtr, y[train], epochs=20, lr=0.08, batch_size=256)
weighted_scores = qk.evaluate.classification(y[test], weighted.predict(Xte))

for name, s in (("naive", naive_scores), ("class_weight='balanced'", weighted_scores)):
    print(f"  {name:24} balanced_accuracy {s['balanced_accuracy']:.3f}"
          f"  mcc {s['mcc']:+.3f}  accuracy {s['accuracy']:.3f}")
```

On the real table accuracy *falls* — 0.811 to 0.714 — while balanced accuracy rises
0.692 to 0.730. The weighted model finds more of the defaults and pays in false
alarms. MCC can move the other way, and the library hands back both rather than
picking the one that flatters the change. Which trade you want is a lending decision,
not a modelling one.

## Is the circuit quietly broken?

```python
# docs: requires torch
print(qk.diagnose(weighted))
```

`hardware_efficient` ends each layer in `rz`, which commutes with the `cx` entanglers
*and* with any Z-basis observable — so a quarter of its weights cannot move the
readout. The optimiser carries them every step and the loss curve never shows it.

## The verdict

```python
# docs: requires torch
def build():
    torch.manual_seed(0)
    model = qk.VQC(n_features=4, n_classes=2, n_qubits=4,
                   class_weight="balanced", seed=0)
    fit = model.fit
    model.fit = lambda a, b: fit(a, b, epochs=15, lr=0.08, batch_size=256)
    return model

Xq = qk.FeaturePipeline(n_qubits=4).fit(X).transform(X)
final = qk.baseline(Xq, y, model=build, cv=3, seed=0,
                    include=["majority", "logistic", "random-forest"])
print(final.verdict)
```

The classical rows are scored on the **same four principal components** the quantum
model sees. Comparing a 4-component model against a 21-feature one would be a claim
about the input rather than the model — the full example reports both bars and clears
neither, and says so in as many words.

A negative result you can defend is worth more than a positive one you cannot.
