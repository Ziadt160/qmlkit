# Study 6 — Clinical data, where the two errors are not the same error

Credit risk in [study 1](01-imbalanced-classification.md) was about a skewed dataset.
This one is about something else that no amount of balancing fixes: **a false negative
and a false positive are not equivalent mistakes**, and no single-number metric knows
that.

Telling a woman with a malignant tumour that she is fine is not the same kind of error
as calling her back for a second scan. Accuracy scores them identically. So does
balanced accuracy, and so does F1.

## The data

```python
# docs: requires sklearn
import numpy as np
from sklearn.datasets import load_breast_cancer

import qmlkit as qk

data = load_breast_cancer()
X, y = data.data, data.target          # 0 = malignant, 1 = benign
print(f"{X.shape[0]} samples, {X.shape[1]} clinical features")
print(f"classes {list(data.target_names)}, malignant rate {(y == 0).mean():.3f}")
```

At 37% malignant this is only mildly imbalanced — `imbalance_report` says as much, and
correctly does not recommend the remedies study 1 needed:

```python
# docs: requires sklearn
report = qk.imbalance.imbalance_report(y)
print(report if report else "nothing to report: the classes are close enough to even")
```

That is the point of the report being falsy when it finds nothing. The skew is not the
problem here; the **asymmetry of the costs** is, and no diagnostic can infer that from
the labels. It comes from the domain.

## Train it

```python
# docs: requires torch
# docs: requires sklearn
import torch

train, test = qk.imbalance.stratified_split(y, test_size=0.3, seed=0)
pipeline = qk.FeaturePipeline(n_qubits=4).fit(X[train])
Xtr, Xte = pipeline.transform(X[train]), pipeline.transform(X[test])
print(f"30 features -> 4 angles, {pipeline.explained_variance_:.0%} of the variance")

torch.manual_seed(0)
model = qk.VQC(n_features=4, n_classes=2, n_qubits=4, n_layers=3, seed=0)
model.fit(Xtr, y[train], epochs=25, lr=0.08, batch_size=256)

scores = qk.evaluate.classification(y[test], model.predict(Xte))
print(f"accuracy {scores['accuracy']:.3f}   balanced {scores['balanced_accuracy']:.3f}")
```

Around 0.93 either way. A good-looking result, and on its own it does not tell you
whether the model is safe to use.

## The number that actually matters

```python
# docs: requires torch
# docs: requires sklearn
per_class = scores.extras["per_class"]
malignant = per_class["0"]
print(f"malignant recall {malignant['recall']:.3f} "
      f"on {malignant['support']} malignant cases")
print(f"malignant precision {malignant['precision']:.3f}")
print("\nconfusion matrix (rows = truth, columns = prediction):")
print(scores.extras["confusion_matrix"])
```

The confusion matrix is where the model stops being a score. The top-right entry —
malignant cases predicted benign — is roughly **five patients** out of sixty-four.
Accuracy 0.93 and *five missed cancers* are the same model.

`qk.evaluate.classification` hands back `per_class`, `support` and the confusion matrix
alongside the summary metrics for exactly this reason. The summary is what you report;
the breakdown is what you decide on.

## Moving the operating point

Nothing above is fixed. The model outputs probabilities, and the threshold is a policy
choice rather than a modelling one:

```python
# docs: requires torch
# docs: requires sklearn
probabilities = model.predict_proba(Xte)[:, 0]          # P(malignant)
print(f"{'threshold':>10}{'malignant recall':>19}{'false alarms':>15}")
for threshold in (0.5, 0.35, 0.2):
    predicted = np.where(probabilities > threshold, 0, 1)
    s = qk.evaluate.classification(y[test], predicted)
    recall = s.extras["per_class"]["0"]["recall"]
    false_alarms = int(((predicted == 0) & (y[test] == 1)).sum())
    print(f"{threshold:>10.2f}{recall:>19.3f}{false_alarms:>15}")
```

Lowering the threshold catches more cancers and calls back more healthy patients.
There is no setting that does both, and the library will not pick for you — that is a
clinical decision with a cost ratio attached, and a library that chose silently would
be making it on your behalf.

`average_precision` is the summary to quote when the threshold is not fixed, because it
integrates over all of them:

```python
# docs: requires torch
# docs: requires sklearn
full = qk.evaluate.classification(y[test], model.predict(Xte), model.predict_proba(Xte))
print(f"average_precision {full['average_precision']:.3f}   roc_auc {full['roc_auc']:.3f}")
```

## The verdict

```python
# docs: requires torch
# docs: requires sklearn
Xq = qk.FeaturePipeline(n_qubits=4).fit(X).transform(X)
table = qk.baseline(Xq, y, cv=3, seed=0,
                    include=["majority", "logistic", "random-forest"])
print(table.verdict)
```

A logistic regression on the same four components is hard to beat here, which is the
expected outcome and worth stating. The study's contribution is not the model. It is
that **the metric was chosen from the decision the model feeds**, not from the shape of
the data — and that the library gives you the confusion matrix and the threshold sweep
without being asked twice.

The full-size version, on all 30 features with a per-qubit comparison, is experiment 3
in [`examples/experiments.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/experiments.py).
