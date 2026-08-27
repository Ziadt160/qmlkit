# Evaluating a quantum model honestly

Three things go wrong between a trained model and a reported result, and none of
them raises an exception.

1. **The metric flatters the model.** Accuracy on a skewed dataset is high for a
   model that has learned to predict the majority class.
2. **The comparison is missing.** "Compared to what?" is the first question a
   reviewer asks, and the RBF-kernel SVM usually never got run.
3. **The number cannot be reproduced.** Library version, SDK version, backend and
   seed all move the answer, and none of them is recorded.

This guide covers the three modules that close those gaps:
[`qmlkit.evaluate`](../reference/evaluation.md), [`qmlkit.imbalance`](../reference/evaluation.md)
and [`qk.baseline`](../reference/evaluation.md).

## Every metric for the task, in one call

`qk.evaluate` groups metrics by task rather than making you assemble them:

```python
import numpy as np
import qmlkit as qk

y_true = np.array([0] * 95 + [1] * 5)
y_pred = np.zeros(100, dtype=int)          # a model that always says "class 0"

scores = qk.evaluate.classification(y_true, y_pred)
print(round(scores["accuracy"], 3), round(scores["balanced_accuracy"], 3))
```

That prints `0.95 0.5`. The model has learned nothing, and accuracy says 95%.

The point of returning every metric at once is that the disagreement between them
stays visible. `Scores` also says so directly:

```python
print(scores.primary)
print(scores.notes[0][:60])
```

`primary` is `balanced_accuracy` here rather than `accuracy`, because the class
distribution makes accuracy unusable — and `notes` explains why in a sentence.

There are four tasks: `classification`, `regression`, `clustering` and
`generative`. Each returns the same `Scores` object, which indexes like a dict:

```python
reg = qk.evaluate.regression([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8])
sorted(reg.keys())[:4]
```

Nothing in `qmlkit.evaluate` needs scikit-learn. Everything in it is asserted equal
to scikit-learn in `tests/test_evaluate.py`, on randomly generated inputs — the same
cross-validation-against-a-second-implementation approach used for the
[PennyLane parity suite](../about/validation.md).

## Skewed classes

Imbalance breaks the loss and the split, not only the score.
`qk.imbalance.imbalance_report` says which:

```python
report = qk.imbalance.imbalance_report(y_true)
print(report.codes)
```

Each finding carries the call that fixes it. The three that matter:

```python
qk.imbalance.class_weights(y_true)          # {0: 0.526..., 1: 10.0} - for a weighted loss
qk.imbalance.pos_weight(y_true)             # 19.0 - for BCEWithLogitsLoss
train, test = qk.imbalance.stratified_split(y_true, test_size=0.2, seed=0)
int(y_true[test].sum())                      # 1, never 0
```

A random 80/20 split of this data leaves the minority class out of the test set
entirely about a third of the time, which makes the test score noise.
`stratified_split` and `stratified_folds` guarantee it cannot happen.

`VQC` takes the weighting directly, computed from the `y` passed to `fit`:

```python
model = qk.VQC(n_features=2, n_classes=2, class_weight="balanced")
```

`focal_gamma=2.0` additionally down-weights examples the model already gets right
— worth reaching for when the majority class is not merely abundant but trivially
separable. On a variational circuit that matters more than it does classically,
because gradient signal spent on easy examples is a budget measured in circuits.

## The classical bar

`qk.baseline` runs every classical baseline on the same folds, the same metric and
the same preprocessing as the model under test:

```python
X, y = qk.datasets.make_moons(n_samples=60, seed=0)
table = qk.baseline(X, y, cv=3, seed=0, include=["majority", "rbf-kernel-ridge"])
print(table.best_classical.name)
```

`rbf-kernel-ridge` is the one to watch for a quantum kernel method: it is the
*identical* algorithm — a closed-form kernel ridge solve — differing only in which
kernel fills the Gram matrix. Any gap between it and a quantum kernel is
attributable to the kernel and nothing else.

Pass `model=` and the model joins the table with a verdict:

```python
table = qk.baseline(X, y, model=qk.baselines.NearestCentroid(), cv=3, seed=0,
                    include=["majority"])
print(table.beats_classical)
```

The verdict does not call a lead a result when the lead is smaller than the
fold-to-fold spread:

> `quantum (0.810) leads svc-rbf (0.800) by 0.010, which is inside the fold-to-fold
> spread (0.071) — not yet a result`

Baselines that need scikit-learn are listed as **skipped** when it is absent rather
than dropped, because a table that quietly omits the strong baseline is the problem
this module exists to solve.

The companion check for kernel methods is `qk.geometric_difference`, which asks
whether the quantum kernel reaches a geometry the classical one cannot. A large
geometric difference with no accuracy gain is a real finding; a small one says the
classical kernel was always going to be enough.

## Before the run: what it will cost

`qk.plan` computes the circuit budget from the ansatz, the training set size and
the gradient method, and lists the cheaper routes with what each gives up:

```python
budget = qk.plan(qk.hardware_efficient(4, 3), n_samples=100, steps=50, shots=1024)
print(f"{budget.circuits:,} circuits, {budget.hours(0.5):.1f} hours at 0.5 s each")
print([r.name for r in budget.reductions])
```

Qubit-wise-commuting grouping is counted rather than assumed, so a four-term
observable that shares one measurement setting is costed as one circuit, not four.

## After the run: was it right, and can it be repeated

`qk.selfcheck` computes the gradient by every exact route available — adjoint,
backprop, Hadamard-test, parameter-shift — and compares them. They share the
circuit IR and almost nothing else, so agreement is evidence and disagreement
localises the method that is wrong:

```python
ansatz = qk.hardware_efficient(3, 2)
check = qk.selfcheck(ansatz.build(), np.full(ansatz.n_params, 0.3), qk.Z(0))
bool(check)          # False: every route agrees, nothing to report
```

This is what to run when a number looks wrong and nothing raised. It catches a
custom gate with wrong declared `frequencies`, which produces a plausible gradient
rather than an error.

`qk.fingerprint` records the stack that decided the number:

```python
stamp = qk.fingerprint(seed=0, shots=1024)
sorted(stamp.as_dict())[:4]
```

It is JSON-serialisable, so it is cheap enough to attach to every result file.

## Putting it together

The order these run in is the order the questions arise:

| Before training | `qk.imbalance.imbalance_report(y)` · `qk.plan(model, ...)` · `qk.diagnose(ansatz)` |
| During | `class_weight="balanced"` · `stratified_folds(y)` |
| After | `qk.evaluate.classification(...)` · `qk.baseline(X, y, model=...)` |
| Before publishing | `qk.selfcheck(...)` · `qk.fingerprint(...)` |
