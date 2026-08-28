# Study 7 — Image data, and putting the structure in the circuit

Every other study on this page flattens its data and hands it to a general-purpose
ansatz. Images are the case where that is obviously wasteful: neighbouring pixels are
related, and a circuit that treats all wires as interchangeable has to learn that from
scratch with parameters it did not need to spend.

A quantum convolutional network builds the assumption in — a two-qubit filter slid
across the register, then a pooling layer halving the width, repeated until one wire
carries the answer.

## The data

```python
# docs: requires sklearn
import numpy as np
from sklearn.datasets import load_digits

import qmlkit as qk

digits = load_digits()
keep = (digits.target == 0) | (digits.target == 1)
X, y = digits.data[keep], digits.target[keep]
print(f"{X.shape[0]} images, {X.shape[1]} pixels (8x8), classes 0 and 1")

train, test = qk.imbalance.stratified_split(y, test_size=0.3, seed=0)
pipeline = qk.FeaturePipeline(n_qubits=4).fit(X[train])
Xtr, Xte = pipeline.transform(X[train]), pipeline.transform(X[test])
print(f"64 pixels -> 4 angles, {pipeline.explained_variance_:.0%} of the variance")
```

The full-size version uses real MNIST at 784 pixels and eight qubits — experiment 2 in
[`examples/experiments.py`](https://github.com/Ziadt160/qmlkit/blob/main/examples/experiments.py).
It needs a download, so this page uses the bundled 8×8 digits instead.

## The architecture

`QCNNLayer` is a torch module, so it composes with ordinary layers:

```python
# docs: requires torch
# docs: requires sklearn
import torch
from torch import nn

from qmlkit.nn.advanced import QCNNLayer

torch.manual_seed(0)
layer = QCNNLayer(4, filter="su4", tie_weights=True, init_seed=0).double()
model = nn.Sequential(layer, nn.Linear(1, 2).double())
print(f"{sum(p.numel() for p in model.parameters())} parameters in total")
```

**`tie_weights=True` is the convolution.** One filter block is reused at every position
rather than learning an independent block per pair, which is what makes it a
convolution and not just a sparse ansatz. It is also the case the gradient code has to
get right: one logical parameter drives several gates, so the derivative is the sum
over occurrences, each shifted on its own. Shifting them together computes something
else entirely.

```python
# docs: requires torch
# docs: requires sklearn
spec = layer.ansatz.build() if hasattr(layer, "ansatz") else None
print("weight tying means one parameter, several gates — see the parameter-shift guide")
```

## The filter is a choice, and it is measurable

```python
print(qk.list_conv_filters())
```

Four are registered, and `register_conv_filter` adds your own. They are shared with
`mps_ansatz` and `tree_tensor_network`, since all three slide the same two-qubit block.

A name that is not one of them is refused with the right one rather than accepted:

```python
try:
    QCNNLayer(4, filter="ry_cz")
except Exception as error:
    print(error)
```

## Train it

```python
# docs: requires torch
# docs: requires sklearn
inputs = torch.tensor(Xtr[:150])
targets = torch.tensor(y[train][:150], dtype=torch.long)
optimiser = torch.optim.Adam(model.parameters(), lr=0.15)
criterion = nn.CrossEntropyLoss()

for _ in range(12):
    optimiser.zero_grad()
    criterion(model(inputs), targets).backward()
    optimiser.step()

with torch.no_grad():
    predicted = model(torch.tensor(Xte)).argmax(1).numpy()
scores = qk.evaluate.classification(y[test], predicted)
print(f"balanced accuracy {scores['balanced_accuracy']:.3f} "
      f"on {scores.n_samples} held-out images")
```

Around 0.93 from roughly thirty parameters — the parameter count is the interesting
number, not the accuracy. A dense ansatz on four qubits with comparable depth carries
several times as many, and 0 vs 1 is a problem a linear model solves perfectly:

```python
# docs: requires sklearn
table = qk.baseline(np.vstack([Xtr, Xte]), np.concatenate([y[train], y[test]]),
                    cv=3, seed=0, include=["majority", "logistic"])
print(table)
```

## What this study is actually for

Not the accuracy. It is that **structure in the data can be structure in the circuit**,
and that the library makes that a one-line choice — `filter=`, `tie_weights=` — rather
than a rewrite. The same block vocabulary builds `mps_ansatz` and
`tree_tensor_network`, and `qk.compare_ansatze` will score them side by side on
expressibility, entanglement and gradient variance before any of them is trained.

The honest caveat is the one every study here shares: MNIST 0 vs 1 is separable by a
linear model, so nothing below is evidence of advantage. It is evidence that the
architecture can be expressed, trained, and measured without leaving the library.
