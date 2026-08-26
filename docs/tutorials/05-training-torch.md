# 5. Training with PyTorch

Needs the extra:

```bash
pip install "qmlkit[torch]"
```

A circuit becomes an `nn.Module` and everything torch already knows how to do —
optimisers, schedulers, batching, autograd — applies unchanged.

## The two-line path

```python
# docs: requires torch
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 4))
y = (X[:, 0] * X[:, 1] > 0).astype(int)  # XOR-like: not linearly separable

model = qk.VQC(n_features=4, n_classes=2, seed=0).fit(X, y, epochs=30)
print(f"accuracy {model.score(X, y):.1%}")
print(f"loss {model.history_[0]:.4f} -> {model.history_[-1]:.4f}")
```

```text
accuracy 78.3%
loss 0.7836 -> 0.4640
```

`VQC` is a convenience, not a wall — every default is one keyword away from being
something else, and the feature map, ansatz, observables and optimiser are all
arguments.

## The layer underneath

```python
# docs: requires torch
import torch
from torch import nn

import qmlkit as qk

torch.manual_seed(0)

layer = qk.QuantumLayer(
    qk.ZZFeatureMap(3, reps=1),
    qk.hardware_efficient(3, 2),
    [qk.Z(0), qk.Z(1)],
    init_seed=0,
).double()

net = nn.Sequential(nn.Linear(6, 3), nn.Tanh(), layer, nn.Linear(2, 2)).double()

xb = torch.randn(8, 6, dtype=torch.float64)
loss = nn.CrossEntropyLoss()(net(xb), torch.randint(0, 2, (8,)))
loss.backward()

print(f"output shape       {tuple(net(xb).shape)}")
print(f"pre-net grad norm  {float(net[0].weight.grad.norm()):.6f}")
print(f"quantum grad norm  {float(layer.theta.grad.norm()):.6f}")
print(f"post-net grad norm {float(net[3].weight.grad.norm()):.6f}")
```

```text
output shape       (8, 2)
pre-net grad norm  0.330885
quantum grad norm  0.114621
post-net grad norm 0.012605
```

## The line that matters

**`pre-net grad norm 0.330885`.** The `nn.Linear(6, 3)` sitting *before* the quantum
layer receives a real gradient, so it trains.

This is not automatic. It requires `∂f/∂x` — the derivative of the circuit with
respect to its *encoding angles*, not just its weights — and many hand-rolled
implementations return `None` there. When they do, everything upstream of the circuit
silently freezes. The loss still falls, because the quantum weights still train, so
the failure presents as slow convergence rather than as a bug. A dressed network
whose classical pre-net never moves is doing far less than it appears to.

qmlkit computes it by differentiating the circuit with respect to its encoding angles
and finishing the chain rule classically, so a *nonlinear* feature map costs no extra
circuits.

```python
# docs: requires torch
import torch
from torch.autograd import gradcheck

import qmlkit as qk

layer = qk.QuantumLayer(
    qk.AngleFeatureMap(2, entangle=False),
    qk.hardware_efficient(2, 1),
    [qk.Z(0)],
    init_seed=0,
).double()

x = torch.randn(1, 2, dtype=torch.float64, requires_grad=True)
print("gradcheck on the inputs:", gradcheck(lambda v: layer(v), (x,), eps=1e-6, atol=1e-6))
```

```text
gradcheck on the inputs: True
```

`torch.autograd.gradcheck` compares the analytic backward pass against numerical
differentiation of the forward pass. Passing it for the *inputs* is the assertion
that the pre-net gradient is real rather than merely non-`None`.

## Regression

```python
# docs: requires torch
import numpy as np
import qmlkit as qk

xs = np.linspace(-1, 1, 60).reshape(-1, 1)
ys = np.sin(3 * xs).ravel()

model = qk.VQRegressor(n_features=1, seed=0).fit(xs, ys, epochs=60)
print(f"R² {model.score(xs, ys):.4f}")
print(f"loss {model.history_[0]:.4f} -> {model.history_[-1]:.4f}")
```

```text
R² 0.9917
loss 0.6780 -> 0.0042
```

A one-qubit re-uploading model fits `sin(3x)` because three uploads reach frequency 3
— which is exactly the claim [tutorial 7](07-reuploading.md) turns into a measurement
rather than an assertion.

## Choosing the gradient method

`QuantumLayer` takes `grad_method`. On a simulator, leave it alone: the default
resolves to adjoint, which costs one pass regardless of the parameter count. Set it
to `"parameter-shift"` when you want to see what the model would do on hardware,
optionally with `shots`.

```python
# docs: requires torch
import qmlkit as qk

hardware_like = qk.QuantumLayer(
    qk.AngleFeatureMap(2, entangle=False),
    qk.hardware_efficient(2, 1),
    [qk.Z(0)],
    grad_method="parameter-shift",
    shots=2048,
    init_seed=0,
)
print(hardware_like)
```

## Structured architectures

The same layer machinery, arranged into the shapes the literature names:

| | |
|---|---|
| `QCNNLayer` | convolution and pooling with tied weights |
| `MPSLayer` | matrix-product-state contraction order |
| `QLSTM` | a recurrent cell with quantum gates |
| `DressedQuantumNet` | classical → quantum → classical, the transfer-learning shape |

---

**Next:** [Quantum kernels](06-quantum-kernels.md) — the other way to use a feature
map, with no variational training at all.
