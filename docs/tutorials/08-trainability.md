# 8. Trainability

A correct gradient is not the same as a useful one. Variational circuits have a
failure mode where the gradient is exactly right and exponentially small — the
**barren plateau** — and no optimiser recovers from it, because there is nothing to
follow.

The good news is that it is measurable before you spend a training run finding out.

## Cost locality is the lever you actually control

At fixed shallow depth, *what you measure* matters more than how wide the register
is:

```python
import qmlkit as qk

local = qk.barren_plateau_scan(
    lambda n: qk.hardware_efficient(n, 2), [2, 4, 6, 8],
    lambda n: qk.Z(0), n_samples=200, seed=0,
)
global_ = qk.barren_plateau_scan(
    lambda n: qk.hardware_efficient(n, 2), [2, 4, 6, 8],
    lambda n: qk.PauliString(tuple((q, "Z") for q in range(n)), 1.0), n_samples=200, seed=0,
)

print(f"{'n':>3}{'local Z0':>14}{'global Z^n':>14}")
for i, n in enumerate(local["n_qubits"]):
    print(f"{n:>3}{local['variance'][i]:>14.3e}{global_['variance'][i]:>14.3e}")
print(f"\ndecay per qubit:  local {local['decay_per_qubit']:.4f}   global {global_['decay_per_qubit']:.4f}")
print(f"looks exponential: local {local['looks_exponential']}   global {global_['looks_exponential']}")
```

```text
  n      local Z0    global Z^n
  2     3.381e-01     1.250e-01
  4     2.684e-01     1.847e-02
  6     2.793e-01     8.684e-03
  8     2.740e-01     1.437e-03

decay per qubit:  local 0.9322   global 0.2257
looks exponential: local False   global True
```

A local `Z(0)` holds its gradient variance essentially flat from 2 to 8 qubits. A
global `Z^⊗n` on the *same circuits* collapses by a factor of 87 — decaying to 22% of
its value per added qubit. Same ansatz, same depth, same initialisation. The only
difference is the observable.

## Depth eventually wins anyway

Cost locality is not a cure, and claiming otherwise would be the comfortable
mistake. Let the depth grow with the width and the local cost collapses too:

```python
import qmlkit as qk

deep = qk.barren_plateau_scan(
    lambda n: qk.hardware_efficient(n, 2 * n), [2, 4, 6],
    lambda n: qk.Z(0), n_samples=200, seed=0,
)
for i, n in enumerate(deep["n_qubits"]):
    print(f"n={n}, L=2n: variance {deep['variance'][i]:.3e}")
print(f"decay per qubit {deep['decay_per_qubit']:.4f}, looks exponential {deep['looks_exponential']}")
```

```text
n=2, L=2n: variance 1.496e-01
n=4, L=2n: variance 3.540e-02
n=6, L=2n: variance 6.744e-03
decay per qubit 0.2123, looks exponential True
```

A **local** cost at depth `2n` decays at 0.2123 per qubit — indistinguishable from
the global cost's 0.2257 at shallow depth. So the honest statement is: a local cost
buys you room at shallow depth, and depth takes it back.

The practical defences are the unglamorous ones — shallow circuits, local costs,
small (near-identity) initialisation, and structured ansätze like QCNN or MPS whose
tied weights keep the effective parameter count low.

## Optimisers built for circuits

Adam and SGD come from torch. These three exploit structure a general optimiser
cannot see.

```python
import numpy as np
import qmlkit as qk

ansatz = qk.hardware_efficient(3, 2)
spec, start = ansatz.build(), ansatz.init("uniform", seed=1)
cost = qk.Z(0) + qk.Z(1) + qk.Z(2)  # minimum is -3

def loss(theta):
    return qk.expval(spec, cost, theta=theta)

_, roto = qk.minimize_rotosolve(loss, start, n_sweeps=12)
_, qng = qk.minimize_qng(spec, start, cost, n_steps=25, lr=0.15)
_, spsa = qk.minimize_spsa(loss, start, n_iterations=200, seed=0)

plain = start.copy()
for _ in range(25):
    plain = plain - 0.15 * qk.grad(spec, plain, cost)

print(f"start                  {roto[0]:+.6f}")
print(f"plain GD   (25 steps)  {loss(plain):+.6f}")
print(f"QNG        (25 steps)  {qng[-1]:+.6f}")
print(f"Rotosolve  (12 sweeps) {roto[-1]:+.6f}")
print(f"SPSA       (200 iters) {spsa[-1]:+.6f}")
print(f"minimum                -3.000000")
```

```text
start                  -0.085469
plain GD   (25 steps)  -2.995834
QNG        (25 steps)  -3.000000
Rotosolve  (12 sweeps) -2.997598
SPSA       (200 iters) -2.998708
minimum                -3.000000
```

**Rotosolve** exploits the fact that a circuit expectation is a *sinusoid* in any
single Pauli-rotation angle. Three evaluations pin that sinusoid down exactly, so you
jump to its minimum instead of stepping toward it — no learning rate at all. It is
coordinate descent, so it converges slowly near the optimum on correlated parameters.

**Quantum natural gradient** follows the Fubini–Study geometry rather than the
Euclidean one. Same 25 steps and same step size as plain gradient descent, and it
reaches the minimum where plain GD does not.

**SPSA** uses two circuit evaluations per step regardless of the parameter count.
Worth it when `P` is large or the evaluations are noisy.

## The geometry underneath

```python
import numpy as np
import qmlkit as qk

ansatz = qk.hardware_efficient(3, 1, rotations=("ry",), pattern="chain")
spec, theta = ansatz.build(), ansatz.init(seed=0)

g = qk.metric_tensor(spec, theta, approx=None)
print(f"metric shape {g.shape}, symmetric {np.allclose(g, g.T)}")
print(f"QFIM = 4·g: {np.allclose(qk.quantum_fisher_information(spec, theta), 4 * g)}")
```

The metric is computed by differentiating the state in closed form — exact, and with
no ancilla. That matters because QNG is only as good as the metric it follows.

!!! note "`approx=\"block-diag\"` means something different in PennyLane"
    PennyLane blocks the metric by circuit *layer* and zeroes cross-layer entries.
    qmlkit computes the exact metric, which costs no more on a simulator. The
    consequence is measurable, and it is in [Validation](../about/validation.md).

## What to check before a long run

```python
import qmlkit as qk

report = qk.metrics.AnsatzReport(qk.hardware_efficient(4, 2), n_samples=300)
print(report)
```

Look at **gradient variance** first. If it is already at `1e-4` on four qubits,
widening the register will not help and the architecture needs changing, not more
epochs.

---

That is the tour. From here: the [guides](../guides/index.md) go deeper on the
parameter-shift rule and on extending the library, and
[Validation](../about/validation.md) covers how any of this is known to be correct.
