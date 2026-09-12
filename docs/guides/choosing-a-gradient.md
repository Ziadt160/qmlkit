# Choosing a gradient method

Six methods, and `qk.grad(spec, theta, obs)` with no `method` picks a sensible one.
This page is for when you want to override it.

## The short answer

| Situation | Use |
|---|---|
| Simulating, and you just want the gradient | **`adjoint`** (the default) |
| The circuit lives inside a torch autograd graph | **`backprop`** |
| You are modelling what hardware would do | **`parameter-shift`**, with `shots` |
| Hardware, and circuit count is the binding constraint | **`hadamard`**, if the ancilla can reach every wire |
| Very many parameters, or very noisy evaluations | **`spsa`** |
| Checking another method | **`finite-diff`** |

## The full picture

| Method | Cost | Exact | Runs on hardware |
|---|---|---|---|
| `adjoint` | one backward pass | yes | no — needs the statevector |
| `backprop` | one autograd pass | yes | no — needs the statevector |
| `hadamard` | `P` circuits + one ancilla | yes | yes, given the connectivity |
| `parameter-shift` | `2P` circuits, more for four-term gates | yes | yes |
| `spsa` | 2 evaluations, any `P` | no — unbiased estimate | yes |
| `finite-diff` | `2P` | no — `O(h²)` bias | technically, but don't |

Measured on a 5-qubit hardware-efficient ansatz with a two-term observable:

| `P` | `adjoint` | `backprop` | `hadamard` | `parameter-shift` | `finite-diff` |
|---|---|---|---|---|---|
| 20 | **2.2 ms** | 8.5 ms | 15 ms | 28 ms | 29 ms |
| 60 | **6.2 ms** | 24 ms | 109 ms | 213 ms | 226 ms |
| 120 | **12.6 ms** | 50 ms | 404 ms | 823 ms | 870 ms |

## Choosing a gradient is not choosing an optimiser

`grad()` decides how the derivative is computed; what you do with it is separate.
Four optimisers are registered, reachable by name wherever a `VQE`-style algorithm
takes one:

| name | needs a gradient | when |
|---|---|---|
| `adam` | yes | the default choice for a variational circuit of any depth |
| `gradient-descent` | yes | when you want the plainest possible baseline to compare against |
| `rotosolve` | no | shallow circuits where each angle drives one Pauli rotation — no learning rate to pick |
| `spsa` | no | two evaluations per step whatever `P` is; the shot-budget option |

`adam` is the one to reach for first. Parameter gradients in a deep ansatz differ in
scale by orders of magnitude — a rotation near the readout moves the expectation far
more than one behind a wall of entanglers — so a single learning rate either crawls
on the small ones or diverges on the large. Adam divides by the running gradient
magnitude, which makes the step size per-parameter.

```python
from qmlkit.optim import minimize_adam
import qmlkit as qk

ansatz = qk.hardware_efficient(3, 2)
spec, obs = ansatz.build(), qk.Z(0)
theta, history = minimize_adam(
    lambda t: qk.expval(spec, obs, theta=t),
    ansatz.init("small", seed=0),
    lambda t: qk.grad(spec, t, obs),
    n_steps=40,
)
print(f"{history[0]:+.4f} -> {history[-1]:+.4f}")
```

`adam_step` and `AdamState` are public for the case where the loop is yours. Keep the
state between steps: dropping it turns Adam back into gradient descent with a
decaying learning rate, which trains, converges, and is not what you asked for.

## What `method="auto"` decides

```python
import qmlkit as qk

ansatz = qk.hardware_efficient(3, 2)
spec = ansatz.build()

print("no shots:  ", qk.choose_method(spec))
print("with shots:", qk.choose_method(spec, shots=1000))
```

```text
no shots:   adjoint
with shots: parameter-shift
```

Adjoint when every gate has a closed-form derivative and the backend can produce a
statevector; parameter-shift otherwise. Asking for `shots` rules out adjoint by
definition — you cannot sample a statevector you are not allowed to read.

Methods that need the statevector **refuse** a shot budget rather than silently
ignoring it:

```python
import qmlkit as qk

ansatz = qk.hardware_efficient(2, 1)
spec, theta = ansatz.build(), ansatz.init(seed=0)
try:
    qk.grad(spec, theta, qk.Z(0), method="adjoint", shots=1000)
except ValueError as exc:
    print(exc)
```

## Notes on each

**`adjoint`** — one forward pass and one backward pass, whatever `P` is. Exact. The
right default on a simulator, and the reason this library exists in a simulator-only
0.x: it makes the cost of a gradient independent of the parameter count.

**`backprop`** — differentiates a torch statevector simulator directly. Exact, and
slower than adjoint for a standalone gradient because of per-gate tensor overhead.
Its reason to exist is that the circuit sits *inside* an autograd graph, which is
what `QuantumLayer` needs. It is also the least physical method here: it reads
intermediate states no device will expose, and its memory grows with depth.

How much slower, measured on a hardware-efficient ansatz:

| qubits | `P` | `adjoint` | `backprop` | `parameter-shift` |
|---|---|---|---|---|
| 3 | 12 | **1.4 ms** | 4.7 ms | 8.2 ms |
| 4 | 24 | **2.3 ms** | 9.0 ms | 31.5 ms |
| 6 | 36 | **3.8 ms** | 15.9 ms | 79.2 ms |
| 8 | 32 | **3.6 ms** | 16.3 ms | 93.5 ms |

Batched over a training batch of 128, the gap widens and `backprop` drops out
entirely - it has no batched form, and `grad_batch` says so rather than falling back:

| qubits | `P` | `adjoint` | `parameter-shift` |
|---|---|---|---|
| 4 | 16 | **7.6 ms** | 58.7 ms |
| 6 | 24 | **28.3 ms** | 430.9 ms |
| 8 | 32 | **118 ms** | 2552 ms |

This is worth stating because the intuition travels badly. In a framework where every
circuit evaluation goes through a dispatch layer, `backprop` can be dramatically
*faster* than `adjoint` - the dispatch dominates, and backprop pays it once instead of
once per parameter. qmlkit's adjoint is a direct NumPy sweep with no dispatch to
amortise, so the ranking inverts. If you arrive expecting backprop to win, measure
before switching; `method="auto"` already picks the fast one here.

**`hadamard`** — one circuit per parameter instead of two, using an ancilla in `|+⟩`
and a controlled generator. Unlike adjoint it is a real measurement, so it stays
valid on hardware. The trade is an ancilla that must couple to every wire the
generator touches; on real devices that routing cost usually eats the saving, which
is why parameter-shift stays the hardware default. It refuses controlled rotations
rather than guessing, because their generators are not Paulis.

**`parameter-shift`** — exact, hardware-valid, and the one worth understanding in
detail: see [The parameter-shift rule](parameter-shift.md).

**`spsa`** — two evaluations per gradient regardless of `P`. Stochastic but unbiased,
so averaging converges on the true gradient. Use `n_avg` to trade evaluations for
variance.

**`finite-diff`** — biased by construction at `O(h²)`, and noisy at `O(1/h)` when
sampling. It exists to check other methods. It should never be the method you train
with, and the fact that it sometimes lands within `1e-9` is luck, not accuracy.

## Bringing your own

```python
import numpy as np
import qmlkit as qk

@qk.register_gradient("my_estimator")
def my_estimator(spec, theta, obs, *, backend=None, shots=None, **kwargs):
    return np.zeros(spec.n_params)

print("my_estimator" in qk.list_gradient_methods())
```

Once registered it is a keyword everywhere the library takes `method=`, including
`QuantumLayer`. See [Extending qmlkit](extending.md).
