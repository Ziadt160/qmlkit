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
