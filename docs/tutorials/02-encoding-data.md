# 2. Getting data in

A quantum model can only learn functions its encoding can express. That makes this the
most consequential choice in the pipeline, and the one most often made by accident —
before any training happens, the encoding has already fixed the hypothesis class.

## Angle encoding: one number per qubit

The default, and usually the right first move.

```python
import qmlkit as qk

spec = qk.angle_encode([0.5, 1.2, 2.0])
print(qk.draw(spec))
```

```text
q0: ─RY(0.50)──
q1: ─RY(1.20)──
q2: ─RY(2.00)──
```

`n` features cost `n` qubits and depth 1. Cheap, shallow, and the amplitudes are
smooth in the data — which is what makes gradients well behaved.

## Basis encoding: bits to qubits

```python
import qmlkit as qk

print(qk.run_counts(qk.basis_encode([1, 0, 1]), 256, seed=0))
```

```text
{'101': 256}
```

Exact, trivial, and no superposition — a single computational basis state. Useful for
combinatorial problems, useless as a feature map for continuous data.

## Amplitude encoding: `2ⁿ` numbers in `n` qubits

Exponentially compact in qubits, and you pay for it in gates.

```python
import numpy as np
import qmlkit as qk

vector = [1, 2, 3, 4]
spec = qk.amplitude_encode(vector)

print(qk.draw(spec))
print("prepared:", np.round(np.abs(qk.statevector(spec)), 4))
print("target:  ", np.round(np.abs(np.array(vector) / np.linalg.norm(vector)), 4))
```

```text
q0: ─RY(2.30)──@────────────@──
q1: ─RY(2.03)──X──RY(0.18)──X──
prepared: [0.1826 0.3651 0.5477 0.7303]
target:   [0.1826 0.3651 0.5477 0.7303]
```

qmlkit builds this from **uniformly-controlled rotations**, not a backend
state-preparation primitive. The circuit is made of ordinary registered gates, so it
runs identically on every backend, can be drawn and transpiled — and its cost is
visible rather than hidden inside an SDK call:

```python
import numpy as np
import qmlkit as qk

for n_qubits in (2, 4, 8):
    spec = qk.amplitude_encode(np.arange(1, 2 ** n_qubits + 1))
    gates = sum(spec.gate_counts().values())
    print(f"{2 ** n_qubits:>4} numbers -> {n_qubits} qubits, {gates:>4} gates, depth {spec.depth()}")
```

```text
   4 numbers -> 2 qubits,    5 gates, depth 4
  16 numbers -> 4 qubits,   37 gates, depth 34
 256 numbers -> 8 qubits,  749 gates, depth 742
```

The qubit count grows logarithmically and the gate count grows linearly in the data
size. "Exponentially compact" is true and, on its own, misleading.

!!! warning "One global phase is dropped"
    The phase cascade reproduces every *relative* phase exactly and drops one overall
    factor, which is unobservable — until you put the block inside a larger
    **controlled** circuit, where it stops being global. Pass `check=True` to
    re-simulate and assert the prepared state is right.

## Pauli feature maps: the ones designed to be hard to simulate

`ZFeatureMap` is a product of single-qubit rotations, so it factorises and a classical
kernel can reproduce it. `ZZFeatureMap` adds entangling terms, and that is the point.

```python
import qmlkit as qk

print(qk.draw(qk.ZZFeatureMap(2, reps=1).build([0.4, 1.3])))
```

```text
q0: ─H──RZ(0.80)──@─────────────@──
q1: ─H──RZ(2.60)──X──RZ(10.10)──X──
```

The `Rz(10.10)` is the two-body term: the default data map sends a pair `(x₀, x₁)` to
`(π − x₀)(π − x₁)`, and the emitted angle is twice that. That factor of two is a
convention — PennyLane's `IQPEmbedding` uses the other one, which is documented in
[Validation](../about/validation.md).

```python
import qmlkit as qk

for name, fmap in [
    ("ZFeatureMap(3)", qk.ZFeatureMap(3)),
    ("ZZFeatureMap(3, reps=2)", qk.ZZFeatureMap(3, reps=2)),
    ("AngleFeatureMap(3)", qk.AngleFeatureMap(3)),
]:
    r = fmap.resources()
    print(f"{name:<26} depth {r['depth']:>3}  1q {r['n_1q']:>3}  2q {r['n_2q']:>3}")
```

```text
ZFeatureMap(3)             depth   4  1q  12  2q   0
ZZFeatureMap(3, reps=2)    depth  16  1q  16  2q   8
AngleFeatureMap(3)         depth   3  1q   3  2q   2
```

## Scale before you encode

An angle is periodic. Feed it raw features spanning `[0, 300]` and distinct inputs
collapse onto the same rotation — the model cannot tell them apart, and no amount of
training fixes it.

```python
import numpy as np
import qmlkit as qk

X = np.array([[0.0, 100.0], [50.0, 200.0], [100.0, 300.0]])
scaler = qk.AngleScaler().fit(X)

print(np.round(scaler.transform(X), 4))
```

```text
[[0.     0.    ]
 [3.1416 3.1416]
 [6.2832 6.2832]]
```

When you have more features than qubits, reduce first — `PCAReducer` is a plain SVD
with no scikit-learn dependency, and `reduce_to_qubits` wires the two together.

## Gradients flow through the encoding too

This is what makes a classical layer placed *before* the circuit trainable. Ask for it
with `trainable=True`:

```python
import numpy as np
import qmlkit as qk

x = np.array([0.3, 0.9])
spec = qk.angle_encode(x, trainable=True)

print("df/dx    =", np.round(qk.grad(spec, x, qk.Z(0) + qk.Z(1)), 6))
print("analytic =", np.round([-np.sin(0.3), -np.sin(0.9)], 6))
```

```text
df/dx    = [-0.29552  -0.783327]
analytic = [-0.29552  -0.783327]
```

Without this, a hybrid network silently freezes everything upstream of the circuit:
the loss still falls, because the quantum weights still train, so the bug looks like
slow convergence rather than a broken model. [Tutorial 5](05-training-torch.md)
returns to this.

## Choosing

| | Qubits | Depth | Use when |
|---|---|---|---|
| **Basis** | `n` bits | 1 | data is already binary |
| **Angle** | `n` features | 1 | the default — cheap, smooth, well-conditioned |
| **Amplitude** | `log₂ n` | `O(n)` | qubits are scarce and depth is not |
| **ZZ / Pauli** | `n` features | `O(reps·n)` | you want a kernel that is hard to reproduce classically |

---

**Next:** [Gradients](03-gradients.md) — six ways to get the same number, and two ways
to get a plausible wrong one.
