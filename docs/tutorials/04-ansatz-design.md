# 4. Designing an ansatz

Most libraries give you a list of templates and a note saying "or write your own",
where writing your own means implementing parameter counting, gradients and resource
estimation from scratch. Here an ansatz is an expression in a small vocabulary, and
anything you build in it inherits correct gradients, resource counting and a PyTorch
layer without opting into any of them.

## The vocabulary

Four block types, composed with `+`, `repeat` and `share`.

| | |
|---|---|
| `RotationLayer("ry", "rz")` | one rotation per qubit, per named axis |
| `EntanglerLayer("cz", "ring")` | a fixed two-qubit gate over a named pattern |
| `ParametricEntangler("crz", "chain")` | the same, but the entangler carries an angle |
| `PoolLayer(...)` | measure-and-discard, for QCNN-shaped circuits |

Patterns are `chain`, `ring`, `full`, `alternating`.

## One line

```python
import qmlkit as qk

brick = qk.Ansatz(
    4,
    qk.repeat(2, qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "alternating")),
    "brick_wall",
)
print(qk.draw(brick.build()))
print(brick)
```

```text
q0: ─RY(θ0)──@──RY(θ4)──────────@─────
q1: ─RY(θ1)──Z────@─────RY(θ5)──Z──@──
q2: ─RY(θ2)──@────Z─────RY(θ6)──@──Z──
q3: ─RY(θ3)──Z──RY(θ7)──────────Z─────
Ansatz('brick_wall', n_qubits=4, n_params=8)
```

Nothing declared the parameter count. It is **inferred** from a dry build, so
miscounting is not a failure mode — a whole category of bug that simply cannot occur.

## It is already a first-class citizen

```python
import numpy as np
import qmlkit as qk

brick = qk.Ansatz(
    4,
    qk.repeat(2, qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "alternating")),
    "brick_wall",
)
spec, theta = brick.build(), brick.init(seed=0)

print("gradient:", np.round(qk.grad(spec, theta, qk.Z(0)), 5))
print("resources:", brick.resources()["depth"], "depth,", brick.resources()["n_2q"], "two-qubit gates")
```

## Measure it, do not assert about it

`AnsatzReport` runs the diagnostics that actually distinguish ansätze:

```python
import qmlkit as qk

print(qk.metrics.AnsatzReport(qk.hardware_efficient(4, 2), n_samples=300))
```

```text
hardware_efficient on 4 qubits
  parameters            16
  depth                 9
  two-qubit gates       6
  gradient circuits     32
  expressibility        0.0736   (KL from Haar,
                                 lower is more expressive)
  entangling capability 0.7254   (Meyer-Wallach Q)
  gradient variance     3.096e-01   (higher = more trainable)
```

**Expressibility** is the KL divergence between the distribution of fidelities the
ansatz produces and the Haar distribution — lower means closer to covering the space
uniformly. **Entangling capability** is the mean Meyer–Wallach measure. Neither is
"good" on its own; they trade against each other and against trainability.

## Comparing candidates

```python
import qmlkit as qk

brick = qk.Ansatz(
    4,
    qk.repeat(2, qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "alternating")),
    "brick_wall",
)

print(f"{'ansatz':<24}{'params':>7}{'depth':>7}{'2q':>5}{'expr':>10}{'entang':>9}")
candidates = ["hardware_efficient", "strongly_entangling", "tree_tensor_network", "mps", "qcnn"]
for name in candidates:
    r = qk.metrics.AnsatzReport(qk.get_ansatz(name, n_qubits=4), n_samples=300).results
    print(f"{name:<24}{r['n_params']:>7}{r['depth']:>7}{r['n_2q']:>5}"
          f"{r['expressibility']:>10.4f}{r['entangling_capability']:>9.4f}")
r = qk.metrics.AnsatzReport(brick, n_samples=300).results
print(f"{'brick_wall (ours)':<24}{r['n_params']:>7}{r['depth']:>7}{r['n_2q']:>5}"
      f"{r['expressibility']:>10.4f}{r['entangling_capability']:>9.4f}")
```

```text
ansatz                   params  depth   2q      expr   entang
hardware_efficient           16      9    6    0.0736   0.7254
strongly_entangling          24     14    8    0.0602   0.8495
tree_tensor_network           6      4    3    0.7550   0.3581
mps                           6      6    3    0.4550   0.4029
qcnn                          4      8    4    0.6075   0.4259
brick_wall (ours)             8      6    6    0.4165   0.4504
```

Read the trade honestly: `strongly_entangling` is the most expressive and the most
entangling, and it costs 24 parameters and depth 14. `tree_tensor_network` is the
least expressive by a wide margin and costs 6 parameters at depth 4. Expressibility
is not free, and — as [tutorial 8](08-trainability.md) shows — it is not always what
you want.

## Weight tying, and why it is a first-class idea

`share` makes several applications of a block use the **same** parameters. That is
what makes a QCNN convolutional rather than merely deep:

```python
import qmlkit as qk

qcnn = qk.get_ansatz("qcnn", n_qubits=8)
spec = qcnn.build()

print(f"logical parameters: {spec.n_params}")
print(f"angle slots:        {len(spec.slots())}")
print(f"gradient circuits:  {qk.gradient_cost(spec, 'parameter-shift')}")
```

```text
logical parameters: 6
angle slots:        22
gradient circuits:  44
```

Six free parameters filling twenty-two slots. The gradient cost scales with the
*logical* parameter count, not the slot count — which is the real advantage of a
convolutional ansatz, and the reason [tutorial 3](03-gradients.md) made such a fuss
about summing over occurrences.

## Initialisation matters more than it looks

```python
import qmlkit as qk

ansatz = qk.hardware_efficient(3, 2)
for strategy in ("small", "uniform", "zeros"):
    theta = ansatz.init(strategy, seed=0)
    print(f"{strategy:<9} mean {theta.mean():+.4f}  std {theta.std():.4f}")
```

```text
small     mean +0.0022  std 0.0704
uniform   mean +0.1695  std 2.1294
zeros     mean +0.0000  std 0.0000
```

`small` is the default and it is not arbitrary: near-identity initialisation keeps
the circuit shallow in effect at the start, which is one of the few reliable defences
against barren plateaus. `zeros` is worse than it looks — a symmetric starting point
can leave whole parameter groups with identical gradients forever.

## Registering it

```python
import qmlkit as qk

@qk.register_ansatz("brick_wall_demo")
def brick_wall(n_qubits, n_layers=2):
    return qk.Ansatz(
        n_qubits,
        qk.repeat(n_layers, qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "alternating")),
        "brick_wall_demo",
    )

print(qk.get_ansatz("brick_wall_demo", n_qubits=3, n_layers=1))
```

```text
Ansatz('brick_wall_demo', n_qubits=3, n_params=3)
```

It is now reachable by name anywhere the library takes one, including
`AnsatzReport`, `QuantumLayer` and `compare_ansatze`.

---

**Next:** [Training with PyTorch](05-training-torch.md) — putting the ansatz in a
network that actually learns.
