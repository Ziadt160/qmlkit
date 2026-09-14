# Coming from PennyLane

You do not have to migrate anything to use this library, and the cheapest way to start
is not to.

## Don't migrate. Borrow the loops that are worth borrowing

`qk.from_pennylane` reads a tape, a QNode or a plain quantum function — templates
decompose on the way in — so a PennyLane circuit can be handed to qmlkit for the parts
where qmlkit is faster and left exactly where it is for everything else.

```python
# docs: skip
import pennylane as qml
import qmlkit as qk

@qml.qnode(qml.device("default.qubit", wires=3))
def circuit(x):
    qml.AngleEmbedding(x, wires=range(3))
    qml.StronglyEntanglingLayers(weights, wires=range(3))
    return qml.expval(qml.PauliZ(0))

spec = qk.from_pennylane(circuit)     # nothing above this line changes
```

Three inner loops are worth stealing, and `examples/accelerate_pennylane.py` runs all
three against PennyLane's own answer before quoting a number:

| Loop | Why it moves | Measured |
|---|---|---|
| **Kernel Gram matrix** | the inversion test's `P(0…0)` *is* the state overlap, so a statevector backend runs one circuit per row and takes one matrix product — linear in the dataset rather than quadratic | **17×** at 20 points, and it widens with the dataset |
| **Batched gradients** | a training batch is the same circuit at one parameter vector per sample, and the batched kernel is a BLAS `gemm` | **5.4×** on a batch of 32 |
| **Fubini–Study metric** | closed-form differentiation of the state against `O(P)` adjoint or `O(P²)` Hadamard tests | **50×** at `P=12`, and it *widens* with parameter count |

All three are algorithmic rather than overhead, which is what makes them worth
stealing: each agrees with PennyLane's own answer to ~`1e-16`, and each gets *better*
the bigger the problem is.

**Every PennyLane column there is its fastest route, and getting that right changed two
of the three numbers.** The Gram row is timed against `default.qubit` *broadcasting* a
stacked array rather than one QNode call per pair, and the gradient row against a single
broadcast backward pass rather than a loop over `lightning.qubit`. Timed the naive way
those rows read 69× and 41×; they are 17× and 5.4× against what PennyLane can actually
do. A loop of QNode calls is what a newcomer writes, not the library's speed.

One more caveat: the state-overlap shortcut needs a statevector, so a *device* still
pays the pairwise count. `QuantumKernel.circuits_on_hardware` reports that number, and
it is the one to budget a hardware run from.

Everywhere else the gap is 1.0–1.2×. See [Validation](../about/validation.md) for the
whole table, including the row where PennyLane wins.

## When you guess the wrong name, the error tells you the right one

Most code written against this library is written by a model, and the training data
holds far more PennyLane than qmlkit — so the first guess is usually PennyLane's. Sixty
foreign names are answered with the qmlkit one:

```pycon
>>> qk.AngleEmbedding
AttributeError: module 'qmlkit' has no attribute 'AngleEmbedding'.
'AngleEmbedding' is PennyLane's name for qmlkit.AngleFeatureMap
(or angle_encode(x) for a one-shot circuit).
```

**It is a translation, not an alias — the foreign name still raises.** An alias would
become API, and it would hide the drift underneath: `qml.expval` takes a QNode,
`qk.expectation` takes a `CircuitSpec` and an observable. A name that silently resolved
would fail later, further from its cause, with a worse message.

A rough map of the ones people reach for first:

| PennyLane | qmlkit |
|---|---|
| `qml.device` | `qk.get_backend` |
| `qml.probs` / `qml.state` / `qml.sample` | `qk.probabilities` / `qk.statevector` / `qk.run_counts` |
| `qml.jacobian` / `qml.grad` | `qk.grad` |
| `AngleEmbedding` / `AmplitudeEmbedding` / `BasisEmbedding` | `AngleFeatureMap` / `amplitude_encode` / `basis_encode` |
| `StronglyEntanglingLayers` / `BasicEntanglerLayers` | `strongly_entangling` / `basic_entangler` |
| `SimplifiedTwoDesign` / `RandomLayers` / `MPS` / `TTN` | `simplified_two_design` / `random_layers` / `mps_ansatz` / `tree_tensor_network` |
| `qml.math.reduced_dm` | `qk.reduced_dm` |
| `qml.about()` | `qk.backend_report()` |

## Four places the two libraries genuinely differ

None is a bug in either, each is pinned by its own parity test so it stays deliberate,
and every one of them would be very hard to spot by eye. The full account is in
[Validation](../about/validation.md#four-convention-differences); the short version:

**IQP angle convention.** PennyLane's `IQPEmbedding` emits `RZ(xᵢ)`; qmlkit follows
Qiskit and emits `Rz(2φ)`. A kernel differing by exactly this factor looks fine.

**Amplitude-encoding phase.** qmlkit builds it from uniformly controlled rotations and
drops one overall factor — unobservable alone, observable inside a *controlled* block.

**Two-qubit "ring".** A ring on two qubits would revisit the same pair, so qmlkit emits
one `CX` where PennyLane's templates emit both directions. On two qubits these are
genuinely different circuits.

**`approx="block-diag"` does not port.** PennyLane blocks the metric tensor by circuit
layer and zeroes every cross-layer entry; qmlkit computes the exact metric, which costs
nothing on a simulator. This one is not cosmetic: on a 3-qubit, 2-layer problem at equal
step count and step size, qmlkit's QNG reaches `−2.9999999` where PennyLane's default
`block-diag` QNG stalls at `−2.22`. Pointed at the exact metric (`approx=None`),
PennyLane traces qmlkit's trajectory to `1e-8`.

## What PennyLane does that qmlkit does not

Worth knowing before you commit to anything.

- **Hardware.** qmlkit is **simulator-only for the whole `0.x` line**. If you need to
  run on a device, you need PennyLane.
- **Gradients through a noise channel.** `default.mixed` backpropagates through the
  channel; qmlkit's density-matrix backends offer parameter-shift only. This is the
  clearest place PennyLane is ahead.
- **JAX and `jit`.** Untested here and unclaimed — the benchmark machine has no JAX.
- **Catalyst, pulse-level control, plugin breadth, quantum chemistry depth.** Not
  contested.

## A dead parameter reads exactly like a fully collapsed plateau

The standard barren-plateau recipe is to fix a parameter index and watch
`Var[dC/dtheta_k]` as the register grows. It has a trap, and the trap does not raise.

Some indices are *structurally* silent for a given circuit and observable — the
gradient is exactly zero, not small — and which index that is moves with the depth and
with the cost you measure. Probe one of those and the number that comes back is
machine zero, which reads as the most collapsed plateau imaginable and is nothing of
the kind:

```python
import qmlkit as qk

n = 6
glob = qk.Z(0)
for i in range(1, n):
    glob = glob * qk.Z(i)

ansatz = qk.hardware_efficient(n, 1)
stats = qk.gradient_stats(ansatz, glob, n_samples=50)

print(f"parameter 0:         {stats.variance[0]:.2e}")
print(f"best live parameter: {stats.best_variance:.2e}")
print(f"silent parameters:   {stats.silent.size} of {stats.n_params}")
```

Thirty-two orders of magnitude between "this model is dead" and "this model is fine",
decided by which index you happened to type. `qk.gradient_variance` carries a warning
for exactly this case, and `qk.gradient_stats` sidesteps it by returning every
parameter at once — one adjoint pass already computes the whole vector, so the other
`p - 1` entries are free and they are what tells a plateau from a dead parameter.

There is no equivalent warning in PennyLane, and none is implied: `qml.grad` is a
differentiation primitive and an exact zero is the correct answer to what it was
asked. The difference is that qmlkit ships the diagnosis on top, because this is the
mistake the barren-plateau literature is easiest to reproduce wrongly with.

## What you get in exchange

The layer that answers *is this result real*, which is the reason the library exists:

```python
# docs: skip
qk.plan(model)             # what the run costs in circuits, before you pay for it
qk.baseline(X, y)          # the classical bar, on identical folds
qk.diagnose(model)         # the failures that return a number instead of raising
qk.diagnose(model, X, y)   # whether the quantum layer earned its place
qk.selfcheck(spec, theta)  # every exact gradient route against every other
qk.progress()              # how much of the run is left
```

## If you do want to move a whole project

```python
# docs: skip
qk.from_pennylane(qnode)    # a tape, QNode or quantum function; templates decompose
qk.from_qasm(text)          # standard library only, no extras needed
qk.from_qiskit(circuit)     # unbound Parameters carried onto ParamRef
qk.from_cirq(circuit)       # sympy symbols carried onto ParamRef
```

Qubit order is where importers actually break, so it is what the tests check:
`from_qiskit(to_qiskit(spec))` reproduces the **statevector** to `1e-12` over randomly
generated circuits, not merely the same list of gates. Qiskit and QASM are little-endian
and get flipped; **PennyLane is big-endian like qmlkit and does not** — and that claim is
asserted against PennyLane's own simulator rather than assumed.

A gate qmlkit has no definition for is refused by name, never approximated. The one
exception is the `u`/`u3` family, decomposed into rotations with a warning that an
overall phase was dropped.
