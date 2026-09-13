# qmlkit

**A quantum machine learning library built to refuse a plausible wrong number.**

In this field a mistake usually does not raise. A re-uploading model whose trainable
rotations commute with its encoding will build, bind, differentiate and train — and
reach one Fourier frequency instead of the eight you designed for. A quantum kernel
concentrates until every pair of points looks alike, and still returns a Gram matrix.
A quantum model beats its classical baseline by less than the spread between folds,
and gets written up as a result. All three run. All three return numbers in the right
range. All three are wrong.

So the parts of qmlkit that catch those are not an add-on:

```python
qk.plan(model)             # what the run costs in circuits, before you pay for it
qk.baseline(X, y)          # the classical bar, on identical folds, before you start
qk.diagnose(model)         # the failures that return a number instead of raising
qk.diagnose(model, X, y)   # ...and whether the quantum layer earned its place
qk.selfcheck(spec, theta)  # every exact gradient route, compared against every other
qk.progress()              # what the run is doing, and how much of it is left
```

That is one arc, in the order the questions arise: before, during, after. `plan` and
`baseline` run before you spend anything. `progress` is the only one of the six that
exists because of a question every library in this field gets asked and none of them
answers — *how much is left* — and it is the reason a kernel Gram matrix is no longer
a silent call that returns when it returns.

`selfcheck` is the one people skip and shouldn't. It computes your gradient by all four
exact routes — adjoint, backprop, Hadamard test, parameter-shift — and runs your
expectation through every installed SDK against the NumPy reference. Those four routes
share almost no code, so agreement to machine precision is strong evidence that all of
them are right, and disagreement localises the bug to a method or a backend rather than
leaving you with one number and no second opinion.

The same principle runs through the rest: `adjoint` refuses on a noisy backend rather
than silently differentiating a noiseless one, an unknown gate is refused by name rather
than approximated, a gate that declares no generator frequencies has its differentiation
*refused* rather than guessed, and `qk.baseline` will not call a lead inside the fold
spread a result.

Underneath that it is the ML layer quantum SDKs leave out — feature maps, an ansatz
vocabulary, quantum kernels, torch layers, and a general-purpose parameter-shift
gradient you can point at *any* circuit and observable — running unchanged on
**SpinQit**, **Qiskit**, **Cirq**, **PyTorch**, or the built-in NumPy reference, and on
two mixed-state backends when you want to ask what noise would have done. Seven in all,
behind one protocol: a backend supplies a statevector, and sampling, basis rotation,
qubit-wise-commuting grouping, expectation and the whole batch stack are derived once
in the base class — which is what makes agreement between backends a property rather
than a coincidence.

**Simulator-only** for the whole `0.x` line. Expectations are exact unless you ask for
shots, and two mixed-state backends take a noise model when you want to ask what a
device would have done. Exact here means no shot noise and no finite-difference bias —
not bit-identical arithmetic; expectations and gradients agree with the analytic value
to machine precision.

The core depends on **NumPy and nothing else** — so `pip install qmlkit` resolves in
seconds, picks no fight with the versions already in your environment, and works
somewhere with no wheel for a C++ simulator. CI enforces it by installing nothing else
in the `core` jobs, and by building a wheel and importing it in a clean virtualenv,
because an editable install imports from `src/` and would keep working even if a module
never made it into the package. That promise has been broken once, by a `scipy.special`
import that was present locally and absent in CI.

**[Documentation](https://ziadt160.github.io/qmlkit/)** — eight tutorials, guides and a generated API reference.
**[Case studies](https://ziadt160.github.io/qmlkit/studies/)** — eight whole problems, raw data to defensible number. In most of them the number is that the quantum model lost.
**[Validation](https://ziadt160.github.io/qmlkit/about/validation/)** — how any of this is known to be correct, including the reference that shares no code with the library it checks.
**[Coming from PennyLane](https://ziadt160.github.io/qmlkit/guides/from-pennylane/)** — borrow the three loops worth borrowing without migrating anything.
**[HANDOFF.md](https://github.com/Ziadt160/qmlkit/blob/main/HANDOFF.md)** — status, conventions, known traps, and what to do next, if you are picking this up.

## Two lines

```python
import qmlkit as qk

model = qk.VQC(n_features=4, n_classes=2).fit(X, y)
model.score(X, y)
```

That is a full hybrid quantum-classical classifier: angle encoding, a
hardware-efficient ansatz, exact gradients, and a torch training loop. Every
default in it is one keyword away from being something else.

## Three layers, and you pick where to stand

```python
# 1. a ready-made model
qk.VQC(n_features=4, n_classes=2).fit(X, y)

# 2. a torch layer, in any nn.Module you like
layer = qk.QuantumLayer(qk.ZZFeatureMap(4), qk.hardware_efficient(4, 2), [qk.Z(0), qk.Z(1)])
net   = nn.Sequential(nn.Linear(8, 4), nn.Tanh(), layer, nn.Linear(2, 2))

# 3. circuits and gradients directly
spec = qk.angle_encode([0.7])
qk.expectation(spec, qk.Z(0))        # 0.7648... == cos(0.7) to machine precision
qk.grad(spec, theta, qk.Z(0))        # exact, method chosen for you
```

Nothing at a higher layer hides a lower one. `VQC` is built from `QuantumLayer`,
which is built from the IR — and you can drop to any level without giving up what
the level above was doing for you.

## Built for research

Every extension point is a registry, and registering makes your thing a first-class
citizen everywhere the library takes that kind of argument.

```python
# a new ansatz -- one line, and it inherits correct gradients, resource counting,
# and a torch layer without opting into any of them
brick = qk.Ansatz(6, qk.repeat(3, qk.RotationLayer("ry")
                                  + qk.EntanglerLayer("cz", "alternating")))

# ...and registering makes it reachable by name
qk.register_ansatz("brick_wall", lambda n_qubits, n_layers=3: ...)   # qk.get_ansatz("brick_wall")
qk.register_gradient("my_estimator", fn)                             # method="my_estimator"
qk.register_backend("my_device", factory, requires="my_sdk")         # backend="my_device"

# a custom gate: declare the generator frequencies and parameter-shift works on it;
# add dmatrix and adjoint differentiation works too
qk.register_gate(qk.GateDef(
    "xy", n_qubits=2, n_params=1,
    matrix=lambda t: ...,
    frequencies=(1.0,),          # without this, differentiation is refused, not guessed
    dmatrix=lambda t: ...,       # optional: enables the fast adjoint path
))
```

| Want to change | Do this |
|---|---|
| The circuit structure | Compose `Block`s, or drop to `Custom(fn)` and write against the builder |
| A gate the library lacks | `register_gate` with its generator frequencies — parameter-shift then works on it |
| How gradients are estimated | `register_gradient`, then pass `method="yours"` anywhere |
| Where circuits run | `register_backend`, or subclass `Backend` and implement `statevector` |
| The data encoding | Subclass `FeatureMap`: `angles`, `n_angles`, `_emit` |
| How data is re-uploaded | Compose `EncodingLayer` with any block — see below |
| The training loop | Use `QuantumLayer` directly and write your own |

### Data re-uploading is a pattern, not a structure

Which encoding, which trainable block, what order, how much sharing — all of it is a
design choice, so none of it is hardcoded:

```python
# docs: run
fmap = qk.AngleFeatureMap(2, rotation="ry")

qk.reupload(fmap, n_layers=3)                      # S W S W S W
qk.reupload(fmap, n_layers=3, order="WS")          # vary before the first upload
qk.reupload(fmap, n_layers=3, share_weights=True)  # one tied block, reused
qk.reupload(fmap, n_layers=3, block=qk.RotationLayer(("rz", "ry")) + qk.EntanglerLayer("cz", "ring"))

# or compose directly — two different feature maps in one model
zz, angle = qk.ZZFeatureMap(2), qk.AngleFeatureMap(2, rotation="ry")
model = qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle))

model.n_inputs        # 5 — the ZZ map owns 3 slots, the angle map the 2 after them
model.angles([.3, .7])  # [0.6, 1.4, 13.876, 0.3, 0.7] — each map's own angles, in slot order
```

Each feature map owns a **disjoint** range of input slots, because a slot holds an
*angle* and every map derives its angles differently: a `ZZFeatureMap` Z term is
`Rz(2 x_i)`, an `AngleFeatureMap` rotation is `Ry(x_i)`. Sharing a slot between them
would feed one map's transformed angles to the other and encode the wrong number
without raising. `n_inputs` is inferred, so there is no count to get wrong; the same
map re-used keeps its own slots, which is what makes re-uploading feed the same data
in again rather than consume new features.

Any of these drops straight into a `QuantumLayer` — or into `VQC` directly, since a
re-uploading model is its own encoding *and* its own trainable block:

```python
qk.VQC(n_features=4, n_classes=2,
       feature_map=qk.reupload(qk.AngleFeatureMap(4), n_layers=3)).fit(X, y)
```

Input gradients included, so classical layers placed before the quantum one train.

> **One trap the library catches for you.** `L` uploads reach frequencies `0..L`
> only when the trainable block does **not** commute with the encoding rotation. If
> it does — `Ry(x) Ry(θ₁) Ry(x) Ry(θ₂) = Ry(2x + θ₁ + θ₂)` — the uploads merge into a
> single rotation, the model reaches one frequency, and every weight becomes a phase
> shift. It looks like a `3L`-parameter model; it is a one-parameter family.
> `reupload()` warns. Verify any model with `qmlkit.fourier.spectrum`.

## Built to be used by a model

Most code written against a library now is written by one, and a model does not read
a documentation site before it types. It guesses a name, runs it, reads the
traceback, and tries again. qmlkit treats that loop as the interface — which helps a
human on their first afternoon exactly as much.

**A wrong name answers with the right one.** The training data for any model holds
far more PennyLane and Qiskit than qmlkit, so the first guess is usually theirs:

```python
qk.AngleEmbedding
# AttributeError: module 'qmlkit' has no attribute 'AngleEmbedding'.
# 'AngleEmbedding' is PennyLane's name for qmlkit.AngleFeatureMap
# (or angle_encode(x) for a one-shot circuit).

qk.get_ansatz("hardware-efficient")
# KeyError: unknown ansatz 'hardware-efficient'. Did you mean 'hardware_efficient'?
# Valid: basic_entangler, hardware_efficient, mps, qaoa, ...
```

A translation, not an alias — the foreign name still raises, because an alias would
become API and would hide the semantic drift underneath it.

**`qk.diagnose(model)` catches what does not raise.** In this field a mistake usually
returns a plausible number rather than an exception. This model builds, binds,
differentiates and trains, and reaches one Fourier frequency instead of three:

```python
fmap  = qk.AngleFeatureMap(2, rotation="ry", entangle=False)
model = qk.Ansatz(2, qk.repeat(3, qk.EncodingLayer(fmap) + qk.RotationLayer("ry")))
print(qk.diagnose(model))
# [error] ENCODING_COMMUTES: 3 uploads, but every trainable rotation is 'ry', the
# same generator the encoding uses ... the model reaches 1 frequency rather than
# 0..3.  Fix: Use a non-commuting block, e.g. RotationLayer(('rz', 'ry', 'rz')).
```

It takes an ansatz, anything holding one, or a Gram matrix, and returns findings with
a stable code, the number measured, and the edit that fixes it. Empty means nothing
found, so `if qk.diagnose(model):` reads the way it should.

**The whole library in one fetch.** [`/llms.txt`](https://ziadt160.github.io/qmlkit/llms.txt)
indexes the site and the constraints that are not inferable from the API;
[`/llms-full.txt`](https://ziadt160.github.io/qmlkit/llms-full.txt) is every tutorial
and guide plus the entire public API with signatures. Both are generated from the
pages and the package, committed, and checked in CI, so neither can drift.

[Working with a coding agent](https://ziadt160.github.io/qmlkit/guides/agents/) is
the long version; [`AGENTS.md`](https://github.com/Ziadt160/qmlkit/blob/main/AGENTS.md) is for working *on* qmlkit.

## Checked against a reference that shares nothing

Everything below compares qmlkit against something that inherited its conventions —
five backends against each other, or PennyLane, whose names this library's gate table
was written by reading. Those all agree when a convention is wrong *everywhere*, and
once they did: the torch backend reimplemented `np.moveaxis` without numpy's
`sorted(zip(destination, source))`, so a two-qubit gate on descending wires permuted
the wires it was not acting on. `Z(3)` read `-0.1288` against `-0.7374`. Nothing
raised, `backprop` differentiated through it, and four references agreed with it.

`tests/densesim.py` is 206 lines that inherited none of it — hand-written gate
matrices, hand-derived derivatives, an explicit bit-index loop where the library uses
`tensordot`, its own product rule, reading `spec.ops` and calling no backend. It is
what found that bug, and four properties in `tests/test_torture.py` run it against
randomly generated circuits.

**When correctness actually matters, at least one reference must share no code and no
conventions with the thing being checked.**

## Cross-validated against PennyLane

A second, independently written implementation catches what an internal suite cannot.
`tests/test_pennylane_parity.py` is **301 executed parity cases** across every layer
both libraries implement, and it runs in CI like any other test:

```bash
pip install pennylane
pytest tests/test_pennylane_parity.py
```

| Layer | What is compared | Agreement |
|---|---|---|
| Gates | 55 matrix comparisons over all 20 gates — the 7 parametric at 6 angles each, the 13 constant once — plus 21 closed-form `dU/dθ` against a differenced PennyLane matrix | `1e-12` |
| Circuits | 40 **randomly generated** circuits over the full gate set, 1–5 qubits — statevectors, probabilities, and random multi-term observables | `1e-12` |
| Gradients | 5 ansätze × 4 observables; all four exact methods; PennyLane's own four methods back against ours; randomised circuits | `1e-10` |
| Encodings | angle (X/Y/Z), amplitude, basis, IQP | `1e-12` |
| Templates | `BasicEntanglerLayers`, `StronglyEntanglingLayers` | `1e-12` |
| Kernels | full Gram matrices, fidelity and swap-test estimators | `1e-10` |
| Quantum info | reduced DMs, von Neumann entropy, purity, mutual information, fidelity — over random states | `1e-10` |
| Fourier | re-uploading spectra at depths 1–4 | `1e-10` |
| Geometry | Fubini–Study metric (full and diagonal), QFIM | `1e-12` |
| Optimisers | Rotosolve and QNG trajectories, step by step | `1e-10` |

The randomised tests are the ones that matter. Hand-picked cases confirm what the
author already believed; a fuzzer explores the space, and every bug found in this
project so far has been of the plausible-wrong-number kind that only a second opinion
catches — including two in this library's own parameter-shift implementation, and two
in its own tests.

Beyond parity, `tests/test_torture.py` states **17 invariants that hold by mathematics
rather than by example** and lets Hypothesis hunt for circuits that break them: the four
exact gradient routes agree, a tied weight's gradient sums over its occurrences, batched
equals looped, `adjoint()` undoes, expectation lies inside the observable's spectrum. A
default run generates 1,825 cases; `QMLKIT_TORTURE_EXAMPLES=1500` before a release is
about 19,500 circuits. When one fails, Hypothesis shrinks it to the smallest circuit
that still shows it.

**Four real convention differences surfaced.** None is a bug in either library, and
each is pinned by its own test so it stays deliberate:

| Difference | Detail |
|---|---|
| IQP angle convention | PennyLane's `IQPEmbedding` emits `RZ(x_i)` / `MultiRZ(x_i x_j)`; qmlkit follows Qiskit and emits `Rz(2φ)`. Halving the data map reconciles them exactly |
| Amplitude encoding phase | qmlkit builds it from uniformly-controlled rotations and drops one overall factor. Unobservable — but it stops being global inside a *controlled* block, which the docstring warns about |
| Two-qubit "ring" | A ring on two qubits would revisit the same pair, so qmlkit collapses it to one `CX`; PennyLane's templates emit both `CNOT(0,1)` and `CNOT(1,0)` |
| `approx="block-diag"` | PennyLane blocks the metric by *layer* and zeroes cross-layer entries. qmlkit computes the exact metric — free on a simulator — so the same keyword does not port |

That last one is not just cosmetic. On a 3-qubit, 2-layer problem at equal step count
and step size, qmlkit's QNG reaches `-2.9999999` where PennyLane's default
`block-diag` QNG stalls at `-2.22`. Pointed at the exact metric (`approx=None`),
PennyLane's optimiser traces qmlkit's trajectory to `1e-8`.

One place qmlkit is measurably more accurate: `state_fidelity` hits the analytic
`|⟨a|b⟩|²` to `1e-16`, while `qml.math.fidelity` takes matrix square roots of rank-1
density matrices and loses about eight digits.

## Speed

`examples/benchmark_pennylane.py` times identical work on both libraries — against
PennyLane's **fastest** configuration, not its reference one. `pennylane-lightning`
ships with every PennyLane install, so `lightning.qubit` is always available, and
`qml.adjoint_metric_tensor` is an `O(P)` route sitting next to the `O(P²)` Hadamard-test
`qml.metric_tensor`. Benchmarking against the slow option when the fast one is one
string away would flatter the author.

| Operation | qmlkit | PennyLane (best) | | vs the naive route |
|---|---|---|---|---|
| Expectation, 12 qubits | 3.8 ms | 4.7 ms `lightning` | 1.2× | 3.0× |
| Gradient, 8 qubits, `P=96` | 10.9 ms | 10.8 ms `lightning-adjoint` | 1.01× *slower* | 6.0× |
| Parameter-shift, 6 qubits, `P=72` | 303 ms | 333 ms `lightning` | 1.1× | 3.7× |
| 20×20 kernel Gram matrix | 0.31 ms | 3.1 ms `broadcast` | **10×** | 654× |
| Exact metric tensor, `P=24` | 6.7 ms | 691 ms `adjoint_metric` | **103×** | 271× |

qmlkit is ahead on 14 of 14 cases, median **1.7×**.

**The comparison is deliberately the unflattering one.** `pennylane-lightning` is a
dependency of PennyLane, so the C++ `lightning.qubit` is in every install whether the
user asked for it or not; `qml.adjoint_metric_tensor` is an `O(P)` algorithm sitting
right beside the `O(P^2)` Hadamard-test route; and `default.qubit` *broadcasts* when it
is handed a stacked array, which turns a Gram matrix into one call rather than one per
pair. Timing against the slow option in any of those three would be timing an opponent
nobody runs. This file has done exactly that twice — it reported 6.1× before
`lightning` was used, and **69× on the Gram row before the broadcast path was**. The
naive column is still printed above, so the size of the difference stays visible rather
than being taken on trust. The 8-qubit gradient is a dead tie that falls either way
between machines, and the table quotes the run where it falls against qmlkit.

`examples/benchmark_pennylane.py` checks that both libraries produce the *same number*
before quoting any speedup. An acceleration that changes the answer is not an
acceleration.

**What that means.** The first three rows are dispatch and interpreter overhead rather
than arithmetic: qmlkit does less per call, so it leads at small register sizes and the
gap closes as `2ⁿ` starts to dominate — the 8-qubit gradient is a tie. Anyone quoting
the naive column as qmlkit's speed advantage is quoting the wrong number.

Two results are algorithmic, and those are the ones worth planning around. The **kernel
Gram matrix** costs one circuit per *row* rather than one per *pair*, because the
inversion test's `P(0…0)` is exactly `|⟨ψ(x′)|ψ(x)⟩|²` and a simulator can hand back the
state — linear in the dataset instead of quadratic, which is 10× against PennyLane
broadcasting and widens as the dataset grows (53× at 128 points). That shortcut needs a
statevector, so a device still pays the pairwise count and `circuits_on_hardware` reports
it. And the **metric tensor** is closed-form differentiation of the state, `P` derivative
states from one forward sweep, agreeing with PennyLane's own routes to `1.7e-16` and
*widening* with parameter count (50× at `P=12`, 103× at `P=24`) rather than narrowing.

JAX is not installed on the benchmark machine, so jit-compiled PennyLane is untested and
unclaimed; it would narrow the overhead rows further.

## Install

```bash
pip install qmlkit
```

Every SDK is optional and imported lazily, so `import qmlkit` never requires any
of them:

```bash
pip install "qmlkit[qiskit]"      # Qiskit backend
pip install "qmlkit[cirq]"        # Cirq backend
pip install "qmlkit[spinqit]"     # SpinQit — needs a Python 3.8-3.10 interpreter
```

SpinQit ships wheels for Python 3.8–3.10 only and pins `numpy<2`, so it sits
behind an environment marker and resolves cleanly to nothing on newer Pythons.

## Backends

```python
qk.backend_report()
# qmlkit backends:
#   [ok]      cirq
#   [ok]      cirq-density
#   [ok]      numpy
#   [ok]      qiskit
#   [ok]      qiskit-aer
#   [missing] spinqit  -> pip install 'qmlkit[spinqit]'
#   [ok]      torch

qk.expectation(spec, qk.Z(0), backend="qiskit")   # per call
qk.set_default_backend("spinqit")                  # for the session
```

`QMLKIT_BACKEND=cirq python train.py` switches an existing script without editing
it. Asking for a backend whose SDK is missing raises `BackendNotAvailable` with an
install command — never an `ImportError` traceback.

Each backend also exposes its native circuit, so you can draw, transpile or hand
it to that SDK's own tooling:

```python
qk.get_backend("qiskit").to_qiskit(spec).draw()
qk.get_backend("cirq").to_cirq(spec)
qk.get_backend("spinqit").to_spinqit(spec)
```

### Noise, when you ask for it by name

Two backends evolve a density matrix instead of a state, so a circuit can be run on a
simulator that makes mistakes:

```python
import cirq
backend = qk.get_backend("cirq-density", noise=cirq.depolarize(0.01))
qk.expectation(spec, qk.Z(0), backend=backend)
```

`qiskit-aer` does the same with a `qiskit_aer.noise.NoiseModel`, including one lifted
off real hardware with `NoiseModel.from_backend(...)`. Aer is a separate distribution
from Qiskit: `pip install 'qmlkit[aer]'`.

**Noise never picks a simulator for you.** `get_backend(noise=...)` without naming a
mixed-state backend raises and lists the ones that would work. A noisy run costs more,
refuses two of the gradient methods, and answers a different question — so which
simulator produced a number stays written down in the code that produced it.

**Two error sources, kept separate.** `shots=None` still means shot-free: the density
matrix is evolved exactly, so the answer is exact *given the noise model*. Decoherence
and sampling error both pull a number around, and studying one with the other layered
on top means never knowing which you are looking at. Ask for `shots=N` when you want
both — that is what a device gives you.

**What is refused, and why.** There is no statevector, so `adjoint` and `backprop`
decline: they would have differentiated a *noiseless* circuit and returned a
machine-precision gradient to someone asking about a noisy one. `parameter-shift` and
`grad_batch` work, because a shift rule never inspects a state.

With no noise model these backends reproduce the pure-state ones to machine precision
— the case that makes the noisy numbers trustworthy, and one the tests assert across
circuits, observables and both SDKs. Depolarizing noise is checked against the closed
form `cos(θ)(1 − 4p/3)`, not only against itself.

What noise does to trainability is the number worth knowing before the experiment. On
a 3-qubit, 3-layer ansatz the gradient *direction* survives — correlation above 0.99
with the noiseless one at `p = 0.05` — while the norm falls to 0.36 of it, and to 0.14
at `p = 0.1`. [The noise guide](https://ziadt160.github.io/qmlkit/guides/noise/) has
the table, the traps, and the boundary: error mitigation belongs to
[Mitiq](https://mitiq.readthedocs.io) and error correction to
[Stim](https://github.com/quantumlib/Stim).

### And circuits come back in

One-way interop is the difference between a library someone *tries* and one someone
*adopts*: an existing project has circuits already.

```python
qk.from_qasm(text)          # OpenQASM 2.0 -- standard library only, no extras needed
qk.from_qiskit(circuit)     # a QuantumCircuit, unbound Parameters included
qk.from_pennylane(qnode)    # a tape, QNode or quantum function; templates decompose
qk.from_cirq(circuit)       # a cirq.Circuit, sympy symbols included
```

`from_qasm` takes no dependency on anything: Qiskit, Cirq, Braket, t|ket> and Q# all
export QASM 2.0, so one stdlib parser reaches all of them. The other three exist
alongside it because QASM cannot carry a *free parameter*: `from_qiskit` maps unbound
`Parameter`s onto `ParamRef` in Qiskit's own order, and `from_cirq` does the same for
`sympy` symbols — `cirq.rx(2 * t)` arrives as `ParamRef(i, scale=2.0)`, since `ParamRef`
carries `scale * theta + offset` and that is exactly the linear form Cirq produces.

Cirq is the importer with nothing to look up: `cirq.S`, `cirq.T` and `cirq.rz` are all
a `ZPowGate`, separated only by exponent and `global_shift`, so it classifies rather
than reads a name. One asymmetry worth knowing: Cirq has no declared register, so a
qubit no operation touches is not in the circuit — the same logical circuit imports
two qubits wide from Qiskit and one from Cirq.

Qubit order is where importers actually break, so it is what the tests check:
`from_qiskit(to_qiskit(spec))` reproduces the **statevector** to `1e-12` across
randomly generated circuits, not merely the same list of gates. Qiskit and QASM are
little-endian and get flipped; PennyLane is big-endian like qmlkit and does not — and
that claim is asserted against PennyLane's own simulator rather than assumed.

A gate qmlkit has no definition for is refused by name, never approximated. The one
exception is the `u`/`u3` family, decomposed into rotations with a warning that an
overall phase was dropped — unobservable alone, observable inside a controlled block.

### Why the translations are trustworthy

`tests/test_cross_backend.py` runs the same circuit zoo through every installed
backend and asserts agreement with the NumPy reference on statevectors,
probabilities, expectations over X/Y/Z and two-body terms, seeded sampling, and
parameter-shift gradients. The zoo deliberately targets where SDKs differ —
endianness, controlled-gate qubit order, idle qubits, basis rotations.

Three findings from building it, all now handled:

| Finding | Handling |
|---|---|
| Qiskit is little-endian; qmlkit is big-endian | qmlkit qubit `i` maps to Qiskit qubit `n-1-i` at build time, so the index conventions coincide and no vector reversal is needed |
| Cirq silently drops qubits a circuit never touches | an explicit `qubit_order` is always passed |
| **SpinQit's `CY` applies `-iY`**, not `Y`, to the control-1 subspace | emitted as `Sd·CX·S` instead. This is a *relative* phase between control branches, so it changes measurement statistics — not a harmless global phase. SpinQit's single-qubit `Y` is correct |

SpinQit's simulator also carries a precision floor near `1e-10` rather than machine
precision, so it is compared at a looser tolerance. `verify_conventions()` re-checks
bit order and gate definitions against a live install in one call.

## What it does today

- **A backend-neutral circuit IR.** A circuit is data — a list of `Op`. Backends
  compile it; gradients, resource counting and drawing all read it.
- **Seven backends behind one protocol** — five pure-state (NumPy exact
  reference, SpinQit, Qiskit, Cirq, Torch) and two mixed-state (`cirq-density`,
  `qiskit-aer`) — with a cross-backend equivalence suite proving they agree.
  A backend supplies a statevector; sampling, basis rotation, qubit-wise-commuting
  grouping, expectation and the whole batch stack are defined once in the base
  class, which is what makes agreement between backends a property rather than a
  coincidence.
- **Pauli observables** — `Z(0)`, `ZZ(0, 1)`, weighted sums — with one
  `expectation()` that is correct for any register width.
- **The full encoding layer** — basis, angle, amplitude, Hamiltonian and data
  re-uploading, plus the `Z`/`ZZ`/`Pauli` feature-map family.
- **Exact parameter-shift gradients**, including the two cases that are easy to
  get silently wrong (see below).
- **Shot-budget arithmetic** — standard errors, and the shots needed for a target
  precision.
- **An ansatz vocabulary** — `RotationLayer`, `EntanglerLayer`, `PoolLayer`,
  `repeat`, `share` — plus a zoo (hardware-efficient, strongly-entangling,
  two-design, QAOA, TTN, MPS, QCNN) that is written *in* that vocabulary.
- **Six gradient methods** behind one `grad()`: adjoint and backprop (exact, one
  pass), parameter-shift and Hadamard-test (exact, hardware-valid), SPSA (two
  evaluations, any `P`), and finite differences for debugging.
- **A PyTorch bridge** — `QuantumLayer`, `VQC`, `VQRegressor` — where inputs get
  gradients, so classical layers placed *before* the quantum one actually train.
- **Quantum kernels** — fidelity / swap / Hadamard estimators, Gram matrices with PSD
  repair, `QSVC`/`QSVR`, trainable kernels, and projected kernels that survive the
  concentration that kills the fidelity kernel at width.
- **Structured models** — `QCNNLayer`, `MPSLayer`, `QLSTM`, `DressedQuantumNet`; and
  generative ones — `QCBM`, `QGAN`, `QuantumBoltzmannMachine`, `QuantumHopfield`.
- **Analysis** — expressibility, Meyer–Wallach entanglement, barren-plateau scans,
  effective dimension, Fourier spectra, and `draw()` / `specs()`.
- **Diagnostics** — `diagnose()` for the circuit (dead parameters, collapsed
  re-uploading, concentrated kernels, flat gradients) and `selfcheck()` for the number
  (every exact gradient route compared against every other, every backend against the
  reference).
- **Evaluation** — `qk.evaluate` returns every metric a task needs at once and says
  when one is misleading; `qk.imbalance` handles the loss and the split that skew
  breaks; `qk.baseline` runs the classical bar on identical folds and refuses to call
  a lead inside the fold spread a result.
- **Budget and provenance** — `qk.plan()` costs a run in circuits before it starts;
  `qk.fingerprint()` records the versions, backend and seed that produced a number.
- **Circuit import** — `from_qasm` (standard library only), `from_qiskit` (unbound
  parameters included), `from_pennylane` (templates decompose) and `from_cirq`
  (`sympy` symbols carried onto `ParamRef`), so circuits come back in as well as out.
- **Noise, named explicitly** — `cirq-density` and `qiskit-aer` evolve a density
  matrix and take a noise model. `shots=None` still means shot-free, so decoherence
  and sampling error stay separable; the gradients that need a pure state refuse.
- **Batched execution** — one circuit at many parameter vectors in one pass. A
  compute-uncompute kernel and a training batch are both *one circuit structure at many
  angle vectors*, which is what makes this possible: **20.4× on a full training step**
  at 4 qubits, **176× on a 20×20 Gram matrix**. The batched kernel is a BLAS `gemm`,
  not an `einsum` — `np.einsum` without `optimize=` never reaches BLAS, which cost 5×
  here until it was measured. Batching is switched off above a measured crossover
  (`NumpyBackend.batch_max_qubits`, 11) rather than assumed to help.
- **A Gram matrix costs one circuit per row**, not one per pair: the inversion test's
  `P(0…0)` *is* `|⟨ψ(x′)|ψ(x)⟩|²`, and a simulator can hand back the state. Linear
  rather than quadratic in the dataset. It is a simulator-only shortcut, so the count
  stays two numbers — `n_evaluations` for what ran, `circuits_on_hardware` for what a
  device would pay.
- **Watching a run** — `qk.progress()` gives a live line with an honest ETA, and
  `run.save_html()` writes the whole run out as one self-contained page. See
  [Watching a run](https://ziadt160.github.io/qmlkit/guides/watching-a-run/).

## Algorithms

Each one is a *loop* over machinery that already exists — so each is thin, and every
structural choice it makes is an argument you pass rather than something baked in. All
of them share one open `OPTIMIZERS` dict (`rotosolve`, `spsa`, `gradient-descent`,
`adam`), and a custom optimiser is a function, not an adapter class.

| | |
|---|---|
| **`VQE`** | with `ADAPT-VQE`, which grows the ansatz one operator at a time |
| **`QAOA`** | returns the *bitstring*, its probability, the cut value and the approximation ratio — not just an energy |
| **Chemistry** | `molecular_hamiltonian`, `h2_curve`, and a real restricted Hartree–Fock loop with STO-3G integrals computed here, so the core stays NumPy-only |
| **`QuantumAutoencoder`** | trained on trash *fidelity*, not trash purity — see below |
| **`QMeans`** | Lloyd's algorithm with a quantum kernel distance, deliberately unchanged otherwise |
| **`QuantumPolicy`** | REINFORCE with exact circuit gradients, no finite differences |

Three things in there are worth knowing before you need them.

**`VQE`, `QAOA` and `AdaptVQE` check themselves against dense diagonalisation** whenever
it is affordable (12 qubits or fewer, by default) and report `error_vs_exact`. A
variational algorithm that converges confidently on the wrong energy is the normal
failure, not the exotic one.

**`QAOA` warns when Rotosolve is invalid for the circuit it just built.** Rotosolve
assumes each angle drives a single sinusoid; QAOA's cost angle drives one `rz` per
edge, which measures as five frequencies on a five-edge problem. It converges
immediately, on the wrong point, and reports it as a result.

**ADAPT ships two operator pools, and the general one is wrong for chemistry.** A
molecular Hamiltonian conserves particle number, so any generator that does not has
*exactly zero* gradient at Hartree–Fock — measured on H₂, every operator in the default
pool scores `0.00e+00`, ADAPT correctly concludes nothing helps, and hands back an
empty circuit. Use `chemistry_operator_pool`. That is physics, not a bug, and a test
pins it.

## Gradients

```python
qk.grad(spec, theta, obs)                        # auto: adjoint when it can, shift when it can't
qk.grad(spec, theta, obs, method="parameter-shift", shots=4096)
```

| Method | Cost | Exact | On hardware |
|---|---|---|---|
| `adjoint` | one backward pass | yes | no — needs the statevector |
| `backprop` | one autograd pass (torch) | yes | no — needs the statevector |
| `hadamard` | `P` circuits, one ancilla | yes | yes, if the ancilla can reach every wire |
| `parameter-shift` | `2P` circuits (more for multi-frequency gates) | yes | yes |
| `spsa` | 2 evaluations, any `P` | no — unbiased estimate | yes |
| `finite-diff` | `2P` | no — `O(h²)` bias | debugging only |

All four exact methods agree to machine precision — they are four independent
routes to the same number, which is exactly why disagreement between them is a
useful bug detector. Measured on a 5-qubit hardware-efficient ansatz with a
two-term observable:

| `P` | `adjoint` | `backprop` | `hadamard` | `parameter-shift` | `finite-diff` |
|---|---|---|---|---|---|
| 20 | **2.2 ms** | 8.5 ms | 15 ms | 28 ms | 29 ms |
| 60 | **6.2 ms** | 24 ms | 109 ms | 213 ms | 226 ms |
| 120 | **12.6 ms** | 50 ms | 404 ms | 823 ms | 870 ms |

Adjoint is the default on a simulator because its cost does not grow with `P` —
**65× faster than parameter-shift at `P=120`**, and the gap widens from there.
`hadamard` halves the circuit count against parameter-shift, and on a simulator
that shows up as roughly half the wall-clock too; the trade is an ancilla that
must couple to every wire the generator touches. On real hardware that routing
cost usually eats the saving, which is why parameter-shift stays the default
there. `backprop` needs `pip install 'qmlkit[torch]'` and exists mainly so a
circuit can sit inside an autograd graph — for a standalone gradient, adjoint is
both faster and lighter on memory.

Second derivatives come from differencing the *exact* gradient, so only the outer
derivative is approximate:

```python
qk.hessian(spec, theta, obs)               # (P, P), symmetric
qk.gradient_cost(spec, "parameter-shift")  # circuits one gradient would cost
```

## The parameter-shift rule, done properly

Two things about parameter-shift produce a *plausible wrong number* rather than an
exception. Both are handled here, and both have tests.

**Shift rules belong to the gate, not the call.** A gate's rule is determined by
the unique positive gaps between its generator's eigenvalues. `ry` has one
frequency (the familiar ±π/2, ±½ rule); `crz` has two and needs four terms. A
circuit mixing them needs both, looked up per gate:

```python
qc = qk.QCircuit(2)
qc.ry(0, qk.ParamRef(0))
qc.crz(0, 1, qk.ParamRef(1))
spec = qc.to_spec()

qk.grad_circuit_cost(spec)     # 6, not 2*2 -- the CRZ costs four evaluations
qk.param_shift_grad_circuit(spec, theta, qk.Z(1))
```

Rules are **derived**, not transcribed: declare a gate's `frequencies` and the
right rule is solved for. A gate with no declared frequencies is refused rather
than differentiated incorrectly.

**Shared parameters shift one occurrence at a time.** When one logical parameter
drives several gates — weight tying, as in a QCNN's shared convolution block — the
derivative is the *sum over occurrences*, each shifted on its own. Shifting them
together computes something else entirely:

```python
qc = qk.QCircuit(3)
shared = qc.param()
qc.rotation_layer(("ry",), shared=shared)   # one parameter, three gates
spec = qc.to_spec()

len(spec.occurrences_of(0))    # 3
qk.param_shift_grad_circuit(spec, [0.7], qk.Z(0) + qk.Z(1) + qk.Z(2))
```

**Gradients flow through the encoding too.** `∂f/∂x` is available from the same
rule, which is what lets a classical pre-net in a hybrid stack actually train:

```python
spec = qk.angle_encode([0.4, 1.1], trainable=True)
qk.param_shift_grad_circuit(spec, [0.4, 1.1], qk.Z(0))   # -> [-sin(0.4), 0]
```

## Encoding

```python
qk.angle_encode([0.3, 1.1])              # one feature per qubit
qk.amplitude_encode([1, 2, 3, 4])        # 2**n numbers in n qubits
qk.hamiltonian_encode(x, t=1.0, steps=3) # data-dependent Ising evolution

fm = qk.ZZFeatureMap(n_features=3, reps=2)
fm.build(x)                               # U(x)
fm.adjoint(x)                             # U(x)^dagger -- the other half of a kernel
```

The compute-uncompute kernel falls straight out of `adjoint()`, and matches the
exact overlap to machine precision:

```python
k = qk.probabilities(fm.build(x).compose(fm.adjoint(xp)))[0]   # P(all zeros) IS k(x, x')
```

**`PauliFeatureMap` is built from its two pieces, and both are public.** A Pauli
feature map needs a basis change that diagonalises each string and a data map that
turns features into angles; they are usually left implicit inside one function. Here
they are `basis_change` and `default_data_map`, so either can be replaced without
rewriting the map — and the maps are tested against the analytic kernels they are
supposed to induce rather than against themselves: the angle map's `cos²((x−x')/2)`,
the Z map's factorisation, and the ZZ map's failure to factorise.

**Amplitude encoding is built from uniformly-controlled rotations**, not a backend
state-preparation primitive. So it emits only `ry`/`rz`/`cx`, runs identically on
every backend, and its exponential gate cost is visible rather than hidden inside
an SDK call.

```python
qk.reduce_to_qubits(X, n_qubits=3)   # PCA (plain SVD) + rescale into [0, 2pi)
qk.AngleScaler().fit(X_train).transform(X_test)   # one range for both splits
```

## Shots are opt-in

`shots=None` (the default) returns the exact expectation. Pass `shots=N` to model
a real device, and ask for the uncertainty alongside the value:

```python
value, err = qk.expectation(spec, qk.Z(0), shots=4096, return_std=True)
qk.shots_for_precision(0.01)      # what a target precision actually costs
```

## Roadmap

| Phase | Status |
|---|---|
| 0 · IR, NumPy backend, observables, execution, parameter-shift | **done** |
| 1 · SpinQit, Qiskit and Cirq backends + cross-backend suite | **done** |
| 2 · Encodings and feature maps (Z, ZZ, Pauli, re-uploading) | **done** |
| 3 · Gradients: adjoint, SPSA, dispatch registry | **done** |
| 3c · Hadamard-test and backprop gradients, `hessian`, `gradient_cost` | **done** |
| 3b · Ansatz vocabulary, zoo, and registry | **done** |
| 4 · Torch bridge: `QuantumLayer`, `VQC`, `VQRegressor` | **done** |
| 5 · Quantum kernels, `QSVC`/`QSVR` | **done** |
| 6 · QCNN, QLSTM, MPS; QCBM, qGAN, QBM | **done** |
| 7 · Docs, tutorials, PyPI | **done** — `pip install qmlkit` |

Beyond `0.1.0`, in the order they are likely to matter: a *mitigation verdict* (whether error mitigation improved an
estimate or only traded bias for variance, on identical seeds with the shot cost
stated — the implementations belong to [Mitiq](https://mitiq.readthedocs.io)),
gradients through a noise channel, and batched/async submission for a real device.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

Runnable examples, none of which quotes a number it did not compute:

```bash
python examples/quickstart.py            # every layer of the library, end to end
python examples/credit_risk.py           # a real dataset, start to finish, with every
                                         #   decision made by a diagnostic
python examples/accelerate_pennylane.py  # borrow three inner loops, migrate nothing
python examples/compare_pennylane.py     # readable cross-check against PennyLane
python examples/benchmark_pennylane.py   # wall-clock, same work on both sides
```

`credit_risk.py` is the one to read if you want to know what the library is *for*. It
takes 32,581 loan applications, works through the skew, the classical bar, the circuit
budget, an ansatz chosen from measured expressibility and gradient variance, and a
kernel that is diagnosed rather than fitted — and it ends by reporting that the
quantum model **lost**, on the same folds and the same inputs. A negative result you
can defend is worth more than a positive one you cannot.

The exhaustive version of the second one lives in the test suite, so it guards every
future change rather than only today's:

```bash
pytest tests/test_pennylane_parity.py    # 301 parity cases
```

Before a release, verify the *built* artifact rather than the source tree — an
editable install imports out of `src/` and keeps working even if a module never made
it into the wheel:

```bash
python -m build && python -m venv /tmp/clean && /tmp/clean/bin/pip install dist/qmlkit-*.whl && /tmp/clean/bin/python scripts/verify_install.py
```

[`RELEASING.md`](https://github.com/Ziadt160/qmlkit/blob/main/RELEASING.md) has the rest of the process.

## License

Apache-2.0 · © 2026 Ziad Tarek Mohammed
