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
**SpinQit**, **Qiskit**, **Cirq**, **OpenQARP**, **PyTorch**, or the built-in NumPy
reference, and on two mixed-state backends when you want to ask what noise would have
done. Ten in all, behind one protocol: a backend supplies a statevector, and sampling,
basis rotation, qubit-wise-commuting grouping, expectation and the whole batch stack
are derived once in the base class — which is what makes agreement between backends a
property rather than a coincidence.

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

## How this is known to be correct

Everything a library checks itself against usually inherited its conventions — five
backends against each other, or PennyLane, whose names this library's gate table was
written by reading. Those all agree when a convention is wrong *everywhere*, and once
they did: the torch backend reimplemented `np.moveaxis` without numpy's
`sorted(zip(destination, source))`, so a two-qubit gate on descending wires permuted
the wires it was not acting on. `Z(3)` read `-0.1288` against `-0.7374`. Nothing
raised, `backprop` differentiated through it, and four references agreed with it.

`tests/densesim.py` is 206 lines that inherited none of it — hand-written gate
matrices, hand-derived derivatives, its own product rule, reading `spec.ops` and
calling no backend. It is what found that bug.

**When correctness actually matters, at least one reference must share no code and no
conventions with the thing being checked.**

On top of that: **301 executed parity cases against PennyLane** (gates, random
circuits, all four gradient routes, encodings, kernels, geometry, optimiser
trajectories), 17 mathematical invariants fuzzed by Hypothesis, one circuit zoo through
every installed SDK, and every snippet on the docs site executed by the suite.
**1,608 tests passing of 1,729 collected**, 0 failures.

On speed, the honest summary is **1.7× median** over 14 cases, timed against
PennyLane's *fastest* configuration rather than its reference one — a choice that costs
most of the headline. That section has had to correct itself twice, both times in
qmlkit's favour, both caught by someone re-running the comparison rather than trusting
the table. The two margins that survive a fair comparison are algorithmic: a Gram
matrix costs one circuit per *row* rather than per pair (10× at 20 points, growing with
the dataset), and the exact metric tensor is closed-form (103× at `P=24`, widening with
parameter count). JAX is not installed on the benchmark machine, so jit-compiled
PennyLane is untested and unclaimed.

→ [Validation](docs/about/validation.md) has the full tables, the four convention
differences, and where qmlkit is *less* accurate.

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
print(qk.backend_report())     # it returns the summary; it does not print it
# qmlkit backends:
#   [ok]      aer
#   [ok]      cirq
#   [ok]      cirq-density
#   [ok]      mps
#   [ok]      numpy
#   [ok]      openqarp
#   [ok]      qiskit
#   [ok]      qiskit-aer
#   [missing] spinqit  -> pip install 'qmlkit[spinqit]'
#   [ok]      torch

qk.expectation(spec, qk.Z(0), backend="qiskit")   # per call
qk.set_default_backend("aer")                      # for the session
```

Ten simulators behind one protocol: a backend supplies a statevector, and sampling,
basis rotation, qubit-wise-commuting grouping, expectation and the whole batch stack
are derived once. Adding one is a single `register_backend` call.

**Noise never picks a simulator for you.** `get_backend(noise=...)` without naming a
mixed-state backend raises and lists the ones that would work — because a noisy run
costs more, refuses two of the gradient methods, and answers a different question, so
which simulator produced a number stays written down in the code that produced it.

**Circuits come back in, too**: `from_qasm`, `from_qiskit`, `from_pennylane`,
`from_cirq`. Building that translation layer turned up three real discrepancies,
including SpinQit's `CY` applying `-iY` rather than `Y` to the control-1 subspace — a
*relative* phase between control branches, so it changes measurement statistics rather
than cancelling as a global phase. All three are handled, and
`tests/test_cross_backend.py` holds them handled.

→ [Backends and conventions](docs/guides/backends.md) · [Integrating OpenQARP](docs/guides/openqarp.md) · [Running under noise](docs/guides/noise.md) · [Backend reference](docs/reference/backends.md)

## What it does today

- **A backend-neutral circuit IR.** A circuit is data — a list of `Op`. Backends
  compile it; gradients, resource counting and drawing all read it.
- **Ten backends behind one protocol** — eight pure-state (NumPy exact
  reference, Aer, MPS, SpinQit, Qiskit, Cirq, OpenQARP, Torch) and two mixed-state
  (`cirq-density`, `qiskit-aer`) — with a cross-backend equivalence suite
  proving they agree.
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

Six methods — `adjoint`, `backprop`, `parameter-shift`, `hadamard`, `spsa`,
`finite-diff`. `auto` takes adjoint when the backend can hand back a statevector and
every gate declares a closed-form derivative, and parameter-shift otherwise. Asking for
one that cannot be honoured **raises and names the alternative**, rather than quietly
substituting a simulator-only route and returning a number that could never come off
hardware.

**Two things about parameter-shift produce a plausible wrong number rather than an
exception**, and both are handled here. Shift rules belong to the *gate*, not the call:
a rule follows from the unique positive gaps between a generator's eigenvalues, so `ry`
takes the familiar two-term ±π/2 rule while `crz` has two frequencies and needs four
terms — a circuit mixing them needs both, looked up per gate. And a tied parameter
driving several gate occurrences must shift **one occurrence at a time** and sum; shift
them together and you get a directional derivative along the wrong axis, smooth and
finite and wrong.

→ [Choosing a gradient method](docs/guides/choosing-a-gradient.md) · [The parameter-shift rule](docs/guides/parameter-shift.md)

## Encoding

```python
qk.angle_encode([0.3, 1.1])              # one feature per qubit
qk.amplitude_encode([1, 2, 3, 4])        # 2**n numbers in n qubits
qk.hamiltonian_encode(x, t=1.0, steps=3) # data-dependent Ising evolution

fm = qk.ZZFeatureMap(n_features=3, reps=2)
fm.build(x)                               # U(x)
fm.adjoint(x)                             # U(x)^dagger -- the other half of a kernel
```

Amplitude encoding compresses — 1024 features into 10 qubits — and the qubits are not
what you pay in: preparing that state costs **4,052 CNOTs at depth 6,027**, about four
two-qubit gates per feature. Loading is linear in the data, which is the number every
exponential-speedup claim over classical data has to get past.
`qk.loading_cost` and `qk.qram_cost` price it.

→ [Getting data in](docs/tutorials/02-encoding-data.md) · [Re-uploading and Fourier](docs/tutorials/07-reuploading.md)

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
