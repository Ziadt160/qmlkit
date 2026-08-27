# qmlkit

A backend-agnostic quantum machine learning library for **SpinQit**, **Qiskit**,
**Cirq** and **PyTorch**.

`qmlkit` provides the ML layer that quantum SDKs leave out: reusable feature maps,
an ansatz vocabulary, quantum kernels, torch layers, and a general-purpose
parameter-shift gradient you can point at *any* circuit and observable.

Exact here means no shot noise and no finite-difference bias — not bit-identical
arithmetic. Expectations and gradients agree with the analytic value to machine
precision.

**Simulator-only** for the whole `0.x` line, and **backend-agnostic**: the same
circuit runs on SpinQit, Qiskit, Cirq, or the built-in exact NumPy reference.
Expectations are exact unless you ask for shots.

**[Documentation](https://ziadt160.github.io/qmlkit/)** — tutorials, guides and a generated API reference.
**[HANDOFF.md](HANDOFF.md)** — status, conventions, known traps, and what to do next, if you are picking this up.

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
qk.reupload(fmap, n_layers=3)                      # S W S W S W
qk.reupload(fmap, n_layers=3, order="WS")          # vary before the first upload
qk.reupload(fmap, n_layers=3, share_weights=True)  # one tied block, reused
qk.reupload(fmap, n_layers=3, block=qk.RotationLayer("ry") + qk.EntanglerLayer("cz", "ring"))

# or compose directly — two different feature maps in one model
qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle),
          n_inputs=3)
```

Any of these drops straight into a `QuantumLayer`, input gradients included.

> **One trap the library catches for you.** `L` uploads reach frequencies `0..L`
> only when the trainable block does **not** commute with the encoding rotation. If
> it does — `Ry(x) Ry(θ₁) Ry(x) Ry(θ₂) = Ry(2x + θ₁ + θ₂)` — the uploads merge into a
> single rotation, the model reaches one frequency, and every weight becomes a phase
> shift. It looks like a `3L`-parameter model; it is a one-parameter family.
> `reupload()` warns. Verify any model with `qmlkit.fourier.spectrum`.

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
#   [ok]      numpy
#   [ok]      qiskit
#   [missing] spinqit  -> pip install 'qmlkit[spinqit]'

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

### Cross-validated against PennyLane

A library's own test suite can only catch the bugs its author thought of. Agreeing
with a second, independently written implementation catches the rest.
`tests/test_pennylane_parity.py` is **301 parity cases** across every layer both
libraries implement, and it runs in CI like any other test:

```bash
pip install pennylane
pytest tests/test_pennylane_parity.py
```

| Layer | What is compared | Agreement |
|---|---|---|
| Gates | all 20 gate matrices at 6 angles, and every closed-form `dU/dθ` against a differenced PennyLane matrix | `1e-12` |
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
catches.

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

### Speed

`examples/benchmark_pennylane.py` times identical work on both libraries. qmlkit is
faster on 14/14 cases, median **6.1×**:

| Operation | qmlkit | PennyLane | |
|---|---|---|---|
| Expectation, 12 qubits | 3.8 ms | 11.3 ms | 2.9× |
| Gradient, 8 qubits, `P=96` | 11.1 ms | 63.9 ms | 5.7× |
| Parameter-shift, 6 qubits, `P=72` | 308 ms | 1161 ms | 3.8× |
| 20×20 kernel Gram matrix | 31 ms | 212 ms | 6.8× |
| Exact metric tensor, `P=24` | 6.9 ms | 1863 ms | **268×** |

Read those two ways. The first four are mostly dispatch overhead — PennyLane carries
a general transform pipeline qmlkit does not have, which buys features this benchmark
never exercises, and the gap narrows as `2^n` starts to dominate (4.4× at 4 qubits,
2.9× at 12). The metric tensor is different in kind: PennyLane runs `O(P²)`
Hadamard-test circuits and needs a spare ancilla wire, while qmlkit differentiates the
state in closed form — `P` derivative states from one forward sweep. That gap widens
with size rather than narrowing.

## What it does today

`0.1.0.dev0` is the foundation layer:

- **A backend-neutral circuit IR.** A circuit is data — a list of `Op`. Backends
  compile it; gradients, resource counting and drawing all read it.
- **Four backends behind one protocol** — NumPy (exact reference), SpinQit,
  Qiskit and Cirq — with a cross-backend equivalence suite proving they agree.
  A backend supplies a statevector; sampling, basis rotation and expectation
  semantics are defined once in the base class.
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

**`PauliFeatureMap` is the runnable version of the lecture's.** `Lecture3` cell 42
calls `_basis(...)` and `_phi(...)`, neither of which is defined anywhere in that
repository, so the cell cannot execute. Both are supplied here as `basis_change`
and `default_data_map`, and the maps are tested against the analytic kernels they
are supposed to induce — the angle map's `cos²((x−x')/2)`, the Z map's
factorisation, and the ZZ map's failure to factorise.

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
| 7 · Docs, tutorials, `v0.1.0` on PyPI | in progress |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
mypy
```

Three runnable examples, none of which quotes a number it did not compute:

```bash
python examples/quickstart.py            # every layer of the library, end to end
python examples/compare_pennylane.py     # readable cross-check against PennyLane
python examples/benchmark_pennylane.py   # wall-clock, same work on both sides
```

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

[`RELEASING.md`](RELEASING.md) has the rest of the process.

## License

Apache-2.0 · © 2026 Ziad Tarek Mohammed
