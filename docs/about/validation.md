# Validation

How any of this is known to be correct.

A library's own test suite can only catch the bugs its author thought of. qmlkit
therefore leans on three independent checks, each of which can fail for reasons the
others cannot.

| | |
|---|---|
| **Cross-backend equivalence** | The same circuit through five backends, compared to the NumPy reference |
| **Cross-library parity** | 301 cases against PennyLane, and every metric in `qk.evaluate` against scikit-learn — both independently written implementations |
| **Executable documentation** | Every snippet on this site runs in CI |

Plus the ordinary suite: **1121 tests, 94% combined coverage** measured in CI,
`ruff` and `mypy --strict` clean.

## Parity with PennyLane

```bash
pip install pennylane
pytest tests/test_pennylane_parity.py
```

| Layer | Compared | Agreement |
|---|---|---|
| Gates | all 20 gate matrices at 6 angles, and every closed-form `dU/dθ` against a differenced PennyLane matrix | `1e-12` |
| Circuits | 40 **randomly generated** circuits over the full gate set, 1–5 qubits — statevectors, probabilities, random multi-term observables | `1e-12` |
| Gradients | 5 ansätze × 4 observables; all four exact methods; PennyLane's own four back against ours; fuzzed circuits | `1e-10` |
| Encodings | angle (X/Y/Z), amplitude, basis, IQP | `1e-12` |
| Templates | `BasicEntanglerLayers`, `StronglyEntanglingLayers` | `1e-12` |
| Kernels | full Gram matrices, fidelity and swap-test estimators | `1e-10` |
| Quantum info | reduced density matrices, von Neumann entropy, purity, mutual information, fidelity, over random states | `1e-10` |
| Fourier | re-uploading spectra at depths 1–4 | `1e-10` |
| Geometry | Fubini–Study metric (full and diagonal), QFIM | `1e-12` |
| Optimisers | Rotosolve and QNG trajectories, step by step | `1e-10` |

The **randomised** tests are the ones that matter. Hand-picked cases confirm what the
author already believed; a fuzzer explores the space. Every bug found in this project
so far has been of the plausible-wrong-number kind that only a second opinion catches
— including two in this library's own parameter-shift implementation, and two in its
own tests.

## Four convention differences

None is a bug in either library. Each is pinned by its own test so it stays
deliberate rather than drifting.

**IQP angle convention.** PennyLane's `IQPEmbedding` emits `RZ(xᵢ)` and
`MultiRZ(xᵢxⱼ)`; qmlkit's `PauliFeatureMap` follows the Qiskit convention and emits
`Rz(2φ)`. Halving the data map lines them up exactly. A kernel differing by precisely
this factor would be very hard to spot.

**Amplitude encoding phase.** qmlkit builds amplitude encoding from uniformly
controlled rotations rather than a state-preparation primitive, and the phase cascade
drops one overall factor. Unobservable in isolation — every probability and
expectation is identical — but it stops being global inside a *controlled* block. The
docstring warns about it, and `check=True` re-simulates and asserts.

**Two-qubit "ring".** A ring on two qubits would revisit the same pair, so
`entangler_pairs` collapses it to a single `CX`. PennyLane's templates run their loop
uniformly and emit both `CNOT(0,1)` and `CNOT(1,0)`. A two-qubit strongly-entangling
layer is genuinely a different circuit in the two libraries; adding the second `CNOT`
by hand reconciles them exactly, which is what the test asserts.

**`approx="block-diag"`.** PennyLane blocks the metric tensor by circuit *layer* and
zeroes every cross-layer entry. qmlkit computes the exact metric, which costs no more
on a simulator. The same keyword does not port between the two libraries.

That last one is not cosmetic. On a 3-qubit, 2-layer problem at equal step count and
step size:

| | reaches |
|---|---|
| qmlkit QNG (exact metric) | **−2.9999999** |
| PennyLane QNG, default `approx="block-diag"` | −2.22 |
| PennyLane QNG, `approx=None` | traces qmlkit's trajectory to `1e-8` |

## Where qmlkit is more accurate

`state_fidelity` hits the analytic `|⟨a|b⟩|²` to `1e-16`. `qml.math.fidelity` takes
matrix square roots of rank-1 density matrices, which is ill-conditioned, and loses
about eight digits. Recorded as a test so a future tolerance change there is a
decision rather than an accident.

## Cross-backend equivalence

`tests/test_cross_backend.py` runs one circuit zoo through every installed backend,
asserting agreement on statevectors, probabilities, expectations over X/Y/Z and
two-body terms, seeded sampling, and parameter-shift gradients. The zoo deliberately
targets where SDKs differ: endianness, controlled-gate qubit order, idle qubits,
basis rotations.

It found three real upstream discrepancies, all handled — including **SpinQit's `CY`
applying `−iY` instead of `Y`** to the control-1 subspace, which is a relative phase
and therefore physically observable. Details in [Backends and
conventions](../guides/backends.md).

## Executable documentation

Every Python block on this site is executed by `tests/test_docs.py`. The snippets are
not illustrations of the API — they are tests of it, so a rename that breaks a
tutorial breaks the build, and the outputs shown were produced by running the code.

This caught two errors while the docs were being written: a wrong `RotationLayer`
call signature, and a hand-typed number that did not match what the code printed.

## Speed

`examples/benchmark_pennylane.py` times identical work on both libraries, against
PennyLane's **fastest** configuration rather than its reference one.

This section previously reported a median 6.1× against `default.qubit` alone. That was
not a fair comparison: `pennylane-lightning` is a dependency of PennyLane, so the C++
`lightning.qubit` is present in every install, and `qml.adjoint_metric_tensor` is an
`O(P)` statevector algorithm sitting right next to the `O(P²)` Hadamard-test
`qml.metric_tensor`. The numbers below are against both, summarised on the faster.

| Operation | qmlkit | PennyLane (best) | | vs `default.qubit` |
|---|---|---|---|---|
| Expectation, 12 qubits | 3.9 ms | 4.6 ms `lightning` | 1.2× | 2.9× |
| Gradient, 8 qubits, `P=96` | 11.0 ms | 11.3 ms `lightning-adjoint` | 1.02× | 6.1× |
| Parameter-shift, 6 qubits, `P=72` | 300 ms | 347 ms `lightning` | 1.2× | 3.9× |
| 20×20 kernel Gram matrix | 3.2 ms | 219 ms `default` | **69×** | 69× |
| Exact metric tensor, `P=24` | 6.8 ms | 715 ms `adjoint_metric` | **105×** | 276× |

qmlkit is ahead on 13 of 14 cases, median **1.7×**.

Read that in three parts. The expectation, gradient and parameter-shift rows are
dispatch and interpreter overhead rather than arithmetic — qmlkit does less per call,
leads at small registers, and ties by 8 qubits. The kernel Gram matrix is ~69×
because the whole matrix is one batched evaluation against one QNode call per pair;
per-call overhead dominates there so completely that `lightning` is actually slower
than `default.qubit`. The metric tensor is different in kind:
closed-form differentiation of the state, agreeing with PennyLane's own routes to
`1.7e-16`, and *widening* with parameter count (49× at `P=12`, 105× at `P=24`) rather
than narrowing.

Single machine, single thread, small registers, exact simulation throughout. JAX is not
installed on the benchmark machine, so jit-compiled PennyLane is untested and unclaimed.
Nothing here says anything about running on hardware.
