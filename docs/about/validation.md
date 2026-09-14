# Validation

How any of this is known to be correct.

A library's own test suite can only catch the bugs its author thought of. Worse, a
suite of *agreeing* implementations can only catch the bugs they do not share — five
backends and a second library once agreed, unanimously, with a wrong number.

So qmlkit leans on five checks, ordered here by how much each one can prove:

| | | |
|---|---|---|
| 1 | **An independent reference** | `tests/densesim.py` — 206 lines sharing no code and no conventions with the library it checks |
| 2 | **Cross-library parity** | 301 executed cases against PennyLane, and every metric in `qk.evaluate` against scikit-learn |
| 3 | **Property-based testing** | 17 mathematical invariants over Hypothesis-generated circuits — 1,825 per default run, ~19,500 before a release |
| 4 | **Cross-backend equivalence** | One circuit zoo through every installed SDK, compared to the NumPy reference |
| 5 | **Executable documentation** | Every snippet on this site is run by the test suite |

Plus the ordinary suite: **1,630 tests passing of 1,753 collected**, 0 failures, 94%
combined coverage measured in CI, `ruff` and `mypy --strict` clean over the whole
package.

Every one of the 123 skips is accounted for and none of them is silent: 65 constant
gates tested at a single angle rather than six, 37 needing SpinQit's Python 3.10, 19
documentation pages with no runnable Python, and 2 where the observable is wider than
the circuit. CI asserts each optional SDK actually imported before running its jobs,
because a suite guarded by `importorskip` and never given its dependency turns the job
green by skipping.

These counts drifted once before anyone noticed — the page claimed 1,563 collected
while the suite collected 1,729 — so `scripts/check_doc_numbers.py` now measures them
against the repository and says which sentence has gone stale.

## The reference that shares nothing

This is the check that matters most, and it is the newest.

Everything else here compares qmlkit against something that inherited its conventions —
five backends against each other, or against PennyLane, which qmlkit's gate table was
written by reading. Those agree when a convention is wrong *everywhere*. `densesim.py`
inherited none of them: hand-written gate matrices, hand-derived derivative matrices,
an explicit bit-index loop where the library uses `tensordot`, its own product rule.
It reads `spec.ops` and nothing else — no backend, no gradient routine, no execution
code.

It earned its place immediately. The torch backend had reimplemented `np.moveaxis`
without numpy's `sorted(zip(destination, source))`, so a two-qubit gate on descending
wires permuted the wires it was not acting on. `Z(3)` read `-0.1288` against `-0.7374`.
Nothing raised, `method="backprop"` differentiated through it, and it was reachable
from a built-in `conv_block(filter="su4")`. Four other references agreed with it,
because not one of them was independent of the convention that caused it.

The lesson generalises past this project: **when correctness actually matters, at least
one reference must share no code and no conventions with the thing being checked.**

Four properties in `tests/test_torture.py` run it against randomly generated circuits —
statevectors, expectations, gradients, and every installed backend at once.

## Property-based testing

`tests/test_torture.py` states 17 invariants that hold by mathematics rather than by
example, and lets Hypothesis look for circuits that break them: the four exact gradient
routes agree; a tied weight's gradient sums over its occurrences; batched equals looped;
`adjoint()` undoes; expectation lies inside the observable's spectrum; Qiskit and Cirq
round-trips reproduce the statevector; sampling lands within five standard errors.

A default run generates **1,825 cases**. Before a release:

```bash
QMLKIT_TORTURE_EXAMPLES=1500 pytest tests/test_torture.py
```

which is about 19,500 circuits and ten minutes. When something fails, Hypothesis
shrinks it to the smallest circuit that still shows it and replays that case
thereafter. No CI job runs the deep campaign — it is a release step, done by hand.

## Parity with PennyLane

```bash
pip install pennylane
pytest tests/test_pennylane_parity.py
```

| Layer | Compared | Agreement |
|---|---|---|
| Gates | 55 matrix comparisons over all 20 gates — the 7 parametric ones at 6 angles each, the 13 constant ones once — plus 21 closed-form `dU/dθ` against a differenced PennyLane matrix | `1e-12` |
| Circuits | 40 **randomly generated** circuits over the full gate set, 1–5 qubits — statevectors, probabilities, random multi-term observables | `1e-12` |
| Gradients | 5 ansätze × 4 observables; all four exact methods; PennyLane's own four back against ours; fuzzed circuits | `1e-10` |
| Encodings | angle (X/Y/Z), amplitude, basis, IQP | `1e-12` |
| Templates | `BasicEntanglerLayers`, `StronglyEntanglingLayers` | `1e-12` |
| Kernels | full Gram matrices, fidelity and swap-test estimators | `1e-10` |
| Quantum info | reduced density matrices, von Neumann entropy, purity, mutual information, fidelity, over random states | `1e-10` |
| Fourier | re-uploading spectra at depths 1–4 | `1e-10` |
| Geometry | Fubini–Study metric (full and diagonal), QFIM | `1e-12` |
| Gradient statistics | 4,800 gradient vectors — the batched sampling the barren-plateau tooling runs on, over 24 cells of `n` 4–6 × `L` ∈ {1,2,4,8} × {local `Z₀`, global `Z⊗ⁿ`}, 200 draws each | `1e-15` |
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

`tests/test_cross_backend.py` runs one circuit zoo through Qiskit, Cirq and — on Python
3.10 — SpinQit, asserting agreement with the NumPy reference on statevectors,
probabilities, expectations over X/Y/Z and two-body terms, seeded sampling, and
parameter-shift gradients. The zoo deliberately targets where SDKs differ: endianness,
controlled-gate qubit order, idle qubits, basis rotations.

**It does not cover the torch backend, and that omission has already cost something.**
Torch was excluded on the reasoning that a differentiable simulator is a different kind
of backend — and torch is exactly where the worst defect in 0.1.1 lived. An agreement
test that skips a participant tests nothing about that participant. Torch agreement is
now asserted instead by `tests/test_grad_batch.py` and by the property suite, which
iterates `available_backends()` and explicitly declines to skip it.

The zoo found three real upstream discrepancies, all handled — including **SpinQit's
`CY` applying `−iY` instead of `Y`** to the control-1 subspace, which is a relative
phase and therefore physically observable. Details in [Backends and
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

That choice is the point, and it costs us most of the headline. `pennylane-lightning`
is a dependency of PennyLane, so the C++ `lightning.qubit` is in every install whether
the user asked for it or not; and `qml.adjoint_metric_tensor` is an `O(P)` statevector
algorithm sitting right next to the `O(P²)` Hadamard-test `qml.metric_tensor`. Timing
against `default.qubit` and the `O(P²)` route would be timing an opponent nobody runs.

This section has had to correct itself twice, which is the useful part of it. It once
timed `default.qubit` when `lightning.qubit` ships with PennyLane, and reported a median
of 6.1×. It then timed a Gram matrix against a pair-at-a-time QNode loop and reported
**69×** on that row — when `default.qubit` *broadcasts* a stacked array and does the
same work in one call. Both were flattering; both were caught by someone re-running the
comparison rather than trusting the table. The figure is **1.7×**, it is the one below,
and the naive column is printed beside it so the size of the difference stays visible.

| Operation | qmlkit | PennyLane (best) | | vs the naive route |
|---|---|---|---|---|
| Expectation, 12 qubits | 3.8 ms | 4.7 ms `lightning` | 1.2× | 3.0× |
| Gradient, 8 qubits, `P=96` | 10.9 ms | 10.8 ms `lightning-adjoint` | 1.01× *slower* | 6.0× |
| Parameter-shift, 6 qubits, `P=72` | 303 ms | 333 ms `lightning` | 1.1× | 3.7× |
| 20×20 kernel Gram matrix | 0.31 ms | 3.1 ms `broadcast` | **10×** | 654× |
| Exact metric tensor, `P=24` | 6.7 ms | 691 ms `adjoint_metric` | **103×** | 271× |

qmlkit is ahead on 14 of 14 cases, median **1.7×**. The 8-qubit gradient row is a dead
tie that falls either way between machines; the table quotes the run where it falls
against qmlkit.

`examples/benchmark_pennylane.py` checks that both libraries produce the *same number*
before it reports a speedup, and prints the agreement alongside. An acceleration that
changes the answer is not an acceleration.

Read that in two parts. The expectation, gradient and parameter-shift rows are dispatch
and interpreter overhead rather than arithmetic — qmlkit does less per call, leads at
small registers, and ties by 8 qubits. Those margins shrink as `2ⁿ` grows, and nothing
should be planned around them.

The last two rows are algorithmic, and they survive the fair comparison. A Gram matrix
costs one circuit per *row* rather than one per *pair*, because the inversion test's
`P(0…0)` is exactly `|⟨ψ(x′)|ψ(x)⟩|²` and a simulator can hand back the state — linear
in the dataset instead of quadratic, so the margin *grows* with it (10× at 20 points,
53× at 128). That shortcut needs a statevector, so a device still pays the pairwise
count and `QuantumKernel.circuits_on_hardware` reports it; budget a hardware run from
that number rather than from `n_evaluations`. A PennyLane user who hand-rolls the same
state-overlap trick comes within 0.86× of it, which is the honest framing: the algorithm
is the win, and what qmlkit contributes is that `QuantumKernel(fmap)(X)` already does
it. The metric tensor is closed-form differentiation of the state, agreeing with
PennyLane's own routes to `1.7e-16` and *widening* with parameter count (50× at `P=12`,
103× at `P=24`) rather than narrowing.

Single machine, single thread, small registers, exact simulation throughout. JAX is not
installed on the benchmark machine, so jit-compiled PennyLane is untested and unclaimed.
Nothing here says anything about running on hardware.
