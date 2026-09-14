# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed - `QMeans.fit(X, seed=...)` did not make a shots-based run reproducible

`_kernel` always read the *constructor's* `seed`, never `fit`'s, so the run seed
reached the centroid draw and not the kernel's sampling. An unseeded `QMeans(shots=300)`
fitted twice with the same `fit(X, seed=42)` returned inertia 6.740 and 5.613 --
`fit(X, seed=...)` reads like "make this run reproducible" and did not. It now seeds
both, and a constructor seed still works on its own.

### Added - which gradient methods keep a torch layer batched

`QuantumLayer` computes a whole batch's gradient in one backend call, which is the
difference between a training step costing one call and one per sample. Only
`auto`, `adjoint` and `parameter-shift` take that path. `hadamard`, `spsa`,
`finite-diff` and anything registered fall back to a per-sample Python loop -- no
error, no warning, correct answers, an order of magnitude slower. `backprop` is refused
outright, which the gradient guide already covered; the silent fallback it did not.

Set `grad_method="hadamard"` on a `VQC` because the comparison table says it costs half
what parameter-shift does, and you get the per-slot saving *and* lose the batching. The
guide now says so, with the table.

### Added - narrative for the parts of the library that had only a signature

- **`qmlkit.generative`** had no prose at all. Born machines sample easily and cannot
  score a point; energy-based models score easily and sample only through a chain --
  the split decides what you can ask of them. `QuantumBoltzmannMachine.grad` returns a
  *lower-bound* gradient, so a run that stops improving may have hit the bound rather
  than the optimum, which is worth knowing before you debug the optimiser.
- **`qmlkit.shadows`** had one sentence. Shadows predict `M` observables at a cost
  growing with `log M` rather than `M`, against a variance penalty that grows sharply
  with locality: they win on many local observables and lose on a few global ones.
  `shadow_shot_cost` and `kernels.kernel_shot_cost` price the two routes; the crossover
  moves with locality and no rule of thumb survives it.
- **The capacity half of `qmlkit.metrics`** -- `generalization_bound`,
  `samples_for_gap`, `effective_dimension`, `noise_survival` -- was reachable only by
  already knowing the names. `hardware_efficient(4, 3)` has 24 parameters and a bound
  of 0.87 against 100 samples, which for a metric in `[0, 1]` constrains nothing;
  reaching a gap of 0.1 needs 7,628 samples. At `hardware_efficient(6, 8)` it is 2.09
  and 43,818. That arithmetic is usually the real reason a model that trains
  beautifully does not generalise.
- **`AdaptVQE` and the RL policy** had no worked example anywhere in the docs. Both now
  have one, and because they are executable the suite runs them: the ADAPT example
  reaches `-1.13730603` Ha from one operator, and the REINFORCE example reports 0.70
  against a random-policy baseline of 0.50 -- quoted together, because a mean return
  without its baseline beside it says nothing.
- **`QuantumPolicy`** documented none of its nine constructor arguments. It documents
  all nine now, `beta` included: expectations live in `[-1, 1]`, a narrow range to take
  a softmax over, so it decides whether the policy commits or keeps exploring.


### Fixed - errors that named the wrong thing, and counts that understated

The second pass over the audit findings. None of these returned a wrong number; each
one either failed in a way that pointed somewhere else, or reported a cost that was not
the cost.

**`QAOA` did not check its ansatz against the problem**, though `VQE` does. A 2-qubit
ansatz on a 4-node graph constructed happily and failed inside `run()` with
`IndexError: tuple index out of range`. It now says which width it got and which it
wanted, in the same words VQE uses.

**`QAOA` required edge tuples specifically.** `isinstance(problem[0], tuple)` sent a
list-of-lists down the *observable* branch, where it died on `'list' object has no
attribute 'support'` - an error about the wrong thing entirely. Any pair-shaped
sequence is now an edge; an `Observable` still routes to the observable branch, and a
test holds that guard.

**`Ansatz(..., n_inputs=N)` on a block with no `EncodingLayer` created inert slots.**
Nothing read them, so `bind(x, weights)` accepted any `x` and returned the same
circuit - verified: two different feature vectors, identical statevectors, no error.
The existing check could not catch it, because it needs a declared encoding to disagree
with and there is none. Refused now, naming the fix.

**`gradient_cost(spec, "spsa")` was hardcoded to 2**, ignoring `n_avg`. SPSA costs two
evaluations *per average*, so tutorial 3's own example - which calls it with
`n_avg=50`, i.e. 100 evaluations - printed a cost fifty times too small directly
beneath the call it was describing. The function now takes `n_avg`.

**`VQEResult.n_evaluations` counted loss calls only.** Every gradient the optimiser
asked for was free, so ten steps of gradient descent reported 11 - the same number
whatever the gradient method. It now adds each gradient's real circuit cost: 21 under
adjoint, 251 under parameter-shift, for the identical run.

### Fixed - four docstrings that described something other than the code

- **`FeatureMap`** said "subclasses implement `build`". `build` is concrete and no
  shipped subclass overrides it; the abstract three are `angles`, `n_angles` and
  `_emit`. Following the old advice leaves `build_parametric` calling the base
  `_emit`, which raises - so the torch layer breaks and nothing else does.
- **`qmlkit.evaluate`** said "four tasks" while exporting five: `selective` and
  `risk_coverage` were missing from its own count, and the evaluation guide had copied
  the undercount.
- **`qmlkit.optim`** opened with "Adam and SGD come from torch" immediately above its
  own NumPy Adam - which is what every `qmlkit.algorithms` solver means by
  `optimizer="adam"`.
- **The Hadamard gradient** was described as "one circuit per parameter" in its own
  module docstring, the gradient guide and tutorial 3. It is one per parameterised
  *slot*: a weight tied across three occurrences costs three circuits, not one. This is
  the same slot-versus-parameter distinction the parameter-shift guide devotes a
  section to.

### Added - the constructor `QSVC` and `QSVR` actually have

Both inherit everything from a private base, so the generated reference showed
`feature_map`, a solver argument and `**kwargs` - with `shots`, `backend`, `bandwidth`,
`estimator`, `repair_psd` and `seed` invisible. `bandwidth` matters most: the source
calls it "the first thing to try when a kernel has concentrated", and it appeared
nowhere in the docs. Now a table on the kernels reference page, alongside the inherited
methods and the `clone` contract that lets these sit inside a scikit-learn `Pipeline`.


### Fixed - ADAPT-VQE reported the reference energy as a converged result

Following the documented chemistry recipe exactly - `chemistry_operator_pool`, no
`reference=` - returned **+0.720 Ha against a true -1.137**. An error of 1.86 Ha, about
1,165 kcal/mol where chemical accuracy is 1, with no warning of any kind.

The cause is that a particle-conserving pool cannot change the particle number, so from
the default vacuum every candidate has exactly zero gradient, the selection loop breaks
on its first pass, and the reference energy is returned looking converged. The docs
said the pool avoids the particle-number trap "at Hartree-Fock" - but reaching
Hartree-Fock needs `reference=[0, 1]`, and `AdaptVQE.__init__` had no docstring at all.

Selecting **zero** operators is now a warning, because it is never convergence: it means
nothing in the pool can move the energy off the reference state. `reference` is
documented, with the two numbers above in it. The default is unchanged - the vacuum is
right for a generic pool, and ADAPT cannot infer an electron count.

### Fixed - the quantum autoencoder honoured two of the four optimisers it advertised

`QuantumAutoencoder.fit(optimizer="adam")` and `"gradient-descent"` raised
`TypeError: _adam() missing 1 required keyword-only argument: 'grad'`. VQE, QAOA and
ADAPT all inject the gradient; this class never did, and `tests/test_optimizer_wiring.py`
- written for exactly this bug class, after `optimizer="adam"` once raised from all
three - parametrises over those three and not this one.

`gradient_of_loss` now supplies it. The trash projector expands to an ordinary Pauli
sum specifically so the loss is a plain expectation value that `qk.grad` can
differentiate, which is what `_trash_projector`'s docstring has always said the design
was for; the method it implies was simply never written. It agrees with finite
differences to `8e-11`.

### Fixed - the re-uploading collapse warning fired on correct circuits

`DataReuploadEncoder` warned that "the uploads collapse into a single rotation" whenever
the trainable rotations shared the encoding's generator - without checking its own
`entanglement` parameter, which defaults to `"chain"`.

`Ry(x) Ry(t) Ry(x)` only composes on one wire with nothing in between. Measured at
`n_features=2`: with no entangler the model does reach one frequency and the warning is
right; with the default `"chain"` it reaches the full `0..3` spectrum and the warning
was telling the caller to redesign a circuit that was correct. `_commutes_with_encoding`
in `qmlkit.ansatz.reupload` already reasons this out for the other implementation of the
same idea; this one now matches it.

A false positive here is worse than a false negative - it teaches people to ignore the
tool - so the new test asserts the warning stays **quiet** on the default configuration,
and separately that the frequencies it claims are the frequencies actually reached.

### Fixed - two documented call paths raised AttributeError

`qmlkit.progress.log(name, value)` cannot work. Importing the `progress` context manager
into the top-level namespace rebinds `qmlkit.progress` to the *function*, so the dotted
path resolves to a function attribute that does not exist - and it stays rebound even
after an explicit `import qmlkit.progress`.

That exact path was printed in two places: `docs/guides/watching-a-run.md`, and the
"No series were logged" panel of every generated HTML run report. Both now say
`from qmlkit.progress import log`, and the guide says why the dotted form does not work.

### Fixed - `resources()` understated circuit cost by the observable count

`QuantumLayer.resources()` reported the single-observable figure. Both `forward_batch`
and `backward_batch` loop once per observable with no sharing, so a default 4-qubit
`VQC` - which reads one `Z(i)` per qubit - cost **164 circuit evaluations per sample
under parameter-shift against the 41 reported**.

Both figures now carry the observable factor. The per-observable constants are
unchanged: only the factor was missing, and that much is readable straight off the two
`for j, obs in enumerate(self.observables)` loops rather than inferred from a
measurement. `resources()` exists, in its own docstring's words, "so nobody discovers
it an hour in"; being wrong by 4x for the default configuration is the opposite of
that.


### Fixed - four documentation claims that had drifted away from the library

A six-way audit of every subsystem against its docs. The counts had gone stale in the
one direction nobody checks, because `docs/llms.txt` is generated and CI fails on a
stale copy, while the numbers written into prose had no such gate:

| Claim | Said | Measured |
|---|---|---|
| `validation.md` test suite | 1,563 collected | **1,731** |
| `AGENTS.md` map | "seven" backends | **nine** |
| `guides/backends.md` | "eight backends" | **nine** |
| `README.md` backend listing | seven, no `aer`/`mps` | **nine** |

The README's `backend_report()` example was wrong twice over: a stale listing, and
written as a bare call when the function *returns* the summary rather than printing it,
so a reader who copied it saw nothing at all. It is now `print(qk.backend_report())`
and matches the real output.

`scripts/check_doc_numbers.py` measures these against the repository and names the
sentence that has gone stale. It does not rewrite the prose - a number here carries an
argument around it, and a script that edited those sentences would flatten the voice
they are written in. It caught its own first regression immediately: adding two pages
added two documentation tests, and the freshly-corrected 1,729 was already 1,731.

### Fixed - `QSVC` shipped with no docstring at all

The class docstring sat *below* `_estimator_type = "classifier"`, so Python never
assigned it to `__doc__` - `qk.QSVC.__doc__` was `None`, and the generated reference
page rendered the library's headline kernel classifier with no description. `QSVR`,
directly beneath it, was fine, which is what made it an ordering slip rather than a
choice.

### Added - a reference page for the backends, and the nine are all on it

Seven of the nine backend modules had no reference entry anywhere: `MPSBackend` was
absent from every table and every page, and `AerBackend(device="GPU")`,
`SpinQitBackend(compiler=...)` and `NumpyBackend`'s hard 24-qubit cap were readable
only in the source. A backend that silently truncates past a bond dimension, or refuses
a GPU it cannot find, should say so on the page you look it up on.

`docs/reference/backends.md` now carries one entry per module plus the capability
matrix that previously had to be reconstructed from three separate guides - which is
how `mps` went missing from all of them.

### Added - `recommend()` and `parallel_map()` reached zero documentation pages

Both are in `qmlkit.__all__`, and `about/stability.md` therefore promises them stable.
Neither appeared on any reference page, guide or tutorial - a direct violation of this
project's own rule that anything public gets a reference entry. With them added, **every
one of the 214 public names is now covered by a reference page**, up from 210.

### Changed - the README stopped being a second copy of the documentation site

821 lines, of which roughly 460 restated `guides/agents.md`, `guides/backends.md`,
`guides/noise.md`, `guides/parameter-shift.md`, `guides/choosing-a-gradient.md`,
`tutorials/02-encoding-data.md` and `about/validation.md` - the long version living in
the file that gets read most and maintained least. That is exactly the seam the backend
count drifted through: the guide was updated for `aer`, the README was not, and nothing
was updated for `mps`.

Now 553 lines. Each cut section keeps the one fact that earns its place - the torch
`moveaxis` bug, the 1.7x median timed against PennyLane's fastest configuration, the
4,052 CNOTs to load 1024 amplitudes, SpinQit's `CY` phase - and links to the page that
carries the rest. Nothing was deleted that does not have a better-linked home.

### Added - the connective tissue between sections

`docs/about/index.md`, so About no longer drops a reader straight into a page with no
orientation while every other section has a landing page. `navigation.footer` enabled,
so the 7 of 8 case studies with no forward link get a mechanical previous/next. And the
tutorial sequence now ends by pointing at the case studies, which it never did - a
reader following every "Next:" link in order reached the end without being told the
worked, messy, real-data examples existed.


### Added - cross-library parity now covers the sampling, not just single gradients

Every PennyLane parity row until now compared one gradient at one point. The
barren-plateau tooling does not run on one gradient: it runs on a batch of several
hundred random `theta`, reduced to a variance, and a pipeline can agree on every
individual number and still disagree once sampling and reduction are layered on top.

So the table gains a row for the thing that actually runs: 4,800 gradient vectors over
24 cells of `n` 4-6 x `L` in {1,2,4,8} x {local `Z0`, global `Z^n`}, 200 draws each,
qmlkit's batched adjoint against PennyLane's `lightning.qubit` adjoint on a
hand-matched identical circuit. **Max elementwise difference 1.0e-15** - float
roundoff, across all of them.

### Added - a dead parameter reads exactly like a fully collapsed plateau

New section in the PennyLane migration guide, because this is the mistake the
barren-plateau literature is easiest to reproduce wrongly with, and it does not raise.

Fix a parameter index, watch its gradient variance as the register grows, and you have
the standard recipe. But some indices are *structurally* silent for a given circuit and
observable, and which one moves with depth and with the cost. Measured on
`hardware_efficient(6, 1)` against a global `Z^6`: parameter 0 returns `2.5e-33` while
the best live parameter returns `1.8e-01`. Thirty-two orders of magnitude between "this
model is dead" and "this model is fine", decided by which index you happened to type.

`gradient_variance` already warns here; `gradient_stats` sidesteps it by returning
every parameter at once, which one adjoint pass has already computed. The guide says
plainly that no equivalent warning is implied of PennyLane - `qml.grad` is a
differentiation primitive and an exact zero is the correct answer to what it was asked.


### Fixed - the classical Fisher information averaged over data it never read

`metrics.fisher_information` took an `X`, looped `for _ in rows`, and computed the
same gradient at the same parameters every iteration. The row was discarded. Every
dataset therefore produced the identical rank-1 matrix:

```python
f1 = fisher_information(ansatz, one_row,        theta)
f2 = fisher_information(ansatz, fifty_other_rows, theta)
np.allclose(f1, f2)          # True, for any two datasets
np.linalg.matrix_rank(f2)    # 1, always
```

`effective_dimension` is built on that matrix, so it reported roughly one usefully
independent parameter for every model anyone ever passed it - a plausible-looking
number, never a measured one.

The row now supplies the ansatz's input slots through its own feature maps, and
`theta` is the weights alone. On a 3-qubit re-uploading model with 12 weights the FIM
comes back rank 6 from 60 rows, for an effective dimension of 2.6 - which is the kind
of number the function was always supposed to produce.

An ansatz with no input slots cannot be reached by `X` at all, and now raises instead
of returning the rank-1 matrix silently. Gradients are batched, so the fix is also
faster than the loop it replaces.

### Added - `qmlkit.landscape`, and which of the four barren plateaus you have

A model that will not train gives you a flat loss curve, and a flat loss curve has
four causes with four different fixes. "Barren plateau" names all of them, which is
why the advice attached to it so often fails to help.

`landscape(ansatz)` measures every parameter's gradient in one batched pass - the old
`gradient_variance` probed one parameter and threw away the other `p - 1` entries of
each gradient it computed - and then attributes any flatness by controlled comparison
rather than by rule of thumb. Cost globality is tested by re-measuring the same
circuit against `Z(0)`; noise by re-measuring it on an exact backend. Expressibility
and entanglement are correlations rather than experiments, so they are only reported
when there is a small gradient for them to explain, and a healthy circuit reports
nothing at all.

The separation that matters most is the cheapest: a parameter with **no** gradient is
not on a plateau. `hardware_efficient(4, 2)` measured against `Z(0)` has 11 of its 16
parameters at exactly zero - they change the state and `Z(0)` cannot see them - and
calling that a plateau sends you to reduce depth when the fix is to measure something
else.

Also `minima_scan`, which runs the optimiser from several random starts and reports
whether they agreed, and `hessian_spectrum`, which separates a genuine minimum from a
saddle the optimiser stalled on. Both answer questions a loss curve cannot.

### Fixed - the minima scan was grading the optimiser, not the landscape

Found by running the new tools on a real classifier rather than a fixture, which is
the only way this class of thing turns up.

`minima_scan` spent its whole step budget at one learning rate. Adam at a fixed step
does not settle: on a 4-qubit re-uploading model, five of six runs still had a gradient
norm around `1e-2` after 3,000 steps, while their losses had not moved since step 400.
They were orbiting a basin, not descending a slope, and every endpoint came back
`not-stationary` however long the run.

Two things were wrong underneath:

- **The budget was spent at one rate.** It is now split over three decreasing ones.
  Same total steps, gradient norms two orders of magnitude lower, losses unchanged.
- **Stationarity was an absolute cut on `|g|`.** It is now the outstanding Newton step
  `|g| / lambda_max`, in radians. An absolute threshold called the same point converged
  under `Z(0)` and unconverged under `sum_i Z(i)`, purely because the second observable
  is four times larger.

`SPURIOUS_MINIMA` is now gated on `converged`: a scan whose runs were still descending
reports `MINIMA_SCAN_INCONCLUSIVE` and says to raise `n_steps`, because part of that
spread is unspent budget and the two are not separable from the outside. On the model
above the verdict survives the gate: the spread is 0.1898 at 400 steps and 0.1897 at
5,000, so those minima are real rather than unspent budget. The basin *count* did move
with the budget, 6 at 400 steps and 5 from 1,000 on, so one of the six was the
artefact and the other five were not.

### Added - `overparametrisation`, and the threshold it finds

The QFIM rank saturates at the dimension of the circuit's dynamical Lie algebra, and
past that point extra parameters add flat directions rather than uphill ones. Measured
on `hardware_efficient`, the saturated rank is exactly `2 * 2^n - 2` - the real
dimension of projective Hilbert space - reached at 8 parameters on 2 qubits, 18 on 3
and 32 on 4.

It predicts behaviour, not just geometry. On a 2-qubit Hamiltonian the ansatz one
layer below saturation converged from every start to -1.65803 against a true ground
state of -1.77652; the first ansatz at saturated rank hit the ground state exactly. On
a frustrated 4-qubit model, 4 layers is both where the rank saturates and the first
depth at which 16 random starts all land in one basin - 1, 2 and 3 layers each leave
two, spreading 3.78, 0.11 and 0.11 in final loss.

One scan at one width is not a proof, and `minima_scan` reports the spread rather than
a verdict for that reason.

### Added - what loading the data costs, and the QRAM it would take not to

`loading_cost` prices the encoders against each other, and the number that matters is
not the qubit count. Amplitude encoding puts 1024 features into 10 qubits and spends
4,052 CNOTs at depth 6,027 doing it - 3.96 two-qubit gates per feature, converging on
`4N`. Loading the data is linear in the data, so an algorithm that then runs in
`O(polylog N)` has moved the linear cost into state preparation rather than removed
it. `Theta(n)` two-qubit gates is a lower bound for an arbitrary state, so there is no
better circuit to find.

`qram_cost` prices the device those speedup claims assume. Bucket-brigade QRAM is
`O(log N)` in **depth** and `Theta(N)` in everything else: 1024 addresses need 1024
memory cells and 1023 routing components, all coherent at once. Either way the
`Theta(N)` is paid somewhere.

qmlkit does not ship a QRAM, and the module says why rather than leaving the omission
to be inferred.

A real, non-negative vector needs no phase preparation and costs exactly half - 2,026
CNOTs against 4,052. The cost model reports the generic figure, because counting the
cheap case and quoting it as the price is how a cost model flatters.


### Fixed - the REINFORCE test was seeded everywhere except where it mattered

`test_policy_gradient_learns_a_bandit_with_a_known_optimum` seeded the bandit and the
training loop and then built an **unseeded** `QuantumPolicy`, so the starting
parameters came from fresh OS entropy on every run. Across 40 trials the margin over
the assertion threshold ranged from +0.06 to +0.26 — usually comfortable, occasionally
not, and the failure looked like a regression when it was a coin landing badly.

Now three seeds, each of which must learn. A single seeded run would have removed the
flake while testing less: it would assert that one lucky start learns.

### Fixed - two benchmark rows were timing a Python loop and calling it PennyLane

`default.qubit` **broadcasts** when it is handed a stacked array. Two rows never gave
it the chance, and both were quoted in the README, the validation page and the
migration guide:

| row | timed against | reported | against PennyLane's fast route |
|---|---|---|---|
| 20x20 kernel Gram | one QNode call per pair | 69x | **10x** |
| gradients for a batch of 32 | a loop over `lightning.qubit` adjoint | 41x | **5.4x** |

The gradient one is the subtler of the two: the rows are independent parameter
vectors, so `d(sum_i f(t_i))/dt_i` is exactly each row's own gradient and one backward
pass produces all 32 — 19 ms against the loop's 107 ms. Both scripts now use those
routes, and `examples/benchmark_pennylane.py` prints the naive column beside the fair
one so the gap between "what a newcomer writes" and "what PennyLane can do" stays
visible instead of being folded into a headline.

The table's summary is unchanged at a median of **1.7x** and now reads 14 of 14
rather than 13 of 14, because the kernel row moved and the 8-qubit gradient tie fell
the other way. That row is a genuine tie and the documented figure quotes the run
where it falls against qmlkit.

This is the second time this comparison has had to correct itself — the first was
timing `default.qubit` when `lightning.qubit` ships with PennyLane, which had inflated
the median to 6.1x. Both were found by re-running it rather than trusting the table,
which is now said out loud on the validation page.

### Changed - the batched NumPy kernel now goes through BLAS, and is 5x faster

`_apply_batch` applied a gate to a stack of states with `np.einsum` on generated
subscripts. **`np.einsum` without `optimize=` never calls BLAS** — it runs NumPy's own
nested-loop C kernel — so the hottest loop in the library was leaving a batched `gemm`
on the table. Reshaping to `(batch, 2**k, 2**(n-k))` and using `@` reaches it.

Microseconds per gate, 32-row stack at 10 qubits:

| gate | einsum | now |
|---|---|---|
| 1-qubit, wire 0 | 190.4 | **82.5** |
| 1-qubit, wire 5 | 366.0 | **90.9** |
| 2-qubit, wires 0,1 | 297.9 | **89.9** |
| 2-qubit, wires 6,7 | 755.2 | **352.7** |

The tell was that einsum's cost swung 2.5x with *where the gate sat*: a naive loop is
at the mercy of the stride pattern and a gemm on a packed block is not. The benchmark
that originally chose einsum tried only wires `(0, 1)` and so never saw it.

An earlier note in that function said the reshape-and-copy route loses because it
copies the stack twice per gate. The copy is real and the conclusion was wrong — BLAS
repays it several times over. Dropping einsum also drops a 26-letter subscript
alphabet that had silently capped the batched path at 23 qubits.

End to end on `expectation_over` with 32 rows, against the faster of PennyLane's
`default.qubit` and `lightning.qubit`:

| qubits | before | after | vs PennyLane |
|---|---|---|---|
| 8 | 0.0116 s | 0.0038 s | 1.35x → **4.18x** |
| 9 | 0.0228 s | 0.0062 s | 0.95x → **3.50x** |
| 10 | 0.0548 s | 0.0110 s | 0.61x → **2.93x** |

which removes the 9-to-12 qubit band where qmlkit was slower than PennyLane outright.

`batch_max_qubits` moved 10 → 11 as a consequence: a 5x faster batched path pushes its
crossover against the one-at-a-time loop up with it. 11 is the last width that wins at
every batch size measured (8, 32, 128 rows); at 12 it is a win at 32 rows and a loss at
8, so the boundary now *does* move with batch size, which the previous note said it did
not.

### Changed - a Gram matrix now costs one circuit per row, not one per pair

A fidelity kernel's inversion test reads `P(0...0)` of `U(x')† U(x)|0⟩`, which *is*
`|⟨ψ(x')|ψ(x)⟩|²`. A device has to build that composed circuit because it cannot hand
back a state. A simulator can: evaluate each row once and let BLAS form every overlap.

`QuantumKernel` did the quadratic thing — `m(m-1)/2` composed circuits, batched but
still quadratic. It now evaluates `m` states and takes one matrix product, which is
the difference between linear and quadratic in the size of the dataset:

| qubits | rows | pairs | before | after | |
|---|---|---|---|---|---|
| 4 | 24 | 276 | 24.2 ms | 2.3 ms | 10.4x |
| 4 | 64 | 2016 | 166.6 ms | 4.9 ms | 34.0x |
| 6 | 64 | 2016 | 929.4 ms | 19.8 ms | 46.9x |
| 8 | 64 | 2016 | 4911.3 ms | 81.0 ms | 60.6x |

The cache moved with it, from per pair to per row, which is what makes prediction
cheap: a test matrix against the training set used to evaluate every `(test, train)`
pair and now re-encodes only the test rows.

**This is a simulator-only shortcut and it changes what a circuit count means**, so
the count was split rather than quietly redefined. `n_evaluations` still reports what
actually ran; the new `circuits_on_hardware` — on `QuantumKernel` and on the `QSVC` /
`QSVR` estimators — reports what the same Gram matrices would have cost on a device,
which is the pairwise number. Budgeting a hardware run from the first would under-count
it by `(m-1)/2`.

Found by benchmarking against PennyLane and losing. Against the standard PennyLane
kernel recipe (`IQPEmbedding` composed with its adjoint, broadcast) this is now 8.6x
at 24 points and 40.9x at 128, agreeing to 1e-16. Against a PennyLane user who
hand-rolls the same state-overlap trick it is 2.2x at 24 points and 0.86x at 128 —
that is to say, the algorithm is the win and it is not proprietary. The difference is
that it is what `QuantumKernel(fmap)(X)` already does.

### Fixed - `recommend()` sent single circuits to Aer three qubits too early

Aer's per-job cost is paid once per `run()`, so a batch amortises it across every row
and one circuit pays all of it. `recommend()` applied the batched crossover to
everything, and so recommended `aer` from 12 qubits even when told `batch=1`, where it
is about 15% *slower*. Measured, ms for one expectation, numpy/aer: 3.78/4.49 at 12
qubits, 8.05/9.46 at 14, 14.01/9.08 at 15, 21.13/10.84 at 16.

`batch=1` now gets `AER_CROSSOVER_UNBATCHED = 14`; anything else keeps the measured
batched boundary of 11. Not knowing the batch size is treated as the batched case, not
as one circuit, because every training loop in this library evaluates batches.

### Fixed - a registered gate now reaches every backend that can take one

`register_gate` is advertised as an extension point, and the documentation said a gate
you add gets correct gradients, resource counting and a torch layer without your
writing any of it. On the NumPy reference that was true. On every other backend the
gate raised:

```text
NotImplementedError: gate 'xy' has no Qiskit mapping; add it to _GATE_METHODS
```

An extension point whose failure mode is *edit the library's own source* is not an
extension point. The registry was a NumPy-only feature while the docs promised
otherwise.

A gate the SDK has no name for is now emitted as its matrix — `UnitaryGate` on Qiskit,
`MatrixGate` on Cirq — built from the `matrix=` supplied at registration, so the two
density-matrix backends inherit it as well. Statevectors agree with the NumPy reference
to machine precision and parameter-shift gradients agree to eight decimals.

The subtlety is qubit order, and it is the kind that does not raise. qmlkit is
big-endian and Qiskit is little-endian, so `to_qiskit` already maps qubit `i` to
`n-1-i`; a raw matrix carries its qubit order in its *basis* rather than its wire list,
so the basis needs reversing too, or a two-qubit gate on `(a, b)` quietly acts as
though it were on `(b, a)`. Cirq needs no reversal at all. `tests/test_cross_backend.py`
asserts that against the NumPy reference over ascending, descending and non-adjacent
wire orders and over a three-qubit custom gate, rather than reasoning about it.

`spinqit` and `torch` still refuse, for reasons that are real rather than missing work:
SpinQit's builder takes named gates rather than an arbitrary matrix, and the torch
backend differentiates *through* each gate, which a NumPy `matrix=` cannot support.
Both now say so, and name the alternative, instead of telling the caller to edit a
table.

### Fixed - `seed` did not reach everything random, so nothing was reproducible

Found while testing `n_jobs`, and much the more important of the two. `HybridModel`
passed `seed` to the quantum weights and to nothing else: the batch shuffle came off
torch's **global** RNG and so did the `nn.Linear` initialisation. So two fits of the
same seeded model disagreed, and `qk.search(seed=0)` returned a different table each
time it was called — in a library whose entire argument is that a number can be
trusted.

Both now come from a seed the model owns, and constructing one puts the global RNG
state back afterwards, so it does not silently reseed the caller's process the way
`torch.manual_seed(seed)` would have. `seed=None` still varies, by design.

### Added - `n_jobs` on `qk.search`, and `qk.parallel_map`

The one kind of parallelism that helps. Every other measurement in this project says
threading *inside* a circuit is slower than not — 0.04x on 8 threads at 8 qubits,
because the statevector fits in cache and handing the work out costs 17x the gate.
Configurations, folds and seeds are the opposite: independent, and seconds each.

`qk.search(..., n_jobs=4)` measured **1.74x**, and produces a table identical to the
serial one — same rows, same order, same means to the last bit, asserted by a test.
`qk.parallel_map` is the same thing for your own loops.

Threads rather than processes, and the ceiling is stated rather than hoped for: 1.38x
on 2 workers, 1.59x on 4, **1.75x on 6**. Well short of linear, because the GIL is
held during this library's Python-level dispatch and released only inside NumPy — the
same dispatch cost that dominates everything else here. Processes would scale further
and would have to pickle every argument across a spawn on Windows, which rules out
closures, lambdas and most model objects. A reliable 1.75x beat an unreliable 4x.

Serial is the default everywhere: a library that silently takes every core is a bad
guest inside someone else's parallel loop.

### Added - `backend="mps"` reaches 30 qubits, and `device=` on `backend="aer"`

See the MPS backend notes above. `AerBackend(device="GPU")` is a passthrough for the
separate `qiskit-aer-gpu` distribution; it is **untested here**, because that
distribution publishes no Windows wheel, and asking for a device Aer does not have
raises with what it does have rather than quietly running on the CPU.

### Added - gate fusion above 14 qubits, and `qk.recommend()` to choose for you

Gate fusion was built, measured and **ranked out** earlier: it is 0.68x at 6 qubits,
because a `2**(2k)` block matrix is comparable to a `2**n` state and 84% of the runtime
goes on building blocks. That conclusion was right for the widths this library ran at
and wrong for the widths it now reaches:

| qubits | fused | unfused | |
|---|---|---|---|
| 12 | 3.3 ms | 3.3 ms | 1.00x — off by design |
| 14 | 7.5 ms | 9.5 ms | 1.28x |
| 15 | 15.1 ms | 41.9 ms | **2.77x** |
| 18 | 68.6 ms | 383.2 ms | **5.59x** |
| 20 | 294 ms | 1,598 ms | 5.43x |

It is on automatically above `NumpyBackend.fuse_min_qubits` (14), and the block width
grows with the register — measured optimum `k=3` at 12 qubits, 5 at 15, 6 at 18, close
enough to `n // 3` to use it. The pass is greedy and order-preserving, so no
commutation argument is needed, and `tests/test_fusion.py` compares it against the
unfused path over randomised circuits at 14, 15 and 16 qubits, plus a registered custom
gate. A fusion bug is a *silent wrong number*: the circuit still runs, the state is
still normalised, and the answer is wrong.

**This moved the Aer crossover**, which is the useful part. With fusion on, the NumPy
reference now *beats* Aer at 14 qubits and the boundary sits at ~15 rather than 13 —
two more qubits before an optional dependency is worth installing.

### Added - `qk.recommend()`

`qk.plan` says what a run will cost. `qk.recommend` says what to run it on, which is a
different question with an order-of-magnitude answer:

```text
18 qubits, 159 gates - 4.0 MiB per statevector
  use  backend='aer'            AerSimulator, C++ - the NumPy reference loses above ~15 qubits
  optimisations:                gate fusion off, batching off

  considered and not chosen:
    numpy          fused, but still a Python loop at 18 qubits: 3.3x slower at 16, 11.4x at 20
    qiskit         quantum_info.Statevector is Qiskit's reference, not its fast path

  note: at 18 qubits the state is 4 MiB and no longer cache-resident, so this is
        memory-bound: more threads saturate at ~1.5x rather than scaling
```

It reads its thresholds off `NumpyBackend` rather than restating them, and a test
asserts the advice tracks the code — a recommender that drifts from what it describes
is worse than none. It reports what it did *not* choose and why, because the reasons
are the part you cannot get from the outside: below ~15 qubits this library is
dispatch-bound and the answer is always fewer, fatter calls; above it the cost is
arithmetic on an array too big for cache, and C++ wins.

### Changed - a large untracked row now says where to look

`run.report()` showed 2.5s untracked on a `QSVC` fit, which looked like a gap worth
instrumenting. It was not: `import sklearn.svm` alone is ~1.8s, and the *same run*
warm accounts for **98%** of its own time. Putting a progress bar on an import would
have been a bar that describes nothing.

So the row explains itself instead of growing an instrument — on a first run it is
mostly one-time lazy imports, and re-running shows it. The cheapest fix for a
confusing number is sometimes a sentence rather than a feature.

### Added - `backend="aer"`, and the library now works above 13 qubits

Qiskit ships two statevector simulators and this library was reaching the slower one.
`quantum_info.Statevector` is the pure-Python reference; `AerSimulator` is the C++ one,
and nothing was using it.

Adding it was not the interesting part. The first version was **32x slower** than the
backend it was meant to beat, and profiling said why: `transpile()` cost **57 ms
against 1.1 ms** for the simulation it was preparing. Aer runs this library's gate set
directly — including the `UnitaryGate` a registered gate arrives as — so the transpiler
had nothing to contribute and fifty times the price. Removing it is the whole
optimisation.

What that buys is a different ceiling rather than a percentage:

| qubits | `numpy` | `qiskit` | `aer` |
|---|---|---|---|
| 10 | **0.46s** | 0.98s | 0.59s |
| 12 | **0.92s** | 1.60s | 0.97s |
| 14 | 1.52s | 2.16s | **1.27s** |
| 16 | 9.87s | 13.03s | **0.60s** |
| 18 | 25.91s | 35.99s | **0.66s** |

The crossover sits near **13 qubits** and it is sharp, because it is the same boundary
that turns off the NumPy backend's batching (`batch_max_qubits`, 10): above it the
reference falls back to a Python loop while Aer stays in C++. At 18 qubits that is
**25.91s against 0.66s**.

It is registered as its own name rather than as a silent upgrade to `qiskit`, because
which simulator produced a number belongs in the code that produced it. It inherits
circuit translation, qubit order, the registered-gate matrix fallback and every
measurement semantic from `QiskitBackend` and the base class unchanged, and it joins
`tests/test_cross_backend.py` like any other backend.

### Fixed - the slow backends were the silent ones

`qk.progress()` tracked the pair-at-a-time kernel loop and `HybridModel.fit`, which
meant it reported on the *fast* path and said nothing on the slow one. An exact Gram
matrix on Qiskit took 140x what NumPy took and printed nothing at all, because from the
caller's side it is a single `statevector_batch` call — and the thousands of
simulations happen inside `Backend.statevector_batch_slots`, which was a bare list
comprehension.

That loop is now where progress is reported from, which fixes it for every backend
without a native batch at once — Qiskit, Cirq, SpinQit, and anything registered later:

```text
qiskit circuits  471/780   60%  1.2s elapsed  ~0.8s left
```

`NumpyBackend` overrides that method with one vectorised call and so reports nothing,
which is correct: there is no incremental progress there, and inventing some would be a
bar that describes nothing. Both behaviours have a test.

Overhead measured at **5 us per row against 960 us of work** — 0.5%, and only when
something is watching.

### Added - the extension guide covers the parts that have no registry

`docs/guides/extending.md` listed four registries; there are eight, and it now lists
all of them with what registering actually buys. It gains three sections: how far a
registered gate travels across backends, **a new optimiser** (`OPTIMIZERS` is an open
dict, so an optimiser is a function and needs no registration), and **a new algorithm**
— the one extension point with no registry, deliberately, because an algorithm is the
outermost layer and nothing inside the library needs to find it by name.

## [0.2.0] - 2026-09-13

**Upgrading from 0.1.0? This release contains eleven correctness fixes as well as the
features below, and four of those defects corrupt a result silently.** They were
prepared as 0.1.1, which was never published — everything in that section further down
ships here. The worst of them is in the torch backend, which computed a *different
circuit* from every other backend, so `backend="torch"` and `method="backprop"` returned
wrong numbers and nothing raised. If you are on 0.1.0, upgrade.

### Fixed - a documented optimiser that never ran

`OPTIMIZERS` has four entries and two of them need the gradient injected by the caller,
but every injection site named `gradient-descent` alone. `optimizer="adam"` therefore
raised `TypeError: _adam() missing 1 required keyword-only argument: 'grad'` from `VQE`,
`QAOA` and `AdaptVQE` alike — one of the four documented optimiser names could not be
used from any of the three algorithms that list it.

`tests/test_optimizer_wiring.py` now parametrises over `OPTIMIZERS` itself rather than a
hand-written list, so a fifth entry cannot be added without being covered, and one test
drives both gradient routes to the known ground state — an injected gradient that was
*wrong* would pass a wiring test and fail that one.

Also fixed: two docstrings whose LaTeX had been eaten by a shell heredoc (`\rho` and
`\rangle` became line breaks, `\theta` became a tab), and three references to things
that do not exist — `kernel.matrix(X)` in a user-facing error message where the call is
`kernel(X)`, `qk.datasets.moons` in a live doctest where it is `make_moons`, and an
"AmplitudeEncoder feature map" in the PennyLane alias table.

### Added - you can watch a run instead of waiting for it

`qk.plan` says what a run will cost before it starts and `qk.diagnose` says what went
wrong after it finishes. In between there was nothing, and in between is where the
hours go — a quantum kernel on a few hundred points is tens of thousands of circuits
behind a single silent call that returns when it returns. "How much is left" is one
of the most common questions asked about every library in this field, and none of
them answers it.

```python
with qk.progress() as run:
    gram = kernel(X)
    model.fit(X, y)
```

```text
kernel gram  3,412/12,720   27%   14.2s elapsed  ~37s left
```

and afterwards, where the time actually went:

```text
>>> print(run.report())
Run finished in 51.4s
kernel gram               12,720 items    47.9s    3.77 ms/item
fit VQC                      600 items     3.2s    5.33 ms/item
  (untracked)                              0.3s   - setup, data handling, and anything outside a task
```

Three properties it holds to, in priority order, each with a test:

1. **It does not change any number.** A seeded kernel and a seeded circuit produce
   bit-identical results watched and unwatched. A reporter that perturbed a result
   would be a worse defect than the silence it replaces.
2. **It is free when nobody is watching.** With no active reporter, the tracking
   calls inside the library cost one comparison against `None`, and the live line
   redraws at most ten times a second however fast the loop runs.
3. **It does not claim to know what it does not.** An estimate from four items in
   half a second says more about scheduling noise than about the run, so it prints
   `estimating` until there is evidence — the same rule the rest of the library
   follows about reporting a measurement.

Wired into the two loops that actually take the time: the pair-at-a-time kernel Gram
(sampled kernels, non-inversion estimators, and any backend without a statevector)
and `HybridModel.fit`, which covers `VQC` and `VQRegressor`. `qk.track` wraps any
iterable, and `qmlkit.progress.task` is what library code calls — it returns a silent
stand-in when nothing is watching, so no loop needs to branch on it.

Nothing is printed unless `qk.progress()` is entered, and the live line goes to
stderr so piping a script's stdout to a file does not collect redraws.

### Added - the run writes itself up

A run produces a number, and a month later the number is all that is left. A run now
also records its *trajectory* — `run.log(name, value, step)` — and can write the whole
thing out as one HTML file:

```python
with qk.progress() as run:
    run.note(dataset="breast-cancer", seed=0)
    model.fit(X, y)

run.save_html("run.html")
```

`HybridModel.fit` logs three series per epoch, so `VQC` and `VQRegressor` get this
without asking: **loss**, **gradient norm** and **parameter norm**. The second is
there because loss alone cannot tell a solved problem from a plateau, and the third
because weights running away looks like nothing at all in a loss curve. Computing
them costs a pass over the parameters, so it happens only when something is watching.

The page is deliberately a *file*, not a server. A dashboard you have to start,
connect to and keep alive is a dependency, a port and a process; a page you can open,
email, and drop next to the result in a directory is none of those and outlives all
of them. The charts are inline SVG for the same reason: nothing to fetch, nothing to
pin, nothing to break in two years. Tests assert it — no `<script>`, no `src=`, no
`http`, and every label and metadata value escaped, because a report is exactly the
kind of artefact that gets passed around.

Both chart edge cases are handled and tested because both are real runs: a single
point has no range to scale against, and a flat series has zero range — which is the
loss curve of a model that never learned, the run you most want to look at. That one
is labelled `flat — never moved` rather than drawn as a misleading straight line at
an arbitrary height.

### Added - `diagnose` can now be asked whether the quantum layer earned its place

`qk.diagnose(model, X, y)` takes a *trained* model and the data it was trained on,
and answers the question the structural checks cannot: not "is this architecture
broken" before the run, but "it trained, it converged, and it works just as well
without the quantum part". That complaint is the most common one in the field and
nothing in any library answers it.

The new finding is `QUANTUM_LAYER_BYPASSED`. A forward hook captures what the
quantum layer received and what it returned; the same hyperparameter-free probe
scores both, averaged over five splits. The finding fires only when separability
measurably *dropped* across the layer by more than the spread across those splits —
the same verdict rule `qk.baseline` uses, where a lead inside the spread is not a
lead. It reports both numbers, so the claim can be argued with:

```text
[warning] QUANTUM_LAYER_BYPASSED: The quantum layer reduced separability: the same
probe scores 0.951 on what the layer received and 0.578 on what it returned, a drop
of 0.373 against a spread of 0.028 across 5 splits. The classical layers around it
are carrying this model.
```

The probe is `NearestCentroid` precisely because it has no solver and no
hyperparameter: neither side of the comparison can win by having been tuned better,
so a difference in score is a difference in the activations. Classification targets
only — a continuous target is declined rather than guessed at. Without `X` and `y`,
or without torch, `diagnose` behaves exactly as before.

`X` and `y` are optional positional parameters, so every existing call is unchanged.

## [0.1.1] - prepared, never published

This version was cut, verified and then superseded before it was tagged: the tree
gained features while it sat, so the correctness work below shipped in **0.2.0**
instead. There is no `0.1.1` on PyPI and there never will be. The section is kept in
full because it is the account of eleven defects that were in 0.1.0, and a reader
upgrading from 0.1.0 needs it.

**Upgrade from 0.1.0.** Everything below was found after 0.1.0 was published, so it is
all still present in the version on PyPI. One defect was reported by a reader using the
library; ten came from an adversarial audit told to find a case where qmlkit returns a
*wrong number* rather than an error. Four of the ten corrupt a result silently, and the
worst of those is in the torch backend, which computed a different circuit from every
other backend - so `backend="torch"` and `method="backprop"` returned wrong numbers,
reachable from the built-in `conv_block(filter="su4")`. Nothing raised in any of the
four.

The audit's account is split over three sections below by where each defect lived:
*three silent wrong numbers*, *five defects*, and the diagnostic in *a diagnostic that
asserted what it had only inferred*. They are one audit, settled against one dense
reference simulator that shares no code with qmlkit.

### Fixed - a composition the README recommended built the wrong circuit

Reported by a reader who tried the two-feature-map example and read the parameter
indices off the built circuit.

**Two feature maps in one model shared each other's angles.** `EncodingLayer`
reserved circuit input slots per *angle* rather than per feature map, always taking
slots `0..n_angles-1`. So in the composition the README advertised verbatim -

```python
# docs: skip - this is the defect, kept as it was written
qk.Ansatz(2, qk.EncodingLayer(zz) + qk.RotationLayer("ry") + qk.EncodingLayer(angle),
          n_inputs=3)
```

— the `ZZFeatureMap` took slots 0, 1, 2, the rotation layer took 3, 4, and the second
encoding took **0 and 1 again**. Those slots already held the ZZ map's *transformed*
angles: `ZZFeatureMap(2).angles([0.3, 0.7])` is `[0.6, 1.4, 13.876]`, because a Z term
follows the `Rz(2 phi)` convention. The trailing `Ry` therefore encoded `2*x_i` where
the caller asked for `x_i`. Nothing raised. The circuit built, bound, differentiated,
trained and converged - on the wrong numbers. That is the failure class this library
exists to refuse, in an example of its own.

Each feature map now owns a **disjoint** range of input slots, allocated in the order
the maps first appear; the *same* map re-used keeps the range it already has, which is
what makes re-uploading feed the same data in again rather than consume new features.
`Ansatz.angles` concatenates the maps' angles in slot order and `Ansatz.angle_jacobian`
stacks their Jacobians, so `bind(x, weights)` and the chain rule down to the data both
follow. A composed model now also satisfies the `Combined` protocol, so it can go
straight into a `QuantumLayer` - which the README promised and which had never worked.

Slots could not instead be keyed by *feature*, so that both maps read the raw `x`:
a slot is referenced by a `ParamRef`, which is affine in one parameter
(`scale * theta[i] + offset`), and a Pauli map's higher-order angle is
`2 * prod_j (pi - x_j)` - nonlinear, in several features at once. The map from features
to angles has to stay classical, which is what `angle_jacobian` is for.

**The error message steered users into the bug.** With `n_inputs=2` the same
composition raised *"the circuit reserves 2 input slots but this feature map needs 3"*,
which tells you to raise `n_inputs` - and raising it to 3 is exactly what produced the
silent collision. `n_inputs` is now **inferred** from the block, so there is no count
to get wrong; passing one that disagrees with what the encodings reserve raises and
names the sum. This closes the last hand-counted number in `Ansatz`, whose docstring
already promised that parameter counts are "inferred from a dry build, never
hand-counted, so a miscount is not a failure mode".

**The README example is fixed, and now runs.** `tests/test_docs.py` executes every
Python block in `docs/`, but never reached `README.md` - the most-read page was the
one page not under the executable-documentation rule, which is why this survived. The
README is a reference rather than a tutorial and most of its blocks are deliberate
fragments naming an API, so it opts *in*: a self-contained block marked `# docs: run`
is executed by `test_readme_blocks_marked_runnable_do_run`.

### Fixed - observable arithmetic that QML cost functions need

`I - Z` and its relatives are how a projector is written, and most of the ways to
write one did not work. `Z(0) + Z(1)` and `2.0 * Z(0)` did; these did not:

| Expression | Was |
|---|---|
| `sum([Z(0), Z(1)])` | `TypeError` - no `__radd__` for `sum`'s `0` start value |
| `Z(0) + 1` | `AttributeError: 'int' object has no attribute 'terms'` |
| `1 - Z(0)`, `Z(0) - Z(1)`, `I() - Z(0)` | `TypeError` - no `__sub__` or `__rsub__` |
| `-(Z(0) + Z(1))` | `TypeError` - `PauliSum` had no `__neg__` |
| `Z(0) / 2` | `TypeError` - no `__truediv__` |

`PauliString` and `PauliSum` now implement `__add__`/`__radd__`, `__sub__`/`__rsub__`,
`__neg__` and `__truediv__`. A scalar promotes to that multiple of the identity, and
the additive identity is dropped rather than carried as `0*I`, which is what lets
`sum(...)` return the sum itself. Numbers are recognised through `numbers.Complex`, so
NumPy scalars work - `np.int64` is not an `int` subclass. An operand with no sensible
reading returns `NotImplemented`, so `Z(0) + "x"` reports unsupported operand types
instead of failing somewhere inside qmlkit.

### Fixed - five defects found by an adversarial audit

An agent was asked to break qmlkit 0.1.0 through its public API, and to settle every
disagreement against a dense simulator it wrote itself - one that reads `spec.ops` and
never calls qmlkit's own execution code. Most of the library held: all 20 gate
matrices column by column, four exact gradient routes against an analytic product
rule, every metric against scikit-learn, the noisy backends against their closed form
to 1e-16. What it found was the other kind of defect - the one where a plausible
number comes back and nothing raises. Five of those were in the kernel, analysis and
metric layers.

**`QuantumKernel(estimator="hadamard")` returned `Re(<x'|x>)^2` instead of
`|<x'|x>|^2`.** The wrapper squared `hadamard_test`, whose `part` defaults to
`"real"`, so wherever a feature map produced a complex overlap the estimator dropped
the imaginary component. Worst observed error 0.831, on a quantity bounded by 1. The
three estimators are documented as agreeing on a simulator; two of them did.

Nothing downstream could catch it. Squaring the real part alone leaves a Gram matrix
that is still symmetric, still positive semi-definite and still unit-diagonal -
`K(x, x)` is real for every feature map, so the diagonal is 1 either way - and
`diagnose` therefore reported nothing. A `QSVC` fitted on the wrong kernel scored
*higher* than one fitted on the right one. A Hadamard test measures one *component*
of a complex overlap, so the modulus takes two circuits: the estimator now runs both
and returns `Re^2 + Im^2`, and `n_evaluations` reports 2 per pair rather than 1,
because a caller budgeting circuits is entitled to the real count.

**`reduced_dm` silently ignored the order of `wires`.** It did
`keep = sorted(set(wires))`, so `reduced_dm(psi, [1, 0])` returned the `[0, 1]`
matrix. That result is still Hermitian, still has trace 1 and still has the right
eigenvalues, so `purity`, `vn_entropy`, `mutual_info` and every other
basis-independent reading built on it agreed with the truth - which is how a
subsystem-ordering defect sits under a passing test suite indefinitely. `wires` is now
honoured as given: qubit `wires[0]` is the returned matrix's most significant bit, and
`[1, 0]` is `SWAP . rho . SWAP` rather than `rho`. `[0, 0]` used to de-duplicate itself
into a 2x2 matrix and now raises, naming the repeated qubit. The PennyLane parity
suite compares descending pairs against `qml.math.reduce_dm`, so a second
implementation holds the convention in place.

**`KERNEL_AT_CONCENTRATION_SCALE` fired on kernels that separate their classes
perfectly, and its message contradicted its own numbers.** Two defects in six lines.
The rule compared against `2 * kernel_spread(n)` while the message printed
`kernel_spread(n)`, so it reported *"spread 5.00e-01 is at or below ... 2.50e-01"* - a
sentence refuted by the two numbers inside it. And because off-diagonal kernel entries
live in `[0, 1]`, their standard deviation cannot exceed 0.5, which is exactly the
threshold at two qubits: below three qubits the check fired on **every** Gram matrix,
including one built from mutually orthogonal points. `bool(qk.diagnose(K))` was
therefore true for a good kernel - the same cry-wolf failure as `ENCODING_COMMUTES`
above, and the one that teaches people to stop reading diagnostics at all. The message
now quotes the threshold it actually used, and the check is skipped where that
threshold exceeds what the statistic can attain, because a comparison nothing can pass
is not a test.

**`geometric_difference(K, K)` returned `sqrt(N)` rather than 1.** Huang et al.'s `g`
is compared *against* `sqrt(N)`; that threshold had been folded into the statistic
itself, which left a number agreeing with no published one and a self-comparison whose
value depended on the sample count. It now returns
`sqrt(||sqrt(K_Q) K_C^-1 sqrt(K_Q)||)`, which is 1 for identical kernels, and the
docstring names `sqrt(N)` as the bar for the caller to apply. The kernel study and the
credit-risk example both compared `g` against a hard-coded `10`; both now compare
against `sqrt(N)`, which is the literature's test and the only one that scales with
the data.

**`evaluate.regression` reported `r2 = 0.0` for a perfect fit on a constant target.**
R2 scores a model against the variance baseline - predict the mean everywhere - and a
constant target has no variance, so the ratio is `0/0`. Reporting 0.0 made a perfect
prediction indistinguishable from one wrong by a factor of 33: both scored zero, in
the module whose premise is that a metric says when it is misleading. `r2` and
`explained_variance` are now `nan` there, with a note naming the value every target
takes and pointing at `mse` and `max_error`, which need no baseline. This is a
deliberate departure from scikit-learn, which returns 1.0 for the perfect case and 0.0
for the rest - a convention that cannot be read back, since a 0.0 could mean either
"undefined" or "no better than the mean".

### Fixed - three silent wrong numbers, found by attacking the library

An agent was asked to break qmlkit: to find a case where it returns a *wrong number*
rather than an error. Its ground truth was a dense simulator it wrote itself, which
reads only `spec.ops`, builds every gate matrix by hand, and never calls qmlkit
execution code. It found ten defects; these three were the ones that corrupt a result
without saying anything.

**The torch backend computed a different circuit.** `_apply_torch` reimplements
`np.moveaxis` and drops the `sorted(zip(destination, source))` numpy does, so for a
two-qubit gate on wires `(a, b)` with `a > b` the *untouched* wires came out permuted.
`Z(3)` on a four-qubit circuit read `-0.1288` where NumPy, Qiskit and Cirq all agreed
on `-0.7374`. `method="backprop"` differentiates through the same function, so its
gradients were wrong too, and `qk.conv_block(filter="su4")` reaches it without anyone
hand-writing a circuit. `qk.selfcheck` catches this and names the cause exactly -
nothing was running it.

**`expectation(..., return_std=True)` reported a fabricated error bar.** It fed any
observable into the single-Pauli formula `sqrt((1 - z^2)/shots)`, which for a sum is
too tight, too loose, or - once `|<O>|` reaches 1 - *exactly zero*. Every molecular
Hamiltonian came back with `+-0.00000`, an error bar that looks converged and does not
move with the shot count. It is now computed rather than approximated: inside a
qubit-wise-commuting group every term is diagonal in the measured basis, so the
group's variance follows from the probabilities the exact path already reads, and
groups measured on independent shots add. Checked against the empirical spread of 400
repeated samplings, it agrees within 2% for single terms, sums, weighted sums and
two-basis observables alike.

**`purity(state, backend=<mixed-state backend>)` returned a hard-coded 1.0**, on the
premise that a statevector is pure by construction - true, and not what was asked when
the caller named a density-matrix backend. It reported 1.0 for a state whose purity
was 0.309.

### Added - property-based torture tests

`tests/test_torture.py`: thirteen invariants that hold *by mathematics* rather than by
design, checked over randomly generated circuits with Hypothesis, which shrinks a
failure to the smallest circuit that still shows it. `hypothesis` had been a dev
dependency and was unused.

The properties are deliberately not comparisons against stored values: a test that
pins today's output catches a change, a test that pins an invariant catches a mistake,
and only the second is worth running against random input. All exact gradient routes
agree; every backend agrees with the reference; batched equals looped; a tied weight's
gradient is the sum over its occurrences; adjoints undo, states normalise, expectation
is linear in the observable and inside its spectrum; Qiskit and Cirq round trips
reproduce statevectors; sampling lands inside its error bars. Depth is tunable with
`QMLKIT_TORTURE_EXAMPLES` - cheap in CI, and a campaign at 1,500 examples per property
(~19,500 circuits) passes clean.

One caveat worth recording: the backend-agreement property originally skipped torch,
on the reasoning that a differentiable simulator is a different kind of backend. That
exclusion is exactly what let the permutation bug above survive. It no longer skips it,
and the reason is written into the test.

### Added - Adam, selective classification, and Study 8

**`adam`** joins `rotosolve`, `spsa` and `gradient-descent`. Its absence was found
the hard way: an agent reproducing a published method outside the torch bridge had to
hand-roll one, and its first run put the paper's method *below* the baseline - its own
diverging optimiser, not the method. A hand-rolled optimiser that diverges looks
exactly like a technique that does not work. `qmlkit.optim.minimize_adam`, with
`adam_step` and `AdamState` public for when the loop is yours; keep the state between
steps or Adam quietly becomes gradient descent with a decaying learning rate.

**`qk.evaluate.selective` and `qk.evaluate.risk_coverage`** score a classifier that is
allowed to decline. Selective accuracy - accuracy on the samples the model chose to
answer - rises monotonically as it abstains more, reaching 1.000 on the single sample
it is surest about, so quoting it against a model that answered everything compares
two different questions rather than two models. `selective` reports coverage beside it
and makes the *comparable* number the primary one, so `scores.score` cannot quietly
become the flattering one. `risk_coverage` gives the whole trade plus AURC, which
abstaining more cannot inflate.

**Study 8** measures it on a real model: a `VQC` scoring 0.8947 answering everything
climbs to a selective **0.9479** at threshold 0.80 while its comparable accuracy
*falls* to **0.7982**. Abstention made the reported number better and the model worse,
and only one of the two columns says so.

### Fixed - a diagnostic that asserted what it had only inferred

**`ENCODING_COMMUTES` reported an error on correct architectures.** The check was
structural: matching rotations implied a collapse. But `Ry(x) Ry(t) Ry(x) Ry(t)`
merges *on one wire with nothing in between* - any entanglement breaks it, and the
check could see neither an entangler in the trainable block nor an entangling feature
map. Measured against the library's own `fourier.spectrum`, it claimed one frequency
where the band was `0..4`. It now uses the structural test as a trigger and confirms
with the spectrum before reporting. A false positive in the honesty layer is worse
than a false negative: it teaches people to ignore the tool.

The same blindness sat in `reupload()`'s construction-time warning, whose message had
a second defect - it interpolated the `rotations` *parameter* rather than the gates
actually found, so passing an explicit block produced "the trainable block only uses
('rz','ry','rz') ... Use a non-commuting block such as ("rz","ry","rz")", recommending
the thing it was complaining about.

**`list_baselines()` repeated a name registered for both tasks.** The registry is
keyed by task and name deliberately, so `rbf-kernel-ridge` serves classification and
regression; the listing read the values and never deduplicated.

**`Scores.get("precision")` returned `None`.** `__getitem__` already answered a wrong
key with a did-you-mean and `.get` bypassed it, so a near-miss became a silent `None`
that surfaced later as a `TypeError` from inside numpy. An explicit default is still
honoured without comment.

### Changed

- `Ansatz(..., n_inputs=)` now defaults to `None`, meaning *infer*. Existing calls that
  passed the correct total keep working; one that passed a different number now raises
  rather than silently building a circuit whose encodings overlap.
- `Ansatz` gained `feature_maps`, `angles`, `angle_jacobian` and `n_features`.
  `ReuploadModel`'s own `angles`/`angle_jacobian` were identical for its single map and
  are now inherited.
- Composing feature maps that read different numbers of features now raises. Every map
  in one model is handed the same `x`, so such a model could never have been bound.
- `scripts/verify_install.py` compared `__version__` against a hardcoded `"0.1.0"` - a
  third copy of the version that had to be bumped by hand, and the gate failed on this
  release for that reason alone. It now reads the installed distribution metadata and
  checks the module constant against it, which is what the check was named for.
- `QuantumLayer` treats a model as carrying its own encoding only when it reserves
  input slots. Every `Ansatz` can map data onto its slots now, so the four-attribute
  check alone no longer distinguishes a model from a bare ansatz.
- **`geometric_difference` now returns values `sqrt(N)` times smaller than 0.1.0's.**
  Code that compared it against a hand-picked constant will read differently and
  should compare against `sqrt(N)` instead, which is the comparison the statistic was
  always for.
- `QuantumKernel(estimator="hadamard").n_evaluations` counts two circuits per pair
  rather than one, which is how many it now runs.

## [0.1.0] - 2026-09-12

First release.

### Fixed - two defects found by using the library, not by testing it

An independent agent was given a dataset and told to build the best classifier it
could with qmlkit, knowing nothing about the project. Its report found two things the
test suite could not, because both were about the API rather than the arithmetic.

**`VQC` could not express data re-uploading.** `HybridModel.__init__` did
`ansatz or hardware_efficient(...)`, so an ansatz was *always* supplied and a
re-uploading feature map collided with it:
`ValueError: a re-uploading model already contains its trainable block`. The pattern
this library recommends most was unreachable from the class it recommends first, and
the only way through was a hand-written training loop. A re-uploading model now goes
in as the feature map with `ansatz=None` supplied automatically, and
`docs/tutorials/07-reuploading.md` has a "Training one" section, which it never had.

**A dead parameter read as a barren plateau.** `gradient_variance` probes one
parameter and defaults to index 0, which on several stock ansaetze is a leading `Rz`
on `|0>` whose gradient against `Z` is identically zero. `AnsatzReport` printed
`1.4e-32` under the gloss "higher = more trainable" — indistinguishable from a
catastrophic plateau, and really "you probed a parameter that does nothing" — while
`diagnose()` called the same ansatz `DEAD_WEIGHTS`. Two of this library's own tools
contradicting each other, in the part of it that exists to stop people believing
wrong numbers. `gradient_variance` now warns when a variance is machine zero rather
than small, and `AnsatzReport` probes the first parameter that actually moves the
readout and names which one it used.

**`print(qk.draw(spec))` crashed on a Windows console.** The diagram uses
box-drawing glyphs and the default Windows code page is cp1252, which encodes none of
them, so looking at a circuit raised `UnicodeEncodeError` from inside the caller with
a traceback naming the codec rather than `draw`. It happened after the model had
already trained, on the first thing a new user does, on the platform where it breaks.
`draw` and `probabilities_bar` now check whether `sys.stdout` can encode what they are
about to return and degrade to ASCII when it cannot, at identical column widths;
`ascii=True`/`False` overrides the detection.

**`mypy` failed on Python 3.10 only, and no local environment could show it.** NumPy
2.3 gave `ndarray`'s shape parameter a default, so mypy stops comparing a
`tuple[int, ...]` against a `tuple[int]`. NumPy 2.2 is the newest release that supports
3.10 and has the parameter without the default, so six assignments that are fine
everywhere else are errors there: a variable whose first binding is 1-D by inference,
reassigned something of unconstrained rank. Found by CI, reproduced on a purpose-built
3.10 + numpy 2.2 environment, fixed by annotating those variables `npt.NDArray[Any]` at
their first binding - the shape-agnostic spelling used elsewhere, which says what was
always true about them. Now clean on four combinations: 3.10/numpy 2.2, 3.10/numpy 1.26,
3.14/numpy 2.5, and a no-torch install.

**CI never installed `qiskit-aer`, so the backend that needs it was never tested.**
The `full` job installed every extra except that one, and every test in
`tests/test_noisy_backends.py` guards itself with `is_available` - so the whole
qiskit-aer noisy backend skipped itself and the job went green anyway. The workflow's
own comment says an SDK job that proves nothing is worse than no SDK job; this was
that. `aer` is now in the full job's install and in its import assertion.

**The suite passed only where the optional extras happened to be installed.** Both
CI's lint job and the release workflow's verify job install `[dev]`, without torch or
scikit-learn, and nine tests failed there while passing locally. Three causes, all
pre-existing:

- `tests/test_search.py` never guarded on torch, unlike the ten other test modules
  that need it. `search()` fits a `VQC`, which is a torch model. Six of its
  twenty-three tests build one, so the guard went on those six rather than the module,
  which would have thrown away the other seventeen.
- Two documentation blocks used `VQC` and `QSVC` without declaring the extra they
  need.
- A third failed for a subtler reason worth recording: blocks on a page share a
  namespace and run in order, so skipping one leaves the *next* one without the
  `import qmlkit as qk` it was relying on, and it fails with `NameError` rather than
  anything that names the real cause. A block that depends on a skipped block needs
  the same directive.

**`mypy` passed only where torch happened to be installed.** Widening the type check
to the whole package brought `qmlkit.nn` into it, and those modules subclass
`torch.nn.Module`. torch is an optional extra, so where it is absent the base class
resolves to `Any` and `--strict` refuses to subclass it: 13 errors in an environment
with no torch, none in one with it. Both CI's lint job and the release workflow's
verify job install `[dev]` without torch, so this would have failed the release before
it built anything. `qmlkit.nn.*` now exempts the three rules that are about
*third-party* base classes and decorators, and the package type-checks identically
with and without torch. Everything else stays strict.

### Packaging

Things nobody sees until the page is live, and a PyPI version cannot be reused to fix
them:

- The summary PyPI shows in search results was the category sentence the README
  deliberately stopped opening with.
- The README's links to `HANDOFF.md`, `AGENTS.md` and `RELEASING.md` were relative.
  PyPI renders the README standalone, so all three were 404s for every reader.
- `Documentation` and `Changelog` were missing from `[project.urls]`, which is the
  sidebar on the project page. Both exist.
- `NOTICE` was not in the wheel. Apache-2.0 section 4(d) expects it to travel with
  the distribution.
- Added the `Operating System :: OS Independent` and `Intended Audience :: Developers`
  classifiers.

### Changed - documentation

- The README leads with what the library is *for* rather than what category it is in.
- `adjoint` vs `backprop` is now measured in the gradient guide. The intuition travels
  badly between libraries: where every circuit evaluation goes through a dispatch
  layer, `backprop` can be dramatically faster because it pays that cost once rather
  than once per parameter. qmlkit's adjoint is a direct NumPy sweep with no dispatch to
  amortise, so the ranking inverts and adjoint wins by 3-4.5x standalone, more when
  batched. `method="auto"` already picks it.
- `QuantumKernel.bandwidth` is documented. It is the first thing to reach for when a
  kernel has concentrated, and it had no prose anywhere — only a default in a
  signature.
- The generated API index (`docs/llms-full.txt`) carries the **whole** docstring for
  classes rather than its first line. Constructor *semantics* live in the body, and a
  signature alone sent readers to the source; this costs about 4% of the file.

### Added - `mypy` over the whole package

`[tool.mypy] files` listed six paths during development; it lists `src/qmlkit`. A partial check reads
like a full one in CI, which is the worst of both - `kernels`, `nn`, `algorithms`,
`encoding`, `evaluate`, `metrics` and the rest were never checked.

Making the other 59 errors pass was mostly a matter of annotations that were wrong
rather than missing, and three of them were worth the trip:

- **`expectation()` is now overloaded.** It returns `float`, or `(float, float)` with
  `return_std=True` - a *flag*, which is what `@overload` is for. Seven callers were
  writing `float(value)` on a union mypy could not narrow, two of them behind a
  `cast()` that is now gone. Every user's IDE was being told the same wrong thing.
- **`ir.bound_angle()`**, one shared "this parameter must be bound by now" check.
  Reading angles off a circuit was `float(p)` in four places, which on an unbound
  circuit raises a `TypeError` about `__float__` that says nothing about circuits.
  Two backends had already grown a private version of it.
- **`nn.layer.Combined`**, a Protocol naming what `_is_combined` had been checking by
  `hasattr`. Re-uploading is a pattern rather than a class, so there is no base to
  test against - the four attributes that make a model combined are now written down,
  and `_is_combined` is a `TypeGuard` that narrows to them.

Signatures taking a parameter vector or a feature vector now say `ArrayLike` rather
than `Sequence[float]`, which is what every caller was already passing.

One real bug surfaced: `draw()` used the name `col` for both a column *index* and a
column's *contents* in the same function. Renamed, and mypy was right.

### Added - `from_cirq`

The fourth importer, completing the set: `from_qasm`, `from_qiskit`, `from_pennylane`,
`from_cirq`. Cirq is the one with no gate names to look up - `cirq.S`, `cirq.T` and
`cirq.rz` are all a `ZPowGate` separated only by exponent and `global_shift` - so it
classifies gates rather than reading a table, and `global_shift` is read rather than
ignored because it is a global phase alone and a *relative* one inside a controlled
block. `sympy` symbols become `ParamRef`s carrying their scale, so `cirq.rx(2 * t)`
imports as `ParamRef(i, scale=2.0)` and binding reproduces `cirq.resolve_parameters`
exactly. Nonlinear expressions and fractional two-qubit powers are refused, not
approximated; `X**s` decomposes to `Rx` with the dropped-phase warning the `u3` family
already uses. Cirq is big-endian like qmlkit, asserted on statevectors over random
circuits rather than assumed, and every gate `to_cirq` emits is asserted to come back.

### Added - mixed-state backends

Two backends that evolve a density matrix and take a noise model: **`cirq-density`**
(`cirq.DensityMatrixSimulator`) and **`qiskit-aer`**
(`AerSimulator(method="density_matrix")`, behind the new `aer` extra, since Aer is a
separate distribution from Qiskit). `NoisyBackend` supplies `density_matrix()` and
`purity()`; everything above the primitive - basis rotation, qubit-wise-commuting
groups, sampling, expectations - is the base class's, unchanged.

Three decisions worth recording:

- **Noise never selects a backend.** `get_backend(noise=...)` without a named
  mixed-state backend raises and names the ones that would work. A mixed-state run
  costs more, refuses two gradient methods and answers a different question, so which
  simulator produced a number stays visible in the code that produced it.
- **`supports_exact` stays true while `supports_statevector` goes false**, and those
  are not the same flag. A density matrix gives a shot-free expectation - exact
  *given the noise model* - which keeps decoherence and shot noise separable.
  `adjoint` and `backprop` refuse; `parameter-shift` and `grad_batch` work, because a
  shift rule never inspects a state.
- **With no noise model these backends reproduce the pure-state ones to machine
  precision**, asserted across circuits, observables and both SDKs. Depolarizing
  noise is checked against the closed form `cos(theta)(1 - 4p/3)` rather than only
  against itself.

`Backend.expectation` gained an exact path for backends that have probabilities but
no statevector, sharing `_group_circuit` with the sampled path so the two cannot
drift apart. `docs/guides/noise.md` has the measured gradient decay, the traps, and
the boundary: error mitigation belongs to Mitiq and error correction to Stim.

`diagnose()` works on these backends, which it did not at first: `_dead_parameters`
and `entangling_capability` compare *statevectors*, so it raised `NotImplementedError`
from four frames down - the diagnostics being the thing that breaks is the worst
version of that bug, since it is what the caller reached for to find out. They are
questions about the *ansatz* rather than the device, so they now run on the exact
reference and the report's subject line names the substitution. Answering them from
probabilities instead would have called a phase-only parameter dead.

Five measures that are defined only on a *pure state* now refuse by name instead of
dying inside `statevector()`: `expressibility`, `entangling_capability`,
`fidelity_samples`, `metric_tensor` and `qng_step`. One shared
`require_statevector(backend, measure)` names the measure and the backend, and points
at `purity()` for the question a mixed state can answer. They refuse rather than
substituting the reference, which is the opposite of what `diagnose()` does and
deliberately so: there the question is about the ansatz, here the caller named a
backend and asked for a measure on it.

`FLAT_GRADIENTS` no longer claims "gradients are exact here" on a backend carrying a
noise model. The flag to ask is **`supports_statevector`, not `supports_exact`** -
the latter is deliberately true on a mixed-state backend, where it means shot-free
rather than undisturbed. The threshold itself is unchanged and still calibrated on
exact gradients, so the finding fires more readily under noise; the message now says
which of the two it is measuring.

### Added — seven worked case studies

`docs/studies/` — whole problems from raw data to a defensible number, on a different
axis from the tutorials: those show how each piece works, these show a problem being
decided.

1. **Imbalanced classification** — the metric that lies, and the one keyword that fixes
   the model. The compressed form of `examples/credit_risk.py`.
2. **Is a quantum kernel worth it?** — concentration and geometric difference, both
   answerable before the Gram matrix is fitted to anything, and the case where they
   point opposite ways.
3. **Regression** — why `r2`, `rmse` and `mae` disagree, and the Fourier-frequency
   ceiling that no learning rate moves.
4. **Chemistry, H2 ground state** — `hardware_efficient(4, 2)` converges confidently to
   `-0.536 Ha`, 601 mHa out and worse than doing nothing, while `(4, 4)` reaches
   `-1.137306` exactly. Nothing raises in either case.
5. **Clustering and generative models** — the metrics that exist because accuracy does
   not apply, including why total variation is the primary and KL is not.
6. **Clinical decisions** — breast-cancer biopsies, where a false negative and a false
   positive are not the same error and no single-number metric knows it. Accuracy 0.93
   and five missed cancers are the same model; the confusion matrix and a threshold
   sweep are what you decide on.
7. **Images and structure** — a QCNN on handwritten digits: weight tying is what makes
   it a convolution rather than a sparse ansatz, and the filter is a registry choice
   that `compare_ansatze` scores before anything is trained.

Studies 6 and 7 are the documented forms of experiments 3 and 2 in
`examples/experiments.py`, which predate the credit-risk work and had no home in the
documentation. The index now maps every study to its full-size example.

Every code block runs in CI like the rest of the documentation, so the numbers are
produced by the code beside them. Most of the studies end with the quantum model
losing, which is the point: a library that only publishes its wins teaches nothing
about when to trust it.

### Fixed — `backprop` computed on a simulator when asked for a device

`method="backprop"` accepted a `backend` argument and never used it. Pointed at a
backend with no statevector, it returned a machine-precision gradient computed on the
torch simulator — to a caller who had asked for one from hardware. That is precisely
the plausible-wrong-number failure this library exists to refuse, and it was in the
library.

It now refuses, the same way `adjoint` does, and names `parameter-shift` as the
measurement-only route. On a statevector backend it is unchanged. The docstring also
now says plainly that backprop always evaluates on the torch simulator whichever
statevector backend is selected — the number agrees, but the device is not honoured.

Found by testing the claim that a backend implementing one method inherits the whole
library. It does: a device supplying only `counts` gets sampled expectations,
qubit-wise-commuting grouping, batched expectations, batched parameter-shift gradients
and quantum kernels, all within shot noise of exact. Three of the four methods that
cannot work already refused. This was the fourth.

### Added — `qk.search`, a grid search that skips what cannot work

    best = qk.search(X, y, ansatz=["hardware_efficient", "strongly_entangling"],
                     n_layers=[2, 3], lr=[0.05, 0.15], class_weight=["balanced"])

Every axis takes a list; anything left out keeps its default. Ansätze and feature maps
are named through the registries, so a custom one registered with `register_ansatz` or
the new `register_feature_map` joins the grid without any special handling.

**Before fitting anything it runs `diagnose` on each assembled model** and skips the
ones already condemned, reporting the reason instead of ranking them last. On the moons
grid that removes a `basic_entangler` configuration whose `d<Z0>/dtheta_0` is exactly
zero — a full fit, and a row in the results table that would otherwise have looked
merely unlucky.

Pruning is by finding **code**, not severity, because severity is the wrong axis here:
`DEAD_WEIGHTS` and `UNMEASURABLE_WEIGHTS` are warnings that mean *wasteful* and
`FLAT_GRADIENTS` is a warning that means *cannot learn*. `prune="error"` (default) skips
only what cannot work at all, `"untrainable"` adds flat gradients, `"warning"` is
aggressive enough to empty a grid — `hardware_efficient` carries an
`UNMEASURABLE_WEIGHTS` finding by construction — and an explicit list of codes works
too. Every configuration is diagnosed either way and its findings print beside its
score, because a point that scores well *and* carries `DEAD_WEIGHTS` is worth seeing as
exactly that.

The rest follows the library's existing habits: identical folds for every
configuration, the imbalance-aware metric for the task, and a verdict that refuses to
call a winner when the lead is inside the fold spread. `dry_run=True` builds and prunes
the grid without fitting anything. `n_qubits` defaults to one per feature rather than a
fixed number, since a feature pipeline can drop columns but cannot invent them, and a
typo'd axis name is an error with a suggestion rather than a sweep that silently varies
nothing.

### Fixed — the flat-gradient probe read a different circuit from the dead-weight probe

Making `diagnose` prefix-aware left `gradient_variance` probing the *bare* ansatz while
the dead-weight check used the composed one. It then picked weight 0 — live with an
encoding in front, dead from `|0>` — and reported a barren plateau that was really a
dead parameter. `strongly_entangling` inside a `VQC` was wrongly flagged
`FLAT_GRADIENTS`; it now reports only the `UNMEASURABLE_WEIGHTS` it actually has. Both
probes now read the same circuit.

### Fixed — `diagnose` now reads the circuit the caller actually runs

`diagnose(model)` analysed the model's *ansatz standing alone*, starting from `|0>`.
Which weights are dead depends on the state the ansatz is handed, so for a model with a
feature map in front it named the wrong ones: `strongly_entangling(4, 2)` was reported
as having weights `[0, 3, 6, 9]` dead — its leading `Rz` on each wire, which is a global
phase on `|0>` and alive the moment anything encodes data before it. The report was
correct about a circuit the caller was not running.

The probe now composes the model's own feature map in front and redraws its angles at
every probe, so a weight is called dead only if no input makes it matter. A bare
`Ansatz` is unchanged and now says `(from |0>)` so the setting is explicit.

### Added — `UNMEASURABLE_WEIGHTS`

Chasing the above turned up a second, sharper failure that nothing was checking: a
weight can change the state and still be unable to change the *number the model reads
out*. `diagnose` now reports those when the model's observables are known.

It fires on `hardware_efficient`, this library's own default ansatz. Every layer ends
in `rz`, which commutes with the `cx` entanglers and with any Z-basis observable — so
**the final `rz` layer cannot move a `Z` expectation at all**. On the default 4-qubit,
2-layer `VQC` that is 4 of 16 weights: a quarter of the parameters carried by the
optimiser every step, invisible to the readout, and invisible in a loss curve.

Verified independently before shipping the check — two gradient methods, four
observables, five random initialisations, all agreeing to `1e-17` — and it is a
property of the *readout*, not a defect in the ansatz: measuring `X` instead makes
every one of those weights live, which is what the new test asserts.

`hardware_efficient` is deliberately left as it is. Changing the rotation order would
alter every existing model's behaviour and break the PennyLane template mapping; the
honest fix is that the tool now tells you, and the fix text names both remedies.

### Added — batched gradients, and the backward pass that was 99% of training

Batched execution reached the forward pass only. Measured on a 4-qubit `VQC` at
batch 128, the forward pass was 13 ms of an 1170 ms training step — so the speedup
had landed on 1% of the work.

- **`qk.grad_batch(spec, thetas, obs)`** — `(batch, n_params)` gradients, dispatching
  to adjoint on an exact simulator and parameter-shift otherwise.
- **`qk.param_shift_grad_batch`** is **backend-agnostic by construction.** A shift rule
  only ever needs the circuit *run* at shifted angles, so a whole batch's gradient is
  one set of evaluations with no state inspection. It goes through
  `Backend.expectation_over_slots`, which every backend has — asserted equal across
  NumPy, Qiskit, Cirq and the torch backend, and shown working on a sampling-only mock
  device. On hardware this is the batched submission a provider wants: one job rather
  than `batch × 2P` blocking calls, which was gap #1 in `examples/toward_hardware.py`.
- **`qk.adjoint_grad_batch`** — the same sweep with the batch as a leading axis.
- `QuantumLayer` uses them, so training benefits with no API change. A gradient method
  with no batched form (SPSA, Hadamard-test) falls back to the loop rather than failing.

**Full training step, batch 128:** 19.8× at 4 qubits, 19.5× at 4 qubits × 4 layers,
8.7× at 6 qubits, 3.3× at 8. Gradients agree with the per-sample path to float32
round-trip.

Batch primitives moved into **slot** space, because that is what differentiation needs:
a shift rule moves one *occurrence* of a parameter and a weight-tied parameter has
several. `CircuitSpec.bind_slots_batch` resolves logical parameters to slot angles for a
whole batch in three NumPy operations, and `Backend.max_batch_rows` chunks the fan-out —
the same answer to a simulator's memory limit and a provider's job limit.

### Changed — the kernel Gram matrix is one batched evaluation

Every entry of a compute-uncompute Gram matrix is the *same* circuit structure at
different angles, and the per-pair loop was throwing that away. `QuantumKernel.__call__`
now assembles the whole matrix as one batched call, still evaluating only the strict
upper triangle and still using the cache — batching changed how misses are evaluated,
not whether hits are reused.

A 20×20 Gram matrix went from 45 ms to 3.2 ms. Against PennyLane that row moved from
6.6× to **69×**, which is where the honest overall median moved from 1.6× to 1.7×.
Sampled kernels, the swap-test estimator and backends without a statevector fall back
to the pair-at-a-time path.

### Added — documentation

- **An API stability policy** (`docs/about/stability.md`) — a written promise about what will not
  change without a deprecation period, and an honest list of what is *not* promised.
  Research code outlives the version it was written against.
- **`examples/accelerate_pennylane.py`** — the three inner loops worth borrowing
  without migrating anything: kernel Gram 127×, batched gradients 44×, exact metric
  tensor 53×, each checked against PennyLane's own answer before the timing is quoted.

### Added — batched execution

`expectation_batch` took a list of circuits and looped. Nothing exploited the fact
that a training batch shares one circuit *structure* and differs only in the encoded
angles, which is the shape every hybrid model actually has.

- **`qk.expectation_over(spec, thetas, obs)`** — one circuit, a `(batch, n_params)`
  array. Knowing the structure is shared is what lets a backend beat a loop.
- **`Backend.statevector_batch` / `Backend.expectation_over`** — new protocol methods
  with working defaults derived from `statevector`, so every backend gains a correct
  batch path immediately and can override it if it can do better. This is also the
  shape a *device* needs: real providers take a list of circuits and return a job, and
  one blocking call per circuit against a queue was gap #1 in
  `examples/toward_hardware.py`.
- **A vectorised NumPy path** that carries the batch as a leading axis and applies each
  gate to the whole stack in one `einsum`, with closed-form batched matrices for the
  rotations that dominate every ansatz. Measured against the one-at-a-time loop on the
  VQC forward pass at batch 128: **24× at 4 qubits, 21× at 4 qubits × 4 layers, 9.4× at
  6 qubits, 3.6× at 8**.
- `QuantumLayer.forward_batch` now uses it, so `VQC`/`VQRegressor` training gets the
  speedup without any API change.

**It is dispatched, not assumed.** Batching trades per-sample Python overhead for worse
memory locality, so it wins while the overhead dominates and loses once `2**n` does. The
measured crossover sits between 10 and 11 qubits and does not move with batch size —
what you would expect if `2**n` alone sets it. Above `NumpyBackend.batch_max_qubits`
(default 10) the loop runs instead. An earlier `moveaxis`-and-reshape implementation was
*slower* than the loop from 10 qubits up, because it copied the whole stack twice per
gate; contracting in place with explicit `einsum` subscripts fixed that and is uniformly
faster.

Every batched result is asserted equal to the loop it replaces, and each entry in the
vectorised gate table is asserted equal to the scalar `gate_matrix` it stands in for —
a vectorised `ry` with a sign error would return numbers in the right range.

### Added — circuits can be read in

`to_qiskit`/`to_cirq`/`to_spinqit` have always existed; the reverse did not, and
one-way interop is the difference between a library someone tries and one someone
adopts. An existing project has circuits already.

- **`qk.from_qasm`** — OpenQASM 2.0, parsed with the **standard library alone**, so it
  works in a bare `pip install qmlkit`. Qiskit, Cirq, Braket, t|ket> and Q# all export
  QASM 2.0, so one parser reaches all of them without qmlkit depending on any. Angle
  expressions (`pi/2`, `-2*pi/3`) are evaluated through a restricted AST walk rather
  than `eval`, which on file contents would be a code-execution hole.
- **`qk.from_qiskit`** — a `QuantumCircuit`, including unbound `Parameter`s, which QASM
  cannot represent; they become `ParamRef`s in Qiskit's own parameter order.
- **`qk.from_pennylane`** — a tape, QNode or quantum function. Templates such as
  `BasicEntanglerLayers` are recursively decomposed into the gates they stand for.
- **`register_importer`** — the same registry pattern as the rest of the library.

**Qubit order is what these tests check.** An importer that maps the register the wrong
way round produces a circuit that runs and returns plausible numbers. So
`from_qiskit(to_qiskit(spec))` is asserted to reproduce the *statevector* to `1e-12`
over randomly generated circuits, and the PennyLane importer is checked against
PennyLane's own simulator rather than against the assumption that both are big-endian.
The test circuits are deliberately asymmetric, because a symmetric one cannot tell a
correct mapping from a reversed one.

Gates with no qmlkit definition raise `UnsupportedGate` naming them, as do `measure`
and `reset`. The `u`/`u3` family is the one exception: decomposed to `rz·ry·rz` with a
warning that an overall phase was dropped, which is unobservable alone and a relative
phase inside a controlled block.

### Changed — the benchmark now runs against PennyLane's fastest configuration

The speed claims were measured against `default.qubit` and the Hadamard-test
`qml.metric_tensor`, and reported a median 6.1×. `pennylane-lightning` ships with every
PennyLane install and `qml.adjoint_metric_tensor` sits next to `qml.metric_tensor`, so
that was not a fair comparison. Re-measured against both, summarised on the faster:
the honest median is **1.6×**, the 8-qubit gradient is a tie, the kernel Gram matrix
holds at 6.6×, and the metric tensor holds at 49–105× while agreeing to `1.7e-16`.
README and the validation page carry both columns.

### Added — the evaluation layer

The half of a quantum machine learning result that is not the circuit: whether the
score means what it appears to, what the classical bar is, and whether the number
can be trusted or reproduced.

- **`qmlkit.evaluate`** — every metric a task needs in one call, for
  `classification`, `regression`, `clustering` and `generative`. The returned
  `Scores` object indexes like a mapping, names the metric to quote as `primary`,
  and carries `notes` that fire when a metric misleads — accuracy sitting at the
  majority-class rate, a class too small for its per-class scores to mean anything,
  ROC AUC flattering a rare positive class. Pure NumPy; cross-validated against
  scikit-learn on randomly generated inputs in `tests/test_evaluate.py`.
- **`qmlkit.imbalance`** — `class_weights` (interchangeable with scikit-learn's
  `class_weight="balanced"`), `pos_weight`, `sample_weights`, `resample`,
  `stratified_split` and `stratified_folds`, plus `imbalance_report`, which names
  what the skew will break and the call that fixes each one. Stratified splitting
  guarantees a rare class cannot vanish from a fold, which is what makes a
  small-data test score mean anything.
- **`qmlkit.nn.losses`** — `weighted_cross_entropy`, `class_weight_tensor` and
  `FocalLoss`. `VQC` now takes `class_weight=` and `focal_gamma=`, computed from
  the `y` passed to `fit` rather than assumed at construction.
- **`qk.baseline`** — every classical baseline on the same folds, the same metric
  and the same preprocessing as the model under test, with a verdict that refuses
  to call a lead a result when it is smaller than the fold-to-fold spread. The
  NumPy-only baselines (majority, nearest-centroid, RBF kernel ridge, linear least
  squares) always run; scikit-learn's are listed as *skipped* when it is absent
  rather than dropped. Extensible through `register_baseline`.
- **`qk.plan`** — the circuit and shot budget for a training run before it starts,
  with wall-clock at a given queue latency and the cheaper gradient methods listed
  alongside what each one gives up. Qubit-wise-commuting grouping is counted, not
  assumed.
- **`qk.selfcheck`** — the parity idea from `tests/test_pennylane_parity.py` turned
  outward: computes the gradient by every exact route available and compares them,
  then runs the circuit through every installed backend. Catches a custom gate with
  wrong declared `frequencies`, which returns a plausible number rather than
  raising.
- **`qk.fingerprint`** — the versions, backend and seed that decided a number,
  JSON-serialisable and cheap enough to attach to every run.
- **Documentation** — a new guide, *Evaluating a model honestly*, and a reference
  page. Every snippet in the guide executes in `tests/test_docs.py` like the rest.

### Changed

- `HybridModel._loss_fn` now takes the training targets, so a subclass can build a
  loss that depends on the label distribution. `VQC` and `VQRegressor` are updated;
  any external subclass overriding `_loss_fn` needs the same one-argument signature.

### Added — Phase 0: foundation

- **Backend-neutral circuit IR** (`CircuitSpec`, `Op`, `ParamRef`, `Slot`). A circuit
  is data; backends compile it, and gradients, resource counting and drawing all read
  the same representation.
- **Slot abstraction** — one angle site per (operation, parameter position). A logical
  parameter may fill several slots (weight tying), which is what makes correct
  per-occurrence gradients expressible.
- **Exact NumPy statevector backend**, double-precision throughout, plus the `Backend`
  protocol and a backend registry with a lazy, clearly-diagnosed SpinQit entry.
- **Pauli observables** — `PauliString`, `PauliSum`, `Z`/`X`/`Y`/`ZZ` shorthands, exact
  and sampled expectations, automatic basis rotation, and qubit-wise-commuting grouping.
- **Execution API** — `statevector`, `run_counts`, `probabilities`, `expectation`,
  `expectation_batch`. `shots=None` is the default and returns exact values.
- **Parameter-shift gradients**, with shift rules *derived* from each gate's declared
  generator frequencies rather than transcribed:
  - per-gate rule lookup, so a circuit mixing `ry` (two-term) with `crz` (four-term)
    is differentiated correctly;
  - per-occurrence shifting, so weight-tied parameters sum over their occurrences;
  - `grad_circuit_cost`, which sums real per-slot rule costs instead of assuming `2P`;
  - gradients with respect to encoded inputs, via `angle_encode(..., trainable=True)`.
- **Fluent `QCircuit` builder** with rotation layers, named entanglement patterns
  (chain, ring, full, alternating) and parametric entanglers.
- **Shot arithmetic** — standard error, variance, `shots_for_precision`, runtime estimates.

### Notes

- Simulator-only for the whole `0.x` line. The NumPy backend is the default and the
  reference implementation.
- Requires Python 3.10+. Python 3.9 reached end-of-life in October 2025, and 3.10 is
  also SpinQit's highest supported version, so it is the overlap that matters.

### Added — Phase 1: multi-backend support

- **SpinQit backend**, verified against a live SpinQit 0.2.x install on Python 3.10.
  Supports both exact statevectors (`result.states`) and sampled counts, plus
  `verify_conventions()` — a one-call self-check that bit order and gate definitions
  still match what the backend assumes.
- **Qiskit backend** and **Cirq backend**, both agreeing with the NumPy reference to
  machine precision. Each exposes its native circuit (`to_qiskit`, `to_cirq`,
  `to_spinqit`) so it can be drawn, transpiled, or handed to that SDK's own tooling.
- **Cross-backend equivalence suite** — one circuit zoo run through every installed
  backend, asserting agreement on statevectors, probabilities, expectations over
  X/Y/Z and two-body terms, seeded sampling, and parameter-shift gradients.
- **Backend registry with availability detection** — `available_backends()`,
  `is_available()`, `backend_report()`. Every SDK import is lazy, so `import qmlkit`
  requires none of them, and a missing SDK produces an install command rather than an
  `ImportError`. `QMLKIT_BACKEND` sets the default from the environment.
- **Shared seeded sampling** (`_sampling.py`), so a seed reproduces identical counts on
  any simulator backend. Qiskit's `sample_counts` takes no seed and Cirq carries its
  own RNG, which would otherwise make results irreproducible and incomparable.
- `sdg` / `tdg` added to the circuit builder.

### Changed

- `Backend` now supplies measurement *semantics* (sampling, basis rotation,
  expectation) while subclasses supply only a statevector. Previously each backend
  would have re-implemented these; one definition is what makes cross-backend
  equivalence meaningful rather than coincidental.

### Fixed — upstream discrepancies worked around

- **SpinQit's `CY` applies `-iY`** to the control-1 subspace instead of `Y`. That is a
  relative phase between control branches, so it is physically observable — a control
  qubit in superposition gives different measurement statistics. The backend emits
  `Sd·CX·S` instead, which reproduces the standard gate exactly. SpinQit's
  single-qubit `Y` is correct; only the controlled form is affected.
- SpinQit's simulator carries a precision floor near `1e-10`, not machine precision;
  cross-backend comparisons use a per-backend tolerance so this is not mistaken for a
  translation error.

### Added — Phase 2: the encoding layer

- **`PauliFeatureMap`**, with `ZFeatureMap` and `ZZFeatureMap` on top of it — and the
  two pieces such a map needs made public, `basis_change` and `default_data_map`, so
  either can be swapped without rewriting the map. Tested against the analytic kernels
  each map induces rather than against itself.
- **`amplitude_encode`**, built from uniformly-controlled rotations rather than a
  backend state-preparation primitive. Emits only `ry`/`rz`/`cx`, so it runs identically
  on every backend and its exponential cost is visible. Handles real, signed and complex
  amplitudes, exact to machine precision; `check=True` re-simulates and asserts the
  result. Exposes `uniformly_controlled_rotation` and `state_preparation_angles`.
- **`hamiltonian_encode`** — Trotterised Ising evolution. Every term is diagonal so the
  terms commute and the split is *exact at any step count*: `steps` buys depth and
  nothing else, which is pinned by a test.
- **`DataReuploadEncoder`** — interleaved `S(x)` / `W(theta)` blocks, with
  `trainable_input=True` to expose `df/dx`. A test confirms by FFT that more uploads
  reach genuinely higher frequencies.
- **`AngleScaler`** and **`PCAReducer`** (plain SVD, no sklearn) plus `reduce_to_qubits`,
  for the two problems every model hits before any quantum step: wrong feature scale, and
  more features than qubits.
- `AngleFeatureMap`, `FeatureMap` base with a free `adjoint()`, and `pauli_terms`.

### Added — Phases 3 & 4: ansätze, gradient methods, and the PyTorch bridge

**A composable ansatz vocabulary.** `RotationLayer`, `EntanglerLayer`,
`ParametricEntangler`, `PoolLayer`, `Custom`, composed with `+`, `repeat` and `share`.
Every built-in is a short expression in it, so a new ansatz is one line and inherits
correct gradients, resource counting and a torch layer without opting into any of them.
Parameter counts are **inferred** from a dry build — a miscount is not a failure mode.

- Zoo: `hardware_efficient`, `strongly_entangling`, `simplified_two_design`,
  `tree_tensor_network`, `mps_ansatz`, `qcnn_ansatz`, `qaoa_ansatz`, plus `conv_block`.
- `share()` and `conv_block(tied=True)` implement genuine weight tying. An 8-qubit QCNN
  has 6 tied parameters against 22 free ones at identical gradient cost — the real
  convolutional tradeoff, and the case the per-occurrence gradient sum exists for.
- `register_ansatz` / `get_ansatz` / `list_ansatze`.

**Adjoint differentiation** — exact gradients in one backward pass, independent of the
parameter count. Agrees with parameter-shift to machine precision (1e-16) on every
ansatz in the zoo, including weight-tied and four-term-rule circuits. Measured 11×
faster at `P=20` and 64× at `P=120`. Requires closed-form gate derivatives, which are
now declared on every parameterised gate and verified against finite differences.

**SPSA** — two evaluations per gradient whatever `P` is, with Spall's decay schedules
including the stability constant `A` that most write-ups omit. `minimize_spsa` for the
optimisation loop.

**One `grad()` with a method registry.** `method="auto"` picks adjoint when every gate
has a derivative and the backend can produce a statevector, parameter-shift otherwise;
shots force parameter-shift. `register_gradient` makes a custom estimator a keyword
everywhere the library takes `method=`.

**The PyTorch bridge.**

- `QuantumFunction` / `QuantumLayer` — a circuit as an `nn.Module`. **Inputs receive
  gradients**, so a classical layer placed before the quantum one actually trains; this
  closes the frozen-pre-net defect, asserted by test. `df/dx`
  through a *nonlinear* feature map works by differentiating the circuit with respect
  to its encoding angles and finishing the chain rule classically — no circuits spent
  on the classical half.
- `VQC`, `VQRegressor`, `HybridModel` — the two-line path, with `fit`/`predict`/`score`.
  Every default is one keyword away from being something else.
- `torch.autograd.gradcheck` passes for inputs and weights, on linear and nonlinear maps.

**Feature maps** gained `angles`, `n_angles`, `build_parametric` and `angle_jacobian`
(closed-form for the standard data map), which is what makes input gradients possible.

### Changed

- `expval()` added as a float-only convenience alongside `expectation()`.
- PyTorch is an optional extra; `qmlkit.nn` is imported lazily so `import qmlkit` never
  requires it.

### Added — Phase 5 and parity with Qiskit ML / PennyLane

**Quantum kernels (Phase 5).**

- Three estimators: `fidelity_kernel` (compute-uncompute, the default — fewest qubits,
  no ancilla), `swap_test_kernel` (two registers plus an ancilla), and `hadamard_test`,
  the only one that keeps the **sign** of the inner product. All three match the exact
  overlap to machine precision.
- `QuantumKernel` with a **symmetric cache**, so `k(a,b)` and `k(b,a)` share one entry;
  a training Gram matrix costs exactly `m(m-1)/2` circuits.
- PSD repair — `threshold_matrix`, `displace_matrix`, `flip_matrix`, `closest_psd_matrix`.
  Shot noise really does push an estimated Gram matrix out of the cone; a test asserts it
  and that each method brings it back.
- `target_alignment` (kernel-target alignment), `center_kernel`, `normalize_kernel`.
- `QSVC` / `QSVR` over scikit-learn's precomputed-kernel solver, `NearestFidelityClassifier`
  (no solver at all), and `TrainableKernel`, which optimises the *embedding* by maximising
  alignment before any classifier is fitted.
- `projected_kernel_matrix` — compares one-qubit reduced density matrices instead of the
  global fidelity, so it stays informative where the fidelity kernel has concentrated.
  Measured at 8 qubits: fidelity spread 0.005 against projected 0.037.
- `concentration_report`, `geometric_difference`, `kernel_shot_cost`.

**Filling the gaps against Qiskit Machine Learning and PennyLane.**

- `qmlkit.metrics` — `expressibility` (KL from the Haar fidelity distribution),
  `meyer_wallach`, `entangling_capability`, `gradient_variance`, `barren_plateau_scan`,
  `effective_dimension` (the normalised-Fisher construction, not the eigenvalue-count
  shortcut), `generalization_bound`, and `AnsatzReport` / `compare_ansatze`.
- `qmlkit.fourier` — `fourier_coefficients`, `spectrum`, `model_spectrum`. Turns the
  central claim of the re-uploading literature into a measurement.
- `qmlkit.info` — `reduced_dm`, `purity`, `vn_entropy`, `mutual_info`, `state_fidelity`,
  `concurrence`, `bloch_vector`.
- `qmlkit.optim` — **Rotosolve** (closed-form per-coordinate minimum, no learning rate),
  the **Fubini-Study metric tensor**, `quantum_fisher_information` (= 4x the metric), and
  **quantum natural gradient**.
- Three more ansatz templates: `basic_entangler`, `two_local`, `random_layers` — ten total.
- `qmlkit.datasets` — `ad_hoc_data` (separable by a quantum kernel by construction),
  `bars_and_stripes`, `make_moons`, `make_circles`, `make_blobs`, `make_parity`,
  `train_test_split`. No downloads, no sklearn.
- `qmlkit.draw` / `qmlkit.specs` — text circuit diagrams and a full cost summary.

### Fixed

- **A commuting trainable block silently collapses data re-uploading.** `Ry(x)Ry(t1)Ry(x)Ry(t2)`
  equals `Ry(2x + t1 + t2)`, so the model reaches a single frequency and every weight
  becomes a phase shift — measured: one frequency at amplitude 1.0, against the full
  `0..L` spectrum with a non-commuting block. `DataReuploadEncoder` now warns.
- `np.trapezoid` in a test made the suite NumPy-2-only; SpinQit pins `numpy<2`, so the
  suite has to run on both. Replaced, and the whole codebase scanned for other NumPy-2-only
  APIs (there were none).

### Changed — re-uploading generalized

**Data re-uploading is a pattern, not a structure**, and the library now treats it as
one. `EncodingLayer` makes a feature map a composable block, so re-uploading is any
interleaving of any encoding with any trainable block:

- `reupload(fmap, n_layers, block=..., order="SW"|"WS", share_weights=...)` builds the
  common shapes; anything else composes directly from the block vocabulary — including
  **two different feature maps in one model**, which the previous fixed class could not
  express at all.
- `BuildContext` gained an input namespace, so encoding angles and trainable weights
  occupy separate ranges of one flat vector. That is what keeps `df/dx` and `df/dtheta`
  separable while a re-uploading model drops straight into `QuantumLayer`.
- `Ansatz.bind(x, weights)` binds data and weights separately; `build(theta)` still
  takes the full vector and now says so when the sizes disagree.
- `DataReuploadEncoder` remains as the plain angle-encoding shortcut, documented as one
  convenient choice rather than the definition.

### Added — Phase 6

- `QCNNLayer`, `MPSLayer`, `QLSTMCell`, `QLSTM`, `DressedQuantumNet` — structured
  architectures as ordinary `nn.Module`s.
- `qmlkit.generative` — `QCBM` (trained by MMD, because a Born machine is implicit and
  admits no likelihood), `QGAN`, `QuantumBoltzmannMachine`, `QuantumHopfield`, plus
  `mmd_squared`, `kl_divergence`, `total_variation`, `boltzmann`, `ising_energy`.

### Fixed

- `test_reupload_widens_the_reachable_spectrum` used a trainable block that commutes
  with its encoding, so it measured the *collapsed* single-frequency case and passed
  for the wrong reason. Now uses a non-commuting block, which is what the claim needs.

### Fixed — a flaky test, and its cause

`test_qcbm_training_reduces_mmd` failed roughly one run in five. The cause was real,
not a tolerance problem: `QCBM.score()` samples without a seed, so it draws on the
backend's shared RNG, whose state depends on whatever ran before. Comparing two noisy
estimates before and after training can go the wrong way by chance even when training
worked.

- `QCBM.score()` gained a `seed` argument for reproducibility.
- `QCBM.exact_distance()` added — measures the distance to the target on the model's
  **exact** distribution, no sampling at all, which is what a before/after comparison
  needs on a simulator.
- The test now asserts on the exact distance. Fifteen consecutive full runs across both
  environments are clean.

### Added — two more gradient algorithms, and cross-validation against PennyLane

**`method="hadamard"`** — the Hadamard-test gradient. For a Pauli-generated rotation,
`d_k E = -Im<phi|O|psi>`, and that imaginary part is exactly what a Hadamard test reads
out: put an ancilla in `|+>`, insert a *controlled* generator right after gate `k`, and
measure `<Y_a (x) O>`. One circuit per parameter instead of parameter-shift's two, and
unlike adjoint it is a real measurement, so it is valid on hardware. It refuses
controlled rotations with an explanation rather than guessing — `CRZ`'s generator is
not a Pauli, so there is no controlled form to insert.

**`method="backprop"`** — a differentiable statevector simulator written in torch
(`TorchBackend`, `torch_expectation`), so autograd differentiates the circuit directly.
This is the least physical method in the library and is documented as such: it reads
intermediate states no device will expose, and its memory grows with depth. It exists
because a circuit inside an autograd graph is genuinely useful; for a standalone
gradient, adjoint is both faster and lighter.

- `hessian()` — second derivatives by differencing the *exact* gradient, so only the
  outer derivative is approximated. Returned symmetrised.
- `gradient_cost(spec, method)` — circuits one gradient costs under any method, so the
  tradeoff is queryable instead of folklore.

All four exact methods (adjoint, backprop, hadamard, parameter-shift) agree to machine
precision across the ansatz zoo, including weight-tied circuits and parameter scaling.
Measured on 5 qubits with a two-term observable, `P=120`: adjoint 12.6 ms, backprop
50 ms, hadamard 404 ms, parameter-shift 823 ms.

**Cross-validated against PennyLane.** `examples/compare_pennylane.py` checks 19
quantities — expectations, every gradient method in both libraries, feature-map
kernels, reduced density matrices and entropies, and the Fourier spectrum of a
re-uploading model — and agrees to `1e-16` on all of them. `examples/quickstart.py`
walks every layer of the library end to end.

One genuine convention difference surfaced: PennyLane's `IQPEmbedding` emits `RZ(x_i)`
and `MultiRZ(x_i x_j)` where qmlkit's `PauliFeatureMap` follows the Qiskit convention
and emits `Rz(2 phi)`. Neither is wrong; they match once the data map absorbs the
factor of two. It is recorded because a kernel differing by exactly this factor would
be very hard to spot.

### Fixed

- `hadamard_grad` called `expval` once **per observable term**, re-preparing the state
  each time — so a `k`-term observable cost `k` circuits per parameter, not one, and
  the method's whole circuit-count advantage over parameter-shift disappeared into it.
  It now passes the lifted observable as a single `PauliSum`, which the backend
  accumulates from one state preparation. Wall-clock at `P=120` went from 749 ms to
  404 ms — against parameter-shift's 823 ms, which is the 2x the circuit count
  predicted all along. Found by benchmarking rather than by a test, because every test
  still passed: the answer was always correct, only the cost was wrong.
- `hessian()` annotated `Sequence[float]` without importing `Sequence`. Invisible at
  runtime under `from __future__ import annotations`, caught by mypy.
- `torch_expectation` reused the loop variable `p` for both gate parameters and Pauli
  letters, which mypy flagged as a genuine type collision.

### Added — a real cross-validation suite against PennyLane

`tests/test_pennylane_parity.py` — **301 parity cases**, run as part of the normal
suite rather than as a one-off script, so they guard every future change. Coverage:
all 20 gate matrices at 6 angles; every closed-form `dU/dtheta` against a differenced
PennyLane matrix; 40 **randomly generated** circuits over the full gate set on 1-5
qubits (statevectors, probabilities, random multi-term observables); gradients across
5 ansatze x 4 observables, all four exact methods, PennyLane's own four methods back
against ours, and randomised circuits; angle/amplitude/basis/IQP encodings;
`BasicEntanglerLayers` and `StronglyEntanglingLayers`; kernel Gram matrices; reduced
density matrices, entropies, purity, mutual information and fidelity over random
states; re-uploading Fourier spectra; the Fubini-Study metric and QFIM; and Rotosolve
and QNG trajectories compared step by step rather than only at their endpoints.

`examples/benchmark_pennylane.py` — wall-clock on identical work. qmlkit is faster on
14/14 cases, median 6.1x, with the caveats stated in the file: sections 1-4 are
largely dispatch overhead and narrow as `2^n` grows, while the metric-tensor result is
an algorithmic difference that widens.

**Four genuine convention differences surfaced, each now pinned by its own test.**
None is a bug in either library, and all four are the kind that produce a plausible
wrong number rather than an exception:

- PennyLane's `IQPEmbedding` emits `RZ(x_i)` / `MultiRZ(x_i x_j)`; qmlkit follows the
  Qiskit convention and emits `Rz(2 phi)`.
- `amplitude_encode` drops one overall phase, since it is built from uniformly
  controlled rotations rather than a state-prep primitive. Unobservable in isolation,
  observable inside a controlled block — which the docstring already warned about.
- A `"ring"` on two qubits collapses to a single `CX` here; PennyLane's templates emit
  both `CNOT(0,1)` and `CNOT(1,0)`. `entangler_pairs` now documents this.
- `approx="block-diag"` means something narrower in PennyLane: it blocks the metric by
  circuit layer and zeroes cross-layer entries. The practical consequence is measured
  in `test_qng_beats_pennylanes_default_because_its_metric_is_exact` — on a 3-qubit,
  2-layer problem at equal steps and step size, qmlkit's QNG reaches `-2.9999999`
  where PennyLane's default stalls at `-2.22`. Pointed at the exact metric, PennyLane
  traces qmlkit's trajectory to `1e-8`.

### Changed — the metric tensor is now exact

`metric_tensor` differenced the *circuit* to get each `|d psi / d theta_k>`, at
`eps=1e-4`. Every other exact quantity in the library is closed-form, and this one was
not: it agreed with PennyLane only to `2e-10`, and it was the input to QNG.

It now takes one forward sweep, carrying each open derivative state through the next
gate and opening a new one at each parameterised gate from that gate's declared
`dU/dtheta`. Same `(P, 2^n)` memory as before, **fewer** simulations than `2P`
differenced circuits, no step size, and agreement with PennyLane improves from `2e-10`
to `1.7e-16`. Finite differences remain as the fallback for a custom gate registered
without a `dmatrix`, which is the only case that still consults `eps`.

### Fixed

- `qmlkit` is measurably more accurate than PennyLane on one quantity, now recorded so
  a future tolerance change is deliberate: `state_fidelity` hits the analytic
  `|<a|b>|^2` to `1e-16`, whereas `qml.math.fidelity` takes matrix square roots of
  rank-1 density matrices and loses about eight digits.
- The parity fuzzer drew gate names from the live registry, which other test modules
  write throwaway gates into at run time — so it passed alone and failed in a full
  run. It now draws from a snapshot taken at import, and a separate test asserts the
  PennyLane mapping covers every built-in gate, so adding a gate cannot silently
  escape this file's coverage.

### Added — generic molecules, feature pipeline, scikit-learn interoperability

- `qmlkit.algorithms.molecule`: `from_integrals()` takes one- and two-electron
  integrals from anywhere — PySCF, OpenFermion, Psi4 — so *any* molecule becomes a
  qubit Hamiltonian, with `active_space` to fit it on the register you have.
  `molecular_hamiltonian(Molecule(...))` computes them here for s-orbital elements
  with a real restricted Hartree-Fock loop, covering H2, H3+, H4 chains and rings and
  HeH+ at arbitrary geometry. Validated against the H2-specific builder it
  generalises: the two agree to 7.4e-09.
- `FeaturePipeline` — standardise, reduce, scale to angles, fitted once on training
  data. `PCAReducer` already matched scikit-learn to 1e-16, but `reduce_to_qubits`
  refits on every call and so could not be applied to a held-out set.
- `SklearnCompatible` — `get_params`/`set_params` read off the constructor signature,
  and a lazily imported `__sklearn_tags__`. `QSVC` and `QSVR` now survive `clone`,
  which is what `Pipeline`, `cross_val_score` and `GridSearchCV` are built on, while
  scikit-learn stays an optional dependency.
- `qmlkit.algorithms.chemistry.h2_hamiltonian` and `chemistry_operator_pool`.
- `HANDOFF.md` — status, the four enforced conventions, known traps, ranked next steps.

### Fixed

- **The core library briefly depended on SciPy.** The Boys function imported
  `scipy.special.erf`, which is installed locally alongside torch and absent from a
  bare install, so every `core` CI job failed on a dependency nobody had declared —
  breaking the README's promise of NumPy and nothing else. `math.erf` has been in the
  standard library since 3.2 and both call sites were already scalar.
  `scripts/verify_install.py` now lists SciPy among the modules that must not leak in;
  it could not have caught this before, because its list was hand-maintained.
- The Pauli decomposition was `Tr(P @ H)` per term: `4^n` strings each a `2^n x 2^n`
  matrix product, `O(16^n)` — about a minute for a four-atom molecule. The same
  coefficients come out of `n` applications of one 4x4 change of basis, `O(n * 4^n)`,
  so H4 builds in 4.5 seconds instead of minutes.
- `.ravel()` gets a shape-typed result under newer NumPy stubs, so reassigning a
  general-shape array to it failed strict type-checking. Only CI's Python 3.10 NumPy
  tripped on this; two other NumPy generations tested locally did not.

### Added — the API as a coding agent meets it

Most code written against a library now is written by a model, which does not read
the documentation site before it types: it guesses a name, runs it, reads the
traceback, and tries again. Four changes take that loop seriously. All four help a
reader on their first afternoon in exactly the same way.

- **`qmlkit.utils.errors.unknown`** builds the message for an unrecognised name, and
  the twenty-one registry and mode lookups raise through it. Alongside the valid set,
  which several already listed, it names the nearest match — and treats case and
  separator drift as certain rather than fuzzy, since `parameter_shift` for
  `parameter-shift` is what half-remembering another library looks like, not a typo.
  `UnknownName` exists because `KeyError` alone renders as `repr(args[0])`, escaping
  every quote in a sentence built to be read; it is still a `KeyError`.
- **`qmlkit._aliases`** answers a name that is not misspelled but belongs to a
  different library: `qk.AngleEmbedding` reports that PennyLane calls it that and
  qmlkit calls it `AngleFeatureMap`. These stay errors rather than becoming aliases —
  an importable name is one somebody will depend on, and `qml.expval` takes a QNode
  where `qk.expectation` takes a spec, so a name that silently resolved would fail
  later and further from its cause. `tests/test_agent_api.py` asserts every target in
  the table still exists, so a rename cannot leave it pointing at nothing.
- **`qk.diagnose(model)`**, for the failures that return a number instead of raising:
  an encoding that commutes with its trainable block and collapses `L` uploads to one
  frequency, a parameter the circuit cannot feel, a circuit that only makes product
  states, a gradient that no realistic shot budget resolves, a kernel that has
  concentrated or stopped being positive semi-definite. Every ingredient already
  existed across `metrics`, `fourier`, `info` and `kernels`, reachable only by someone
  who already suspected the problem; this is the one call that does not need the
  suspicion. It takes an `Ansatz`, anything holding one, or a Gram matrix, and returns
  findings with a stable code, the number measured, and the edit that fixes it. Checks
  that can be exact are (a parameter is dead if shifting it cannot change the state);
  checks that are statistical report their number, and the gradient threshold is
  calibrated against measured variances from 2 to 10 qubits and 2 to 20 layers.
- **`docs/llms.txt` and `docs/llms-full.txt`**, generated by
  `scripts/generate_llms_txt.py` from the pages and from the package, committed, and
  verified in the docs workflow. One fetch gets every tutorial and guide plus the whole
  public API with signatures. Generated rather than written for the reason the docs are
  executable: a hand-maintained summary of an API is a second copy of the truth.

`docs/guides/agents.md` covers all of it with snippets that run under
`tests/test_docs.py`, and `AGENTS.md` is the short form of `HANDOFF.md` for an agent
working on the library rather than with it. `mypy --strict` now also covers `utils`,
`_aliases` and `diagnostics`.
