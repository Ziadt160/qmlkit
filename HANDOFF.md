# Handoff

Where the project stands, how to work on it, and what to do next. Written to be the
first thing read in a fresh session.

---

## Where everything lives

| | |
|---|---|
| **Repository** | <https://github.com/Ziadt160/qmlkit> — public, Apache-2.0 |
| **Documentation** | <https://ziadt160.github.io/qmlkit/> — deploys from `main` |
| **Source of truth** | The `qmlkit/` subdirectory of the upstream working repository; this repo is a subtree split of it |
| **PyPI** | <https://pypi.org/project/qmlkit/> - 0.1.0 is published; 0.1.1 is prepared here and not yet tagged |

### The one non-obvious thing about the workflow

The published repo is produced by a **subtree split** of the upstream working
repository's `qmlkit/` subdirectory. Commit
inside `qmlkit/` there, then:

```bash
git branch -D qmlkit-standalone
git subtree split --prefix=qmlkit -b qmlkit-standalone
git push qmlkit qmlkit-standalone:main
```

The split is deterministic, so re-pushing fast-forwards. `LIBRARY_PLAN.md` stays in the
upstream repository, because it is about that project rather than about qmlkit.

---

## Status

**1340 tests on Python 3.14, 909 on SpinQit's 3.10, 0 failures in either.** ruff clean
and `ruff format --check` clean; `mypy --strict` clean over **the whole package** - the
config listed six paths until 2026-08-28 and now lists `src/qmlkit`, so "mypy clean" and
"the package type-checks" finally mean the same thing. **94% coverage**, combined in
CI across every job — the number CI actually computes, not a local estimate. Two of the
twelve coverage files still fail to map (macOS and Windows record different absolute
roots), so the figure is carried by the `full` job, which installs every extra and runs
the whole suite on Linux.

Phases 0–6 are done, plus the algorithm and interoperability work. 0.1.0 is released;
0.1.1 is prepared and untagged.

Run the suite in **both** environments — SpinQit needs Python 3.10 and pins `numpy<2`:

```bash
pytest
```

```bash
C:/Users/pc/miniconda3/envs/spinq_env/python.exe -m pytest
```

---

## The four conventions

These are enforced by tests, not by discipline. Breaking one is the main way to make
the library unusable from outside.

### 1. An algorithm owns its loop, not its circuit

Every model takes `ansatz=` / `feature_map=` / `filter=` and must *actually use it*.
`tests/test_injection.py` injects two ansätze of different sizes and asserts the
model's parameter count follows — because a constructor that accepts `ansatz` and
silently ignores it looks identical from outside.

QCNN is the worked example: `qcnn_ansatz(8, filter="su4", pool="controlled")`. The
filter registry is shared with `mps_ansatz` and `tree_tensor_network`, since all three
slide the same two-qubit block.

### 2. Estimators must be scikit-learn clonable

Constructor arguments are stored under their own names; `SklearnCompatible` supplies
`get_params`/`set_params`, and `__sklearn_tags__` borrows scikit-learn's own tags
object lazily. This is what lets `QSVC` run inside `Pipeline`, `cross_val_score` and
`GridSearchCV` — while scikit-learn stays *optional*.

### 3. NumPy and nothing else

The core library depends only on NumPy. Everything else is an extra, imported lazily,
and a missing one produces an install command rather than a traceback.

`scripts/verify_install.py` checks this against a **built wheel in a clean venv** — an
editable install imports from `src/` and would keep working even if a module never made
it into the wheel.

### 4. Documentation is executable

`tests/test_docs.py` runs every Python block on every documentation page. The snippets
are tests, not illustrations, so a rename that breaks a tutorial breaks the build.
Every number shown was produced by running the code.

`README.md` is the exception and opts *in* instead: it is a reference whose blocks are
mostly deliberate fragments, so a self-contained one is marked `# docs: run` and
executed by `test_readme_blocks_marked_runnable_do_run`. The 0.1.1 encoding defect
shipped in a README example precisely because nothing ran it.

---

## Traps that have already cost time

**Three NumPy generations behave differently.** NumPy ≥2.3 gives `ndarray`
type-parameter defaults; earlier versions demand explicit ones under `--strict`; newer
stubs give `.ravel()` a shape-typed result. Always write `npt.NDArray[Any]`, never bare
`np.ndarray`, and **never set `python_version` in `[tool.mypy]`** — it makes mypy parse
*dependency* stubs at that version, and NumPy's use PEP 695 syntax. Cross-version
signal comes from CI running mypy on 3.10.

**Rotosolve is not always valid.** Its three-point fit assumes the loss is a *single*
sinusoid in each angle. That fails when one angle drives several gates that do not
compose (QAOA's cost angle drives one `rz` per edge — measured: five frequencies), and
when the loss is non-linear in the state (purity is `Tr(ρ²)`, so it carries double
frequencies). In both cases it converges instantly on the wrong point and reports it as
a result. `supports_rotosolve(spec)` checks the first case.

**A molecular Hamiltonian conserves particle number.** Any ADAPT operator that does not
has *exactly zero* gradient at Hartree–Fock, so the generic pool grows an empty
circuit. Use `chemistry_operator_pool`. This is physics, not a bug, and a test pins it.

**An input slot holds an *angle*, not a feature.** Each feature map derives its
angles its own way - a `ZZFeatureMap` Z term is `Rz(2 x_i)`, an `AngleFeatureMap`
rotation is `Ry(x_i)` - so angle `i` of one map is not angle `i` of another. Each map
therefore owns a disjoint range of slots, and the *same* map re-used keeps its own,
which is what makes re-uploading re-upload. Slots cannot be keyed by feature instead:
`ParamRef` is affine in one parameter, and a Pauli map's higher-order angle is
`2 * prod_j (pi - x_j)`. Before 0.1.1 two maps silently shared slots 0..n and the
second encoded the first's transformed angles - see the 0.1.1 changelog entry.

**Gate and gradient registries are process-wide.** Other test modules register throwaway
gates at run time, so anything iterating them must snapshot at import.

**SpinQit's `CY` is wrong** — it applies `−iY`, a physically observable relative phase.
Emitted as `Sd·CX·S`. Its simulator also carries a `1e-10` precision floor.

---

## What to do next

Revised 2026-09-13, after surveying what users actually complain about in PennyLane
(their forum, and 134 open `bug`-labelled issues on their tracker). Most of those
complaints this library already answers. The ranking below is what is *missing*, and
the ordering is: unblock, then build the substrate, then build on it.

**The standing warning still applies.** 190+ exports, one author, and the bottleneck
stopped being capability several releases ago. Items 2-6 are mostly *honesty layer*
rather than new capability surface, which is the only kind of growth this library can
afford. Where an item is pure capability (threads, GPU), keep it as small as it can be.

### 1. Release 0.1.1 - nothing below is reachable until this ships

0.1.0 is on PyPI and `git ls-remote --tags qmlkit` shows only `v0.1.0`. 0.1.1 is
prepared here and **not yet tagged**: it fixes a composition the README recommended
that built the wrong circuit without raising, and fills in the observable arithmetic
that identity-shifted cost functions need. The changelog entry has the whole account.

Everything on this side is ready, verified 2026-09-12: 1340 tests green on 3.14 and
909 on SpinQit's 3.10, `ruff check`, `ruff format --check` and `mypy` clean over the
whole package, `mkdocs build --strict` clean, the wheel builds and `twine check`
passes, and a clean venv installing only the wheel pulls in **numpy and nothing else**
before `verify_install.py` passes. Re-date the changelog if tagging later.

**The tag goes on the subtree-split commit, not on this repository.** `release.yml`
only exists in the published one, so `git push origin main --tags` tags the upstream
repository and triggers nothing. `RELEASING.md` has the procedure; it names `v0.1.1`.

**A PyPI version number can never be reused**, so the tag is deliberately not pushed
until the release is wanted.

This is also a positioning blocker, not only an engineering one. The published version
returns a silent wrong number, and a library whose pitch is *refuses a plausible wrong
number* cannot be promoted in that state.

### 2. The run record - one substrate, three features on top

`provenance.py` records a `Fingerprint`: everything that could change a number, at one
instant. What does not exist is the *trajectory* - what happened during the run. Note
that `VQC.fit` already keeps `self.history_` (per-epoch loss) and **nothing reads it**.

Build a structured, JSON-serialisable run record: config and fingerprint, then per
epoch the loss, gradient norm, parameter norm, and a prediction summary; plus the
resources actually used. Items 3 and 4 both consume it, and reproduction needs it
(there is no `serialize.py` either).

Three constraints, and they are what make this worth doing rather than a log dump:

- **Recording must not change the numbers.** Assert it: a run with recording on and
  one with it off must agree exactly, seeded.
- **Recording must be cheap.** Per-step logging that slows training would be absurd
  given that the loudest complaint about the competition is speed. Levels and
  sampling, with the default cheap.
- **The record is complete; the report is selective.** "Log everything" as a raw dump
  is the opposite of this library's style, which is to name the finding and the edit
  that fixes it. Keep the full record machine-readable, and render a report that
  answers questions rather than printing the record back.

No new dependencies - core stays NumPy-only.

### 3. Post-run diagnosis - the highest-volume complaint in the field

`diagnose()` takes an `Ansatz` or a Gram matrix. It answers *is this architecture
broken* before the run. Every "my model doesn't learn" thread is the other question:
*it trained, and the accuracy is 0.5.* That thread type is the most common on
PennyLane's forum and no QML library answers it.

Extend to `diagnose(model, X, y)` / `diagnose(run)`:

| Code | Catches |
|---|---|
| `QUANTUM_LAYER_BYPASSED` | the classical head alone scores the same - an ablation |
| `PREDICTIONS_CONSTANT` | one class for every input, so accuracy sits at the prior |
| `THRESHOLD_DEGENERATE` | loss falls but accuracy does not - scores all one side of the cut |
| `OUTPUT_RANGE_COLLAPSED` | the quantum layer's output range is too small to use |
| `WEIGHTS_DID_NOT_MOVE` | parameters unchanged since init - learning rate, or a detached graph |
| `LOSS_FLAT` | `history_` plateaued from the first epoch |

The first two need only a trained model and data, so they can ship before item 2. The
last two need the run record.

`QUANTUM_LAYER_BYPASSED` is the one to build first: "it works fine without the quantum
layer" recurs on that forum, `qk.baseline` does **not** answer it (that compares
against classical estimators on identical folds, which is a different question from
ablating the quantum layer inside the user's own hybrid model), and it is the most
uncomfortable finding the library could ship - which is the point.

**Test discipline, non-negotiable.** Construct a model that genuinely has each defect
and assert the finding is *true*, not that it *fired*. The old diagnostics tests
asserted firing, and that is exactly how three wrong findings survived. A false
positive in the honesty layer is worse than a false negative.

These run on trained models with real data, so they will be noisier than the
structural probes, which are exact. Budget for calibration, and cut a finding that
cannot be made reliable rather than shipping it at warning severity.

### 4. Resource planning, and recommending a backend

`budget.Plan` reports circuits and wall-clock at a given seconds-per-circuit. It never
reports **bytes**, and memory is the complaint that ends runs: 7 qubits with 30-40
parameters reported at 7-8 GB under backprop, kernels dying at 20 qubits.

Two parts:

- **Bytes in `plan()`** - statevector, batch fan-out, and the chosen gradient method.
  `adjoint` being the default is what avoids the blowup, and nothing currently *says*
  so; a user learns it only after already choosing this library.
- **`recommend()`** - given qubits, shots, gradient method, noise and available RAM,
  which *installed* backend fits, its projected peak memory, and what will fail and
  why. The refusals are worth as much as the picks.

This is the strongest of the newer ideas: it is honesty-layer work rather than feature
chasing, and nobody ships it.

### 5. Parallelism - the "12% CPU" complaint, answered correctly

Users report 10-20% CPU during training and conclude the library is single-threaded.
It is, but **threads are the wrong cure and adding them would not help**. At 6-12
qubits the statevector is kilobytes to a few megabytes, far below where NumPy or BLAS
threading pays for itself; the bottleneck is Python-level dispatch per circuit. 12% is
one core of eight, and one core is all the work there is.

Confirmed absent from `src/`: no `multiprocessing`, `joblib`, `concurrent.futures`,
thread pool, or `n_jobs` anywhere.

In priority order:

1. **Say it.** A dispatch-bound diagnostic - "6 qubits, 4000 circuits, one at a time;
   more threads will not help, batching will, here is the call." Cheapest item here
   and worth more than the knob, because the user's own conclusion is wrong.
2. **Process-level fan-out over independent work**, where real cores do help: kernel
   Gram blocks, CV folds, `search()` configurations, multi-seed runs, and
   parameter-shift shifts on backends that cannot batch. One `n_jobs` convention
   everywhere, default serial.
3. **BLAS thread control**, once 2 exists, so that processes times BLAS threads stop
   oversubscribing. Classic slowdown, and it will appear as soon as fan-out lands.

Remember `NumpyBackend.batch_max_qubits` (default 10): batching *loses* above the
10-11 qubit crossover, so any recommendation here has to respect it rather than
assume batching is always the answer. Do not raise it without re-measuring.

GPU stays out of scope. It is lost on headcount and costs nothing strategically.

### 6. The conformance contract - "runs on anything" without shipping hardware

The architecture is already device-generic and this is under-claimed:
`param_shift_grad_batch` never inspects a state, routes through
`Backend.expectation_over_slots`, and works on a sampling-only device;
`examples/toward_hardware.py` is a mock QPU with no statevector at all.

What is missing is the *contract* that lets someone else prove their backend works.
Verified absent: there is **no `plugins.py`, no `serialize.py`, and no
`qmlkit/testing/`**, despite a note elsewhere claiming they landed. They never did.

- **Entry-point discovery**, so a third-party backend is installable and found
- **A conformance suite that ships inside the package**, runnable as
  `python -m qmlkit.testing --backend=...`, so conformance is demonstrable rather
  than asserted
- **Capability declarations** already exist in spirit (`supports_exact`,
  `supports_statevector`); make them the thing the suite checks

This is the sole-author-compatible version of "generic for any simulator or QPU": ship
the contract, not the hardware. **0.x stays simulator-only for shipped backends** -
real devices break exact-by-default, refuse adjoint and backprop, and drag in queues,
credentials, native gate sets and mitigation. Revisit for 1.0.

Whenever a backend gains or loses a capability, sweep the public API against it. The
noisy backends broke `diagnose()`, `expressibility`, `entangling_capability`,
`fidelity_samples`, `metric_tensor` and `qng_step`, and the test suite did not notice.

### 7. A stability promise with teeth

PennyLane ships ~4 releases a year with a 1-2 release deprecation window - under six
months from deprecated to removed, 11 pending removals in v0.46 alone - and "my code
broke on upgrade" is a constant on their forum. A research user's paper outlives that
window.

`docs/about/stability.md` already states what is and is not promised. Add an explicit
policy: a minimum deprecation window in *months*, what the promise covers, what it does
not. Cheap, and structurally unmatchable by a vendor shipping at that cadence.

### 8. Real users - the only thing that can calibrate the thresholds

Still the highest-value non-feature item, and items 3 and 4 add to the pile: the
thresholds in `qmlkit.diagnostics` (`_FLAT`, `_CONCENTRATED`), the imbalance cutoff in
`qmlkit.imbalance`, the fold-spread verdict rule in `baselines` and `search`, and the
prune levels in `qmlkit.search` are each one person's opinion until somebody else runs
them and disagrees.

### 9. Smaller, worth doing

- A benchmark suite with published reference numbers - what makes a library citable
- **Noise**: neither density-matrix backend differentiates *through* the channel -
  parameter-shift only - which is the one place PennyLane's `default.mixed` is ahead.
  The `diagnose()` thresholds are calibrated on exact gradients and over-fire under
  shot noise
- **A mitigation verdict**, not a mitigation implementation: whether mitigation
  improved an estimate or only traded bias for variance, on identical seeds with the
  shot cost stated. That is `qk.baseline`'s shape pointed at a new question. The
  implementations belong to Mitiq and QEC belongs to Stim
- `QLSTMCell` gate-level circuits and `QGAN` generator/discriminator still take their
  defaults less flexibly than the convention above wants
- Async job submission, the other protocol-changing hardware gap; see
  `examples/toward_hardware.py`

---

## Runnable things

```bash
python examples/quickstart.py            # every layer, end to end
python examples/experiments.py           # H2, MNIST QCNN, breast-cancer VQC (~20 min)
python examples/head_to_head.py qiskit   # same experiment in qmlkit and PennyLane
python examples/compare_pennylane.py     # readable cross-check
python examples/benchmark_pennylane.py   # wall-clock, identical work
python examples/toward_hardware.py       # a mock QPU with no statevector at all
```

```bash
pytest tests/test_pennylane_parity.py    # 301 cross-validation cases
```

---

## Evidence, for when a claim needs backing

- **301 parity cases against PennyLane**, including randomised circuit fuzzing over the
  whole gate set. Four genuine convention differences surfaced and are each pinned by a
  test — see `docs/about/validation.md`.
- **Faster on 13/14 benchmarked operations**, median **1.7×** — against PennyLane's
  *fastest* configuration (`lightning.qubit`, and the O(P) `adjoint_metric_tensor`
  rather than the O(P²) Hadamard-test one). Never quote the `default.qubit` column:
  the older "14/14, median 6.1×" was measured against it and is not a fair comparison.
  Expectation and gradient rows are dispatch overhead and narrow to a tie by 8 qubits;
  the two real results are the **kernel Gram matrix at 69×** and the **metric tensor at
  105× (P=24)**, which is algorithmic and *widens* with parameter count. JAX is not
  installed on the benchmark machine, so jit-compiled PennyLane is untested and
  unclaimed. `docs/about/validation.md` has the table.
- **H₂ exact** — 0.00000 mHa across the whole dissociation curve, from one ADAPT-selected
  operator, against a Hamiltonian computed here from STO-3G integrals.
- **Both ML experiments lose to logistic regression** (97.8% vs 99.7% on MNIST; 89.5% vs
  93.6% on breast cancer). That is in the tutorial rather than omitted from it.
