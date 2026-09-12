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
| **PyPI** | **Published.** `pip install qmlkit` installs 0.1.0 (2026-09-12) |
| **Released version** | `0.1.0`, tagged `v0.1.0` at the split commit. **It has a known wrong-number bug — see below** |

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

## Read this first: 0.1.0 is published and returns a wrong number

`pip install qmlkit` works. It should not be recommended to anyone yet.

**The torch backend in 0.1.0 evaluates a different circuit than the one it was
given.** `_apply_torch` reimplements `np.moveaxis` without the
`sorted(zip(destination, source))` numpy does, so any two-qubit gate on wires
`(a, b)` with `a > b` permutes the *untouched* wires. `Z(3)` on a four-qubit circuit
reads `-0.1288` where NumPy, Qiskit and Cirq all agree on `-0.7374`. Nothing raises.
`method="backprop"` differentiates through the same function, so its gradients are
wrong too, and `qk.conv_block(filter="su4")` reaches it without anyone hand-writing a
circuit.

It is **fixed on this branch and not yet released**. Cutting 0.1.1 is the first
priority, and its release note should say plainly that 0.1.0 returns wrong numbers
under that condition — anyone who ran a `backprop` gradient on it has no way to know.

Two things worth carrying forward from how it was found. `qk.selfcheck` catches it and
names the cause exactly; nothing was running `selfcheck`. And the cross-backend suite
could not: it compares the backends against *each other*, and the torch backend was
excluded from the property that would have noticed. An agreement test that skips a
participant tests nothing about that participant.

## What is fixed here and waiting for 0.1.1

Everything below is committed on `claude/library-motivation-error-correction-bcd9d1`
and absent from the published 0.1.0:

| | why it matters |
|---|---|
| torch backend axis permutation | silent wrong number, above |
| `expectation(return_std=True)` | reported `+-0.00000` for any multi-term observable once \|<O>\| reached 1 — every molecular Hamiltonian got a fake error bar |
| `info.purity(..., backend=<mixed>)` | returned a hard-coded `1.0` for a state whose purity was 0.309 |
| `ENCODING_COMMUTES` false positive | fired at *error* severity on correct architectures — **including the example in the README** |
| `VQC` could not express data re-uploading | the pattern the docs recommend most was unreachable from the class they recommend first |
| `AnsatzReport` trainability | probed a dead parameter and reported `1.4e-32`, which reads as a barren plateau |
| `qk.draw` on Windows | `UnicodeEncodeError` on a cp1252 console, i.e. printing a circuit killed the script |
| `list_baselines()` / `Scores.get()` | a duplicated name; a near-miss metric key returning `None` silently |

Plus new: `qk.optim.minimize_adam`, `qk.evaluate.selective` / `risk_coverage`,
`from_cirq`, the mixed-state backends, Study 8, and `tests/test_torture.py`.

## Two fix sessions may still be running

Started from this session and working in their own worktrees. **Check before editing
their files**, and let them land before tagging 0.1.1:

* `EncodingLayer` slot aliasing plus the observable algebra (`ansatz/blocks.py`,
  `core/observables.py`)
* six defects from the adversarial audit (`kernels/`, `info.py`, `evaluate.py`,
  `diagnostics.py`) — the worst is `QuantumKernel(estimator="hadamard")` returning
  `Re(<x'\|x>)**2` instead of `\|<x'\|x>\|**2`

## Status

**1370 tests on Python 3.14, 684 on a bare py3.10 install, 925 on SpinQit's 3.10,
0 failures in any.** ruff clean
and `ruff format --check` clean; `mypy --strict` clean over **the whole package** - the
config listed six paths until 2026-08-28 and now lists `src/qmlkit`, so "mypy clean" and
"the package type-checks" finally mean the same thing. **94% coverage**, combined in
CI across every job — the number CI actually computes, not a local estimate. Two of the
twelve coverage files still fail to map (macOS and Windows record different absolute
roots), so the figure is carried by the `full` job, which installs every extra and runs
the whole suite on Linux.

Phases 0–7 are done: 0.1.0 is on PyPI and the release workflow ran green end to end,
so the mechanism is proven rather than hoped for. What is open is 0.1.1.

### How the library is checked, in order of how much it proves

1. **`tests/densesim.py`** — a dense reference that shares *nothing* with qmlkit.
   Hand-written gate matrices, hand-derived derivatives, reads only `spec.ops`, calls
   no backend. Four properties in `test_torture.py` run it against random circuits.
   This is the only check that can catch a mistake every other one would make
   together, and it is what found the torch bug.
2. **`tests/test_pennylane_parity.py`** — 301 cases against a second library.
3. **`tests/test_torture.py`** — property-based, Hypothesis, invariants that hold by
   mathematics. Depth is tunable: `QMLKIT_TORTURE_EXAMPLES=1500` before a release runs
   ~19,500 circuits and takes about ten minutes. Hypothesis shrinks a failure to the
   smallest circuit that shows it and replays it thereafter.
4. **`tests/test_cross_backend.py`** — the five backends against the NumPy reference.
5. The rest of the suite.

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

**Gate and gradient registries are process-wide.** Other test modules register throwaway
gates at run time, so anything iterating them must snapshot at import.

**SpinQit's `CY` is wrong** — it applies `−iY`, a physically observable relative phase.
Emitted as `Sd·CX·S`. Its simulator also carries a `1e-10` precision floor.

---

## What to do next

### 1. Cut 0.1.1 — a correctness release

0.1.0 is published and returns a wrong number (top of this file). Everything needed is
committed on this branch; what remains is to let the two in-flight fix sessions land,
re-verify, and tag.

The release mechanism is proven — it ran green on 2026-09-12 — and `RELEASING.md` has
the steps. Two things that are easy to get wrong and cost a version number, because a
PyPI version can never be reused:

* **The tag goes on the published repo, not upstream.** `release.yml` exists only in
  the subtree split. Tagging upstream triggers nothing and looks like a broken
  workflow; check with `git ls-remote --tags qmlkit`.
* **Push the split branch first and let CI go green before tagging.** This caught a
  py3.10-only mypy failure that no local environment on this machine could reproduce.
  One CI cycle is cheap; a burnt version number is not.

### 1b. Run the checks in the environment CI actually uses

A green local check has meant nothing three times now, and each time the cause was a
package this machine has and CI does not:

| what looked fine | what CI saw |
|---|---|
| `ruff format --check` | red across 18 untouched files — `dev` allowed `ruff>=0.5` and CI installed a newer one. Now pinned `>=0.15,<0.16` |
| `mypy` | 13 errors — `qmlkit.nn` subclasses `torch.nn.Module`, and CI's `[dev]` install has no torch |
| `pytest` | 9 failures — `test_search.py` had no torch guard; two doc blocks did not declare their extra |
| `mypy` again | 6 errors on **py3.10 only** — NumPy 2.3 gave `ndarray`'s shape parameter a default, and 2.2 is the newest NumPy supporting 3.10 |

So before believing anything, build the environment and run it there:

```bash
python -m venv C:/Users/pc/AppData/Local/Temp/qkci     # SHORT path: no long-path support here
C:/Users/pc/AppData/Local/Temp/qkci/Scripts/python.exe -m pip install -e ".[dev,pennylane]"
```

then `ruff`, `mypy` and `pytest` from that interpreter. A second venv on Python 3.10
(`numpy==2.2`) covers the `core` matrix job. SpinQit's env is a *third* NumPy
generation (`1.26`) and catches NumPy-2-only APIs — `np.trapezoid` and `np.in1d` are
both written down as traps in this file and both have been walked into anyway.

### 2. Real users — the only thing that can calibrate the thresholds

The highest-value item after PyPI, and it is not a feature. Several judgement calls are
baked in and none has been exercised by anyone who did not write them: the thresholds in
`qmlkit.diagnostics` (`_FLAT`, `_CONCENTRATED`), the imbalance cutoff in
`qmlkit.imbalance`, the fold-spread verdict rule in `baselines` and `search`, and the
prune levels in `qmlkit.search`. Every one is one person's opinion until somebody else
runs it and disagrees.

Putting the library in front of a cohort — anyone solving whole problems with it rather
than reading it — is what turns those into calibrated numbers.

**Resist adding features before that happens.** The library is 190+ exports maintained
by one author, and the bottleneck stopped being capability several releases ago.

### 3. Batched submission on a real device

`Backend.expectation_over_slots` is now the single call a provider would turn into a
job, and `param_shift_grad_batch` routes a whole batch's gradient through it without
ever inspecting a state — so the protocol change is made. What remains is a backend
that submits a *list* and polls, plus async. Those two are the items in
`examples/toward_hardware.py` that would still change the `Backend` protocol.

### 4. Smaller, worth doing

- A benchmark suite with published reference numbers — what makes a library citable
- **Noise**: `cirq-density` and `qiskit-aer` evolve a density matrix (see
  `docs/guides/noise.md`). Neither differentiates *through* the channel - parameter-shift
  only - which is the one place PennyLane's `default.mixed` is ahead. The `diagnose()`
  thresholds are still calibrated on exact gradients and will over-fire under shot noise
- **A mitigation verdict**, not a mitigation implementation: whether mitigation improved
  an estimate or only traded bias for variance, on identical seeds with the shot cost
  stated. That is `qk.baseline`'s shape pointed at a new question. The implementations
  belong to Mitiq and QEC belongs to Stim
- `QLSTMCell` gate-level circuits and `QGAN` generator/discriminator still take their
  defaults less flexibly than the convention above wants
- Hardware: batched submission and async jobs are the two gaps that would change the
  `Backend` protocol; see `examples/toward_hardware.py`, which states the rest

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
- **Faster on 14/14 benchmarked operations**, median 6.1×. Sections 1–4 are dispatch
  overhead and narrow with qubit count; the metric tensor is algorithmic and widens.
- **H₂ exact** — 0.00000 mHa across the whole dissociation curve, from one ADAPT-selected
  operator, against a Hamiltonian computed here from STO-3G integrals.
- **Both ML experiments lose to logistic regression** (97.8% vs 99.7% on MNIST; 89.5% vs
  93.6% on breast cancer). That is in the tutorial rather than omitted from it.
