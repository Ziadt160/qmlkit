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

**1401 tests on Python 3.14, 956 on SpinQit's 3.10, 0 failures in either.** ruff clean
and `ruff format --check` clean; `mypy --strict` clean over **the whole package** - the
config listed six paths until 2026-08-28 and now lists `src/qmlkit`, so "mypy clean" and
"the package type-checks" finally mean the same thing. **94% coverage**, combined in
CI across every job — the number CI actually computes, not a local estimate. Two of the
twelve coverage files still fail to map (macOS and Windows record different absolute
roots), so the figure is carried by the `full` job, which installs every extra and runs
the whole suite on Linux.

Phases 0–6 are done, plus the algorithm and interoperability work. 0.1.0 is released;
0.1.1 is prepared and untagged.

**Split the release from a branch that has *both* halves of the audit fixes.** The ten
defects were closed on two branches that never met -
`claude/library-motivation-error-correction-bcd9d1` (F1, F3, F4, F5) and
`claude/audit-fixes-0-1-1` (F2, F6-F10), merge base `4e6e6b2`, neither containing the
other. On either tip alone half the audit is still open, and `git subtree split`
publishes whatever branch it is given, so cutting 0.1.1 from the branch *named* for it
would have shipped the torch backend still computing a different circuit. They are
merged on `claude/fix-documented-bugs-e51eb0`; confirm before splitting with
`git merge-base --is-ancestor <each tip> HEAD`.

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

### 1. Release 0.1.1 - 0.1.0 returns wrong numbers and should not be recommended

0.1.0 is on PyPI. 0.1.1 is prepared here and **not yet tagged**. It closes eleven
defects found after 0.1.0 shipped: one a reader hit using the library - a composition
the README recommended that built the wrong circuit without raising - and ten from an
adversarial audit, four of which corrupt a result silently. The worst is the torch
backend computing a different circuit from every other backend, reachable from the
built-in `conv_block(filter="su4")` and inherited by `method="backprop"`. All eleven
are in the version people can `pip install` today. The changelog entry has the whole
account.

Everything on this side is ready, verified 2026-09-13 **on the merged tree**: 1401 tests
green on 3.14 and 956 on SpinQit's 3.10, `ruff check`, `ruff format --check` and `mypy`
clean over the whole package, `mkdocs build --strict` clean, every reproducer in the
audit's own findings re-run and closed in both environments, the wheel and sdist build,
`twine check` passes on both, and a clean venv installing only the wheel pulls in
**numpy and nothing else** before `verify_install.py` passes.

**The tag goes on the subtree-split commit, not on this repository.** There are two:
`qmlkit/` here is the source of truth, and the standalone repo is a
`git subtree split` of it. `release.yml` only exists in the published one, so
`git push origin main --tags` tags the upstream repository and triggers nothing.
`RELEASING.md` has the procedure and what to do when it goes wrong; it now names
`v0.1.1`.

**A PyPI version number can never be reused**, so the tag is deliberately not pushed
until the release is wanted. If the Trusted Publishing setup from 0.1.0 is still in
place, pushing `v0.1.1` is the whole procedure.

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
