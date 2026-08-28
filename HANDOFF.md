# Handoff

Where the project stands, how to work on it, and what to do next. Written to be the
first thing read in a fresh session.

---

## Where everything lives

| | |
|---|---|
| **Repository** | <https://github.com/Ziadt160/qmlkit> — public, Apache-2.0 |
| **Documentation** | <https://ziadt160.github.io/qmlkit/> — deploys from `main` |
| **Source of truth** | The `qmlkit/` subdirectory of the lecture repo (`Quantum-Machine-Learning-Module`) |
| **PyPI** | **Not published.** `pip install qmlkit` does not work yet |

### The one non-obvious thing about the workflow

The standalone repo is produced by a **subtree split** from the lecture repo. Commit
inside `qmlkit/` there, then:

```bash
git branch -D qmlkit-standalone
git subtree split --prefix=qmlkit -b qmlkit-standalone
git push qmlkit qmlkit-standalone:main
```

The split is deterministic, so re-pushing fast-forwards. `LIBRARY_PLAN.md` stays in the
lecture repo — it maps lectures to library features and belongs with the lectures.

---

## Status

**1202 tests, 0 failures.** ruff + `mypy --strict` clean. **94% coverage**, combined in
CI across every job — the number CI actually computes, not a local estimate. Two of the
twelve coverage files still fail to map (macOS and Windows record different absolute
roots), so the figure is carried by the `full` job, which installs every extra and runs
the whole suite on Linux.

Phases 0–6 are done, plus the algorithm and interoperability work. Phase 7 (release) is
the open one.

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

### 1. Publish to PyPI — the real bottleneck

Nothing else matters as much: none of this is reachable until `pip install qmlkit`
works. Everything on this side is ready — the wheel builds, `twine check` passes, and
the clean-venv verifier passes.

**It needs your account, and cannot be done for you.** Trusted Publishing requires a
publisher registered while signed in to PyPI; the alternative is an API token, which is
a credential Claude does not handle.

1. <https://pypi.org/manage/account/publishing/> → add a *pending publisher*: project
   `qmlkit`, owner `Ziadt160`, repo `qmlkit`, workflow `release.yml`, environment `pypi`
2. Same at <https://test.pypi.org/manage/account/publishing/> with environment `testpypi`
3. GitHub → Settings → Environments → create `pypi` and `testpypi`

Then pushing the tag `v0.1.0` does the rest. `RELEASING.md` has the full procedure and
what to do when it goes wrong. **A PyPI version number can never be reused**, so the
tag is deliberately not pushed yet.

### 2. The seven lecture notebooks — and the first real users

Phase 7's stated acceptance test, and the highest-value item after PyPI. Every lecture
rewritten to `import qmlkit as qk`, every snippet still running, the notebooks getting
*shorter*. Your students are the first people who will use this without having written
it, and they are the only thing that can calibrate the judgement calls now baked in:
the thresholds in `qmlkit.diagnostics` (`_FLAT`, `_CONCENTRATED`), the imbalance cutoff
in `qmlkit.imbalance`, the fold-spread verdict rule in `baselines` and `search`, and the
prune levels in `qmlkit.search`. Every one of those is one person's judgement until
somebody else runs it.

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
