# AGENTS.md

Instructions for a coding agent working **on** qmlkit. To *use* the library, read
[`docs/llms.txt`](docs/llms.txt) instead — it is the same information written for
the caller rather than the contributor.

[`HANDOFF.md`](HANDOFF.md) is the long version of this file: current status, why
each convention exists, and what to do next. Read it before anything non-trivial.
This page is the short list of things that will break the build if you get them
wrong.

## The map

21,000 lines over 86 modules, 214 names in the top-level `__all__`. Two halves:

```
src/qmlkit/
  core/          the IR, and only the IR: ir · builder · gates · observables · execute
  core/backends/ ten of them. base.py supplies the semantics; a backend supplies
                 statevector() and inherits sampling, grouping, expectation, batching
  ansatz/  encoding/  gradients/  kernels/  nn/      the ML layer
  algorithms/    VQE · ADAPT · QAOA · chemistry · autoencoder · clustering · rl
  *.py           the honesty layer — diagnostics · landscape · baselines · budget ·
                 evaluate · imbalance · provenance · search · progress · report · metrics
  utils/errors.py  how every "unknown X" error in the library is built
  _aliases.py      PennyLane and Qiskit names, answered with the qmlkit one
```

The top-level modules are the point of the project. `core/` is machinery in service
of them.

## Commands

```bash
pip install -e ".[dev,torch,qiskit,cirq,sklearn,pennylane]"
pytest -q                                  # the whole suite
pytest -q -m "not pennylane"               # faster, skips the parity cases
ruff check src tests && ruff format --check src tests && mypy
python scripts/generate_llms_txt.py        # after any docs or public-API change
python -m mkdocs build --strict            # after any docs change; CI runs it
python scripts/verify_install.py           # the core really does import with only NumPy
QMLKIT_TORTURE_EXAMPLES=1500 pytest tests/test_torture.py   # ~10 min, before a release
```

SpinQit needs its own interpreter — it ships wheels for Python 3.8–3.10 only and
pins `numpy<2`:

```bash
C:/Users/pc/miniconda3/envs/spinq_env/python.exe -m pytest -m spinqit
```

## Rules the tests enforce

1. **An algorithm owns its loop, not its circuit.** Every model takes `ansatz=` /
   `feature_map=` / `filter=` and must actually use it. `tests/test_injection.py`
   injects two different sizes and asserts the parameter count follows.
2. **Estimators must be scikit-learn clonable.** Constructor arguments stored under
   their own names, plus `SklearnCompatible`. This keeps scikit-learn optional while
   letting QSVC/QSVR run inside `Pipeline` and `GridSearchCV`.
3. **The core depends on NumPy and nothing else.** Not SciPy — use `math.erf` and the
   stdlib. CI installs nothing else in the `core` jobs, and this has been broken once.
4. **Documentation is executable.** `tests/test_docs.py` runs every Python block on
   every page, so an API change and its docs go in the same commit.
5. **Names have to stay findable.** `qmlkit/_aliases.py` maps what PennyLane and
   Qiskit call each thing; `tests/test_agent_api.py` asserts every target still
   exists. Rename a public name and you update that table in the same commit.
6. **`docs/llms.txt` is generated and committed.** Change the docs or the public API
   and regenerate it, or CI fails on the stale copy.

## Traps that have already cost time

- **Never use a NumPy-2-only API** (`np.trapezoid`, `np.in1d`, …) in `src/` or
  `tests/`. SpinQit pins `numpy<2` and the suite must pass in both environments.
- **Always write `npt.NDArray[Any]`, never bare `np.ndarray`.** Type-parameter
  defaults only arrived in NumPy 2.3, so 3.10 CI fails with 60 `type-arg` errors.
- **Never set `python_version` in `[tool.mypy]`.** It makes mypy parse dependency
  stubs at that version too, and NumPy's stubs use PEP 695 `type` statements, which
  are a syntax error before 3.12. Cross-version signal comes from CI running mypy
  on 3.10.
- **The PennyLane parity fuzzer draws gate names from a snapshot taken at import**,
  not from the live registry — other test modules register throwaway gates at run
  time, which made it pass alone and fail in a full run.
- **Extend `tests/test_pennylane_parity.py` when adding a gate.** A test there
  asserts the mapping covers every built-in gate, so a new one cannot escape
  cross-validation. Every bug found in this project has been the
  plausible-wrong-number kind that only a second implementation catches.

## Where a change has to land

The library is built on registries, so adding something is usually one call — and
then three or four places that will not fail loudly if you forget them.

| Adding | Also touch |
|---|---|
| **A gate** | `frequencies` on the `GateDef` or differentiation is refused; `dmatrix` or adjoint is refused; the PennyLane mapping in `tests/test_pennylane_parity.py`, which asserts it covers every built-in gate |
| **An ansatz or conv filter** | `register_*`, and check it is not inert — an all-`rz` filter does nothing at all from `|0…0⟩`, which is why `test_no_shipped_filter_is_inert` exists |
| **A backend** | `supports_exact` and `supports_statevector` are a contract, not metadata. Then **sweep the public API against it**: adding the noisy backends broke `diagnose`, `expressibility`, `entangling_capability`, `fidelity_samples`, `metric_tensor` and `qng_step`, and the suite did not notice |
| **A diagnostic finding** | Write the test so it asserts the finding is **true**, not that it fired. The old tests asserted firing, which is how three wrong findings survived. A false positive here is worse than a false negative: it teaches people to ignore the tool |
| **Anything public** | `docs/llms.txt` regenerated, a reference page entry, and a CHANGELOG entry |
| **A version bump** | `pyproject.toml` **and** `src/qmlkit/__init__.py`. A third hardcoded copy in `scripts/verify_install.py` once failed the release gate |

**Releasing is a tag on this repository**, which `release.yml` turns into a TestPyPI
then PyPI publish over Trusted Publishing - no token is handled anywhere. It asserts
the tag matches `pyproject.toml` before it builds. A PyPI version number can never be
reused, so tags stay unpushed until the release is actually wanted, and a release that
half-completes burns the number. `RELEASING.md` has the procedure.

Until 2026-09-13 this repo was a `git subtree split` of a subdirectory in the lecture
repository. If you find a document that still says so, it is stale - fix it.

## Writing

The prose is a deliberate artefact. Match it rather than inventing a second voice.

- **Claims enter through the failure they prevent**, never through the feature name.
  The order is: here is a way to be wrong, here is why it does not raise, here is the
  call. Almost nothing opens with "qmlkit provides".
- **Every number carries its provenance.** "Measured on a 5-qubit hardware-efficient
  ansatz"; "measured: five frequencies". A bare number reads as an unsupported claim.
- **Caveats get their own sentence, in the same voice as the claims**, usually last and
  usually against the author's interest — "JAX is not installed on the benchmark
  machine, so jit-compiled PennyLane is untested and unclaimed."
- **Headings are assertions, not labels**: "Data re-uploading is a pattern, not a
  structure"; "Noise, when you ask for it by name". Label headings mark reference
  sections, and the contrast is the point.
- A long sentence that does the analysis, then a short one that lands it. British
  spelling. Second person about the reader's actions, never their level.

## Design commitments

- **Simulator-only for the whole 0.x line.** This is a constraint that propagates,
  not a scope trim: it makes `adjoint` the correct default gradient, makes shot
  noise opt-in, and demotes anything whose value is cutting *measurement* cost.
  Parameter-shift stays the teaching subject and the reference that validates
  adjoint — never the performance default.
- **Simple on top, open underneath.** Three layers, and nothing at a higher one
  hides a lower one. Every extension point is a registry.
- **When something is named after a pattern rather than a structure, make it a
  composition, not a class.** Data re-uploading is `EncodingLayer` in the block
  vocabulary, with `reupload()` as a convenience over it.
- **The error message is the documentation.** Most callers are models that will not
  read the docs site; they read the traceback. An error about a name says what was
  wrong, what was probably meant, and what is allowed — build it with
  `qmlkit.utils.errors.unknown`.
