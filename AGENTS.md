# AGENTS.md

Instructions for a coding agent working **on** qmlkit. To *use* the library, read
[`docs/llms.txt`](docs/llms.txt) instead — it is the same information written for
the caller rather than the contributor.

[`HANDOFF.md`](HANDOFF.md) is the long version of this file: current status, why
each convention exists, and what to do next. Read it before anything non-trivial.
This page is the short list of things that will break the build if you get them
wrong.

## Commands

```bash
pip install -e ".[dev,torch,qiskit,cirq,sklearn,pennylane]"
pytest -q                                  # the whole suite
pytest -q -m "not pennylane"               # faster, skips the 301 parity cases
ruff check src tests && mypy               # both must be clean
python scripts/generate_llms_txt.py        # after any docs or public-API change
python scripts/verify_install.py           # the core really does import with only NumPy
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
