# Releasing

A PyPI version number can never be reused, and a release cannot be edited after the
fact — only yanked. Everything below exists to make the irreversible step the last
one, and to make it boring.

## One-time setup

PyPI publishing uses [Trusted Publishing][tp], so there is no API token in the
repository and nothing to rotate or leak. It has to be registered on PyPI first.

For the **first** release, the project does not exist on PyPI yet, so register a
*pending* publisher: PyPI → Your account → Publishing → Add a pending publisher.

| Field | Value |
|---|---|
| PyPI project name | `qmlkit` |
| Owner | `Ziadt160` |
| Repository | `qmlkit` |
| Workflow | `release.yml` |
| Environment | `pypi` |

Repeat the same on [TestPyPI][testpypi] with environment `testpypi`.

Then create both environments in the GitHub repository (Settings → Environments):
`testpypi` and `pypi`. Adding a required reviewer to `pypi` is worth it — it turns
the final upload into something you approve by hand.

[tp]: https://docs.pypi.org/trusted-publishers/
[testpypi]: https://test.pypi.org/manage/account/publishing/

## Cutting a release

1. **Update the changelog.** Move `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`.
   Every entry should say what changed and why, not just what was added.

2. **Set the version in two places**, which must agree or the workflow refuses:
   `pyproject.toml` (`project.version`) and `src/qmlkit/__init__.py`
   (`__version__`).

3. **Run the whole thing locally**, in both interpreters — no single one can import
   all four backends:

   ```bash
   ruff check src tests && ruff format --check src tests && mypy && pytest
   ```

   ```bash
   C:/Users/pc/miniconda3/envs/spinq_env/python.exe -m pytest      # SpinQit, Python 3.10
   ```

4. **Verify the built artifact, not the source tree.** An editable install imports
   out of `src/` and keeps working even if a module never made it into the wheel:

   ```bash
   python -m build && twine check dist/*
   ```

   ```bash
   python -m venv /tmp/clean && /tmp/clean/bin/pip install dist/qmlkit-*.whl && /tmp/clean/bin/python scripts/verify_install.py
   ```

5. **Commit, then tag.** The tag must match the packaged version exactly:

   ```bash
   git tag v0.1.0 && git push origin main --tags
   ```

6. **Watch the workflow.** It runs the suite again on the tagged commit, builds,
   re-verifies the wheel in a clean environment, publishes to TestPyPI, and only
   then publishes to PyPI.

7. **Install it the way a stranger would**, from a machine that has never seen the
   source:

   ```bash
   pip install qmlkit && python -c "import qmlkit as qk; print(qk.__version__, qk.backend_report())"
   ```

## If something goes wrong

- **Bad metadata or a broken README on TestPyPI** — fix it, bump to the next patch
  version, tag again. Do not try to reuse the number.
- **Already published to PyPI and it is broken** — `yank` the release rather than
  deleting it. Yanking leaves existing pins working while stopping new installs from
  resolving to it.
- **The tag does not match `pyproject.toml`** — the workflow fails before publishing
  anything. Delete the tag, fix the version, tag again.
