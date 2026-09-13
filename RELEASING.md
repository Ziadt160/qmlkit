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
   (`__version__`). Those are the only two - `verify_install.py` reads the packaging
   metadata and checks the module constant against it, rather than carrying a third
   copy that has to be remembered.

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

   On Windows the venv puts them in `Scripts` rather than `bin`:

   ```powershell
   python -m venv $env:TEMP\clean
   & "$env:TEMP\clean\Scripts\python.exe" -m pip install (Get-Item dist\qmlkit-*.whl)
   & "$env:TEMP\clean\Scripts\python.exe" scripts\verify_install.py
   ```

5. **Publish the subtree, then tag it *there*.** This is the step that is easy to get
   wrong, because there are two repositories.

qmlkit is developed as the `qmlkit/` subdirectory of a private upstream working
   repository, and **published** as `github.com/Ziadt160/qmlkit` — a `git subtree
   split` of that subdirectory. `release.yml` exists only in the published repo, so
   tagging upstream triggers nothing at all.

   Commit inside `qmlkit/` first, then, from the upstream repository root:

   ```bash
   git branch -D qmlkit-standalone 2>/dev/null
   git subtree split --prefix=qmlkit -b qmlkit-standalone
   git push qmlkit qmlkit-standalone:main
   ```

   The split is deterministic, so re-running it fast-forwards rather than diverging.
   It prints the SHA of the split commit — that is what the tag goes on:

   ```bash
   git tag v0.2.0 $(git rev-parse qmlkit-standalone)
   git push qmlkit v0.2.0
   ```

   The tag must match the packaged version exactly; the workflow checks and refuses
   otherwise.

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
  anything. Delete the tag, fix the version, tag again:

  ```bash
  git push qmlkit :refs/tags/v0.2.0   # delete it on the remote
  git tag -d v0.2.0                   # and locally
  ```

- **Nothing happened when you pushed the tag** — you almost certainly tagged the
  upstream repository instead of the published one. `release.yml` only exists in the
  published repo; check with `git ls-remote --tags qmlkit`.

- **The split branch will not push** — it is deterministic, so a rejected push means
  the remote has commits the split does not contain (someone committed directly to
  the published repo). Reconcile there first; never force-push over it, because a
  published tag must keep pointing at the commit that produced the artifact.
