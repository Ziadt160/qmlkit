# Install

```bash
pip install qmlkit
```

That is the whole install. qmlkit depends on **NumPy and nothing else** — the NumPy
backend is the reference implementation, and every core feature works with no
optional package present.

Python 3.10 or newer. 3.9 reached end of life in October 2025, and 3.10 is also the
highest version SpinQit supports, so it is the overlap that matters.

## Extras

Each extra adds one capability. None of them is required, and `import qmlkit` never
imports any of them — a missing SDK produces an install command, not a traceback.

=== "PyTorch"

    ```bash
    pip install "qmlkit[torch]"
    ```

    Unlocks `QuantumLayer`, `VQC`, `VQRegressor`, the structured architectures, and
    `method="backprop"`.

=== "Qiskit"

    ```bash
    pip install "qmlkit[qiskit]"
    ```

    Adds the Qiskit backend and `to_qiskit()`, so a circuit can be handed to Qiskit's
    own transpiler and visualisation.

=== "Cirq"

    ```bash
    pip install "qmlkit[cirq]"
    ```

=== "SpinQit"

    ```bash
    pip install "qmlkit[spinqit]"
    ```

    SpinQit ships wheels for Python 3.8–3.10 only and pins `numpy<2`, so the extra is
    gated behind an environment marker: on 3.11+ it resolves cleanly to nothing rather
    than failing. Use a dedicated 3.10 environment for it.

=== "Everything"

    ```bash
    pip install "qmlkit[torch,qiskit,cirq,sklearn,viz]"
    ```

`sklearn` enables `QSVC`/`QSVR` (which wrap scikit-learn's precomputed-kernel solver)
and `viz` enables matplotlib plotting helpers.

## Check what you have

```python
import qmlkit as qk

print(qk.__version__)
print(qk.available_backends())
print(qk.backend_report())
```

`backend_report()` prints one line per backend with an install command for the ones
you are missing:

```text
qmlkit backends:
  [ok]      numpy
  [ok]      torch
  [missing] cirq     -> pip install 'qmlkit[cirq]'
  [missing] qiskit   -> pip install 'qmlkit[qiskit]'
  [missing] spinqit  -> pip install 'qmlkit[spinqit]'
```

Set the default with the `QMLKIT_BACKEND` environment variable, or in code:

```python
import qmlkit as qk

qk.set_default_backend("numpy")
print(qk.default_backend().name)
```

## From source

```bash
git clone https://github.com/Ziadt160/qmlkit
cd qmlkit
pip install -e ".[dev]"
pytest
```

## Verifying an install you did not build

If you want to confirm a release is intact — that nothing is missing from the wheel
and no optional dependency is secretly required — the repository ships the check it
runs in CI:

```bash
python -m venv /tmp/clean && /tmp/clean/bin/pip install qmlkit && /tmp/clean/bin/python scripts/verify_install.py
```
