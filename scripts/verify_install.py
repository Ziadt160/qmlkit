"""Post-install smoke test: run this against a *built* qmlkit, not a source checkout.

    python -m venv /tmp/clean
    /tmp/clean/bin/pip install dist/qmlkit-*.whl
    /tmp/clean/bin/python scripts/verify_install.py

An editable install hides packaging bugs: it imports straight from ``src/``, so a
module left out of the wheel, a missing ``py.typed``, or a package that only resolves
because the repo happens to be on ``sys.path`` all keep working. This script exercises
the installed artifact instead, and asserts the two promises the README makes about
a bare install:

* qmlkit needs nothing but NumPy;
* a missing optional SDK produces an install command, not an ``ImportError``.

Exits non-zero on the first broken promise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import qmlkit as qk

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'  — ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


print(f"qmlkit {qk.__version__} · numpy {np.__version__} · python {sys.version.split()[0]}")

# --------------------------------------------------------------------------- #
print("\nthe install itself")
# --------------------------------------------------------------------------- #
here = Path(qk.__file__).resolve().parent
check("imported from site-packages, not the source tree", "site-packages" in str(here), str(here))
check("py.typed shipped, so downstream mypy sees the annotations", (here / "py.typed").is_file())
check("version matches the distribution", qk.__version__ == "0.1.0", qk.__version__)

# --------------------------------------------------------------------------- #
print("\nno optional dependency is secretly required")
# --------------------------------------------------------------------------- #
optional = ("torch", "qiskit", "cirq", "spinqit", "sklearn", "matplotlib", "pennylane")
leaked = [m for m in optional if m in sys.modules]
check(
    "importing qmlkit pulls in no optional SDK", not leaked, f"leaked: {leaked}" if leaked else ""
)
check("numpy backend is available", "numpy" in qk.available_backends())

# --------------------------------------------------------------------------- #
print("\na missing backend explains itself")
# --------------------------------------------------------------------------- #
for name in ("spinqit", "qiskit", "cirq"):
    if qk.is_available(name):
        check(f"{name} present, so nothing to diagnose", True)
        continue
    try:
        qk.get_backend(name)
        check(f"{name} raises when unavailable", False, "no exception at all")
    except Exception as exc:  # noqa: BLE001 - we are testing what users would see
        text = str(exc)
        check(
            f"{name} says how to install it",
            "pip install" in text and not isinstance(exc, ImportError),
            repr(text[:70]),
        )

# --------------------------------------------------------------------------- #
print("\nthe library actually computes")
# --------------------------------------------------------------------------- #
check(
    "<Z> of Ry(0.7)|0> is cos(0.7)",
    abs(qk.expval(qk.angle_encode([0.7]), qk.Z(0)) - np.cos(0.7)) < 1e-12,
)

ansatz = qk.hardware_efficient(3, n_layers=2)
theta = ansatz.init(seed=0)
spec = ansatz.build()
obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
reference = qk.grad(spec, theta, obs, method="adjoint")
for method in ("parameter-shift", "hadamard"):
    check(
        f"{method} agrees with adjoint",
        float(np.abs(qk.grad(spec, theta, obs, method=method) - reference).max()) < 1e-10,
    )

check(
    "amplitude encoding reproduces its target",
    np.allclose(
        np.abs(qk.statevector(qk.amplitude_encode([1, 2, 3, 4]))),
        np.abs(np.array([1, 2, 3, 4]) / np.linalg.norm([1, 2, 3, 4])),
        atol=1e-12,
    ),
)
check(
    "a quantum kernel Gram matrix is symmetric with a unit diagonal",
    (lambda K: np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0))(
        qk.QuantumKernel(qk.ZZFeatureMap(2, reps=2))(
            np.random.default_rng(0).uniform(0, np.pi, (5, 2))
        )
    ),
)
check(
    "sampling is reproducible under a seed",
    qk.run_counts(qk.basis_encode([1, 0, 1]), 128, seed=0)
    == qk.run_counts(qk.basis_encode([1, 0, 1]), 128, seed=0),
)
check("the drawer works on a bare install", "RY" in qk.draw(spec.bind(theta)).upper())

# every advertised gradient method either runs or says how to install itself
for method in qk.list_gradient_methods():
    try:
        qk.grad(spec, theta, obs, method=method, **({"seed": 0} if method == "spsa" else {}))
        check(f"gradient method {method!r} runs", True)
    except Exception as exc:  # noqa: BLE001 - this is exactly what a user would hit
        check(
            f"gradient method {method!r} explains its missing extra",
            "pip install" in str(exc) and not isinstance(exc, ImportError),
            repr(str(exc)[:48]),
        )

# --------------------------------------------------------------------------- #
print()
if failures:
    print(f"{len(failures)} broken promise(s): {', '.join(failures)}")
    sys.exit(1)
print("Clean install verified — nothing but NumPy, and everything above ran.")
