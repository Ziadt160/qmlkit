"""What does the OpenQARP backend actually buy, and where does it lose?

Every number in `docs/guides/openqarp.md` comes out of this file. The claim being
tested is narrow: OpenQARP's engine can contract an expectation without returning the
statevector, and sweep one circuit over many parameter sets without returning to
Python between rows. Both are the shape of a training loop. Neither is worth anything
if the answers move, so agreement is measured first and speed second.

Run it before trusting the guide on your machine::

    python scripts/probe_openqarp.py

It needs `pip install "qmlkit[openqarp]"`; the Aer column is skipped when Aer is not
installed.
"""

import sys
import time
from pathlib import Path

# resolve qmlkit from this checkout, not from whatever an editable install points at
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import qmlkit as qk

ROWS = 256


def ansatz(n_qubits: int, layers: int = 3) -> qk.CircuitSpec:
    """A three-layer ry/rz ring - the shape a variational model actually runs."""
    qc = qk.QCircuit(n_qubits)
    for _ in range(layers):
        qc.rotation_layer(("ry", "rz")).entangle("ring")
    return qc.to_spec()


def observable(n_qubits: int) -> qk.Observable:
    return qk.Z(0) + qk.ZZ(0, n_qubits - 1) + 0.3 * qk.X(1)


def best(fn, repeats: int = 3) -> float:
    """Seconds for the fastest of `repeats` runs, after one warm-up."""
    fn()
    return min(_once(fn) for _ in range(repeats))


def _once(fn) -> float:
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


def slot_rows(spec: qk.CircuitSpec, rows: int = ROWS, seed: int = 4) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (rows, len(spec.slots())))


if not qk.is_available("openqarp"):
    raise SystemExit("openqarp is not installed:\n    pip install 'qmlkit[openqarp]'")

native = qk.get_backend("openqarp")  # pay the SDK import before timing anything
derived = qk.get_backend("openqarp", native_expectations=False)
reference = qk.get_backend("numpy")
aer = qk.get_backend("aer") if qk.is_available("aer") else None

print(f"qmlkit   : {qk.__file__}")
print(f"openqarp : {qk.fingerprint().backends['openqarp']}\n")

# --------------------------------------------------------------------------- #
print("does it agree? (max |difference| against the NumPy reference)")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'statevector':>14}{'<O> contracted':>17}{'<O> swept':>12}{'gradient':>11}")
print("-" * 57)
for n in (4, 6, 8, 10):
    spec = ansatz(n)
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, spec.n_params)
    bound = spec.bind(theta)
    obs = observable(n)
    rows = slot_rows(spec, rows=32)

    d_state = np.max(np.abs(native.statevector(bound) - reference.statevector(bound)))
    d_exact = abs(native.expectation(bound, obs) - reference.expectation(bound, obs))
    d_sweep = np.max(
        np.abs(
            native.expectation_over_slots(spec, rows, obs)
            - reference.expectation_over_slots(spec, rows, obs)
        )
    )
    d_grad = np.max(
        np.abs(
            qk.param_shift_grad_circuit(spec, theta, obs, backend="openqarp")
            - qk.param_shift_grad_circuit(spec, theta, obs, backend="numpy")
        )
    )
    print(f"{n:>3}{d_state:>14.1e}{d_exact:>17.1e}{d_sweep:>12.1e}{d_grad:>11.1e}")

# --------------------------------------------------------------------------- #
print(f"\nthe call it exists for: one expectation_over_slots of {ROWS} rows (ms)")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'openqarp':>11}{'numpy':>11}{'aer':>11}{'vs numpy':>11}")
print("-" * 47)
for n in (4, 6, 8, 10, 12, 14):
    spec, obs, rows = ansatz(n), observable(n), slot_rows(ansatz(n))
    t_native = best(lambda: native.expectation_over_slots(spec, rows, obs), 2)
    t_numpy = best(lambda: reference.expectation_over_slots(spec, rows, obs), 2)
    t_aer = best(lambda: aer.expectation_over_slots(spec, rows, obs), 1) if aer else float("nan")
    print(
        f"{n:>3}{t_native * 1e3:>11.1f}{t_numpy * 1e3:>11.1f}{t_aer * 1e3:>11.1f}"
        f"{t_numpy / t_native:>10.1f}x"
    )

# --------------------------------------------------------------------------- #
print("\nwhere that win comes from: the sweep against this backend's own per-row path")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'swept':>10}{'per row':>10}{'ratio':>8}   identical?")
print("-" * 47)
for n in (8, 12, 14):
    spec, obs, rows = ansatz(n), observable(n), slot_rows(ansatz(n))
    swept = best(lambda: native.expectation_over_slots(spec, rows, obs), 2)
    per_row = best(
        lambda: np.array([native.expectation(spec.with_slot_angles(r), obs) for r in rows]), 2
    )
    same = np.array_equal(
        native.expectation_over_slots(spec, rows, obs),
        np.array([native.expectation(spec.with_slot_angles(r), obs) for r in rows]),
    )
    print(
        f"{n:>3}{swept * 1e3:>10.1f}{per_row * 1e3:>10.1f}{per_row / swept:>7.2f}x"
        f"   {'bit-identical' if same else 'DIFFERENT'}"
    )

# --------------------------------------------------------------------------- #
print("\none exact <O>: contracted in the kernel vs derived from the statevector (ms)")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'contracted':>13}{'derived':>10}{'numpy':>10}{'aer':>10}")
print("-" * 46)
for n in (10, 12, 14, 16):
    bound = ansatz(n).bind(np.random.default_rng(1).uniform(-np.pi, np.pi, ansatz(n).n_params))
    obs = observable(n)
    t_aer = best(lambda: aer.expectation(bound, obs), 2) if aer else float("nan")
    print(
        f"{n:>3}{best(lambda: native.expectation(bound, obs)) * 1e3:>13.2f}"
        f"{best(lambda: derived.expectation(bound, obs)) * 1e3:>10.2f}"
        f"{best(lambda: reference.expectation(bound, obs)) * 1e3:>10.2f}{t_aer * 1e3:>10.2f}"
    )

# --------------------------------------------------------------------------- #
print(f"\nand where it does not win: {ROWS} statevectors, which the reference batches (ms)")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'openqarp':>11}{'numpy':>11}{'winner':>10}")
print("-" * 35)
for n in (6, 8, 10, 12):
    spec, rows = ansatz(n), slot_rows(ansatz(n))
    t_native = best(lambda: native.statevector_batch_slots(spec, rows), 1)
    t_numpy = best(lambda: reference.statevector_batch_slots(spec, rows), 1)
    winner = "openqarp" if t_native < t_numpy else "numpy"
    print(f"{n:>3}{t_native * 1e3:>11.1f}{t_numpy * 1e3:>11.1f}{winner:>10}")

# --------------------------------------------------------------------------- #
print("\nend to end: one batched gradient of 32 samples (ms)")
# --------------------------------------------------------------------------- #
print(f"{'n':>3}{'method':>18}{'openqarp':>11}{'numpy':>11}{'vs numpy':>11}")
print("-" * 54)
for n in (8, 10, 12):
    spec = qk.hardware_efficient(n, 2).build()
    thetas = np.random.default_rng(11).uniform(-np.pi, np.pi, (32, spec.n_params))
    obs = observable(n)
    for method, fn in (
        ("parameter-shift", qk.param_shift_grad_batch),
        ("adjoint", qk.adjoint_grad_batch),
    ):
        t_native = best(lambda: fn(spec, thetas, obs, backend="openqarp"), 2)
        t_numpy = best(lambda: fn(spec, thetas, obs, backend="numpy"), 2)
        agree = np.max(
            np.abs(
                fn(spec, thetas, obs, backend="openqarp") - fn(spec, thetas, obs, backend="numpy")
            )
        )
        assert agree < 1e-9, f"{method} disagrees by {agree:.1e}"
        print(
            f"{n:>3}{method:>18}{t_native * 1e3:>11.1f}{t_numpy * 1e3:>11.1f}"
            f"{t_numpy / t_native:>10.1f}x"
        )
