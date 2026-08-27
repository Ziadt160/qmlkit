# Backends and conventions

One circuit, five backends, one answer. `tests/test_cross_backend.py` runs the same
circuit zoo through every installed backend and asserts agreement with the NumPy
reference on statevectors, probabilities, expectations over X/Y/Z and two-body terms,
seeded sampling, and parameter-shift gradients.

```python
import qmlkit as qk

print(qk.available_backends())
print(qk.backend_report())
```

Every SDK import is lazy, so `import qmlkit` requires none of them and a missing one
produces an install command rather than an `ImportError`. Set the default with
`QMLKIT_BACKEND` or `qk.set_default_backend(...)`, or pass `backend=` per call.

## Qubit ordering

**qmlkit is big-endian: qubit 0 is the most significant bit.** A count key `'011'`
means qubit 0 measured `|0⟩`, qubit 1 `|1⟩`, qubit 2 `|1⟩`. This matches SpinQit and
PennyLane.

Qiskit is little-endian. Rather than reversing statevectors after the fact, the
Qiskit backend maps qmlkit qubit `i` to Qiskit qubit `n−1−i` **at build time**, so
the index conventions coincide and no reversal is needed anywhere downstream.

## Three upstream discrepancies

Building the cross-backend suite turned up three real differences. All are handled;
all are worth knowing about if you go looking at the native circuits.

| Finding | Handling |
|---|---|
| **SpinQit's `CY` applies `−iY`**, not `Y`, to the control-1 subspace | Emitted as `Sd·CX·S` instead. This is a *relative* phase between control branches, so it is physically observable — a control qubit in superposition gives different measurement statistics. SpinQit's single-qubit `Y` is correct; only the controlled form is affected |
| **Cirq silently drops qubits a circuit never touches** | An explicit `qubit_order` is always passed, so an idle qubit still occupies its place in the statevector |
| **Qiskit is little-endian** | Index remapping at build time, as above |

`verify_conventions()` re-checks bit order and gate definitions against a live
install in one call — worth running after an SDK upgrade.

## Precision

SpinQit's simulator carries a floor near `1e-10` rather than machine precision: a
single-qubit `Ry(0.7)` expectation lands about `5.6e-11` from the analytic `cos(0.7)`.
Cross-backend comparisons use a per-backend tolerance so this is not mistaken for a
translation error, and it is worth knowing before anyone reports a "gradient
mismatch" that is really accumulated simulator noise.

```text
TOLERANCE = {"spinqit": 1e-7, "qiskit": 1e-9, "cirq": 1e-9}
```

## Native circuits

Each backend exposes its own object, so you can hand a circuit to that SDK's
transpiler or drawing tools:

```python
# docs: requires qiskit
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
print(type(qk.get_backend("qiskit").to_qiskit(qc.to_spec())).__name__)
```

`to_qiskit`, `to_cirq` and `to_spinqit` are the three.

## Reading circuits in

The translations run both ways. `to_qiskit`/`to_cirq`/`to_spinqit` hand a circuit to
another SDK; `from_qasm`/`from_qiskit`/`from_pennylane` bring one back.

```python
import qmlkit as qk

spec = qk.from_qasm("""
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0],q[1];
""")
print(spec.n_qubits, len(spec.ops))
```

`from_qasm` uses the standard library alone, so it works in a bare `pip install
qmlkit`. Every major SDK exports OpenQASM 2.0, which makes it the widest import path
the library has.

`from_qiskit` exists next to it for the one thing QASM cannot represent — an unbound
`Parameter`, which becomes a `ParamRef` indexed in Qiskit's own parameter order:

```python
# docs: requires qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

qc = QuantumCircuit(2)
qc.ry(Parameter("theta"), 0)
spec = qk.from_qiskit(qc)
print(spec.n_params)
```

### The convention that matters

Importing is where qubit order goes wrong quietly. Qiskit and QASM are little-endian,
so their qubit `j` becomes qmlkit's `n-1-j` — the exact inverse of what `to_qiskit`
does on the way out. PennyLane is big-endian like qmlkit, so its wires pass through.

Neither claim is taken on trust. `tests/test_import.py` asserts the *statevector*
after `from_qiskit(to_qiskit(spec))` over randomly generated circuits at `1e-12`, and
checks the PennyLane importer against PennyLane's own simulator. A circuit whose gates
are symmetric across the register cannot tell a correct mapping from a reversed one,
so the test zoo is deliberately asymmetric.

### What is refused

A gate with no qmlkit definition raises `UnsupportedGate` naming it, rather than being
dropped or approximated. So do `measure` and `reset`, which the 0.x line does not
model. The single exception is the `u`/`u3` family: it is decomposed into `rz·ry·rz`
and warns that an overall phase was dropped — unobservable for the circuit alone, and
a *relative* phase if that circuit is later used inside a controlled block.

## SpinQit needs its own environment

SpinQit ships wheels for Python 3.8–3.10 only and pins `numpy<2`, so the extra is
gated behind an environment marker and resolves to nothing on 3.11+. Use a dedicated
3.10 environment:

```bash
conda create -n spinq python=3.10 && conda activate spinq && pip install "qmlkit[spinqit]"
```

A practical consequence: **nothing in the library may use a NumPy-2-only API**
(`np.trapezoid`, `np.in1d`, …), because the test suite has to pass under `numpy<2`
as well. CI runs both.

## The torch backend

`TorchBackend` is a differentiable statevector simulator, and it is what makes
`method="backprop"` possible. It is the least physical backend here — deliberately —
and exists because a circuit inside an autograd graph is genuinely useful, not
because it could ever run anywhere but a simulator.
