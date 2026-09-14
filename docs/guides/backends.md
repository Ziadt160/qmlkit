# Backends and conventions

One circuit, nine backends, one answer. `tests/test_cross_backend.py` runs the same
circuit zoo through every installed backend and asserts agreement with the NumPy
reference on statevectors, probabilities, expectations over X/Y/Z and two-body terms,
seeded sampling, and parameter-shift gradients.

| | simulator | when |
|---|---|---|
| `numpy` | the built-in reference | the default. Exact, vectorised, fastest up to ~13 qubits |
| `aer` | `AerSimulator(method="statevector")` | **above ~13 qubits** — C++, and it does not degrade the way a Python gate loop does |
| `qiskit` | `quantum_info.Statevector` | Qiskit's reference. Exact, no extra install, and the slowest of the three |
| `cirq` | `cirq.Simulator` | Cirq's reference |
| `spinqit` | SpinQit's simulator | the diploma's own SDK; Python 3.10 only |
| `torch` | a differentiable simulator | `backprop` only |
| `mps` | `AerSimulator(method="matrix_product_state")` | wide but lightly entangled circuits. No statevector, and exact only until the bond dimension truncates |
| `cirq-density` · `qiskit-aer` | density matrices | noise, when you ask for it by name |

### Which one, in practice

The NumPy reference wins at the widths quantum machine learning actually runs at,
because it carries the batch as a leading axis and contracts gate by gate rather than
sample by sample. That advantage is a Python-level one, and it stops paying once the
statevector gets big:

| qubits | `numpy` | `qiskit` | `aer` |
|---|---|---|---|
| 10 | **0.05s** | 0.17s | 0.13s |
| 12 | 0.16s | 0.28s | **0.16s** |
| 14 | 0.33s | 0.63s | **0.21s** |
| 16 | 0.91s | 4.58s | **0.28s** |
| 18 | 2.87s | 17.47s | **0.60s** |

*(a 40×40 Gram matrix on a `ZZFeatureMap(reps=2)`; single machine, exact throughout,
all three agreeing to 2e-16.)*

The crossover is around **12 qubits**, and what sets it is the *Python* cost of the
reference against Aer's C++ kernel: below it the reference is batched and gate-bound,
above it the statevector is large enough that the C++ inner loop wins outright. If you
are working wider than a dozen qubits, `backend="aer"` is the answer; below it, the
default already is.

`NumpyBackend.batch_max_qubits` (11) is a *different* boundary — it is where NumPy
stops carrying the batch as a leading axis and falls back to one sample at a time. The
two sit next to each other and an earlier version of this page claimed they were the
same one. They move independently: making the batched kernel a BLAS `gemm` moved
`batch_max_qubits` from 10 to 11 without touching where Aer takes over.

`aer` is deliberately a separate name rather than a silent upgrade to `qiskit`, so the
simulator that produced a number stays visible in the code that produced it.

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
another SDK; `from_qasm`/`from_qiskit`/`from_pennylane`/`from_cirq` bring one back.

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

`from_cirq` is the interesting one, because Cirq has no gate *names* to look up.
`cirq.S`, `cirq.T` and `cirq.rz` are all a `ZPowGate`; what separates them is the
exponent and the `global_shift`. So the importer classifies rather than reads a table,
and it reads `global_shift` rather than ignoring it — `cirq.X` and `cirq.rx(pi)` differ
by a global phase, which is unobservable alone and *relative* inside a controlled block.

```python
# docs: requires cirq
import cirq
import sympy

t = sympy.Symbol("t")
qubits = cirq.LineQubit.range(2)
spec = qk.from_cirq(cirq.Circuit([cirq.rx(2 * t).on(qubits[0]), cirq.CZ(*qubits)]))
print(spec.n_params, spec.ops[0].params[0])
```

`cirq.rx(2 * t)` carries the exponent `2*t/pi`; multiplied back by pi that is `2*t`,
and `ParamRef` models exactly `scale * theta + offset`, so it survives as
`ParamRef(0, scale=2.0)`. Symbols are indexed by sorted name, matching
`sorted(cirq.parameter_names(circuit))`. Anything nonlinear is refused.

One asymmetry to know about: Cirq has no declared register, so a qubit that no
operation touches is not in the circuit at all. The same logical circuit imports two
qubits wide from Qiskit and one from Cirq.

```python
# docs: requires cirq
print(qk.from_cirq(cirq.Circuit([cirq.X(cirq.LineQubit(0))])).n_qubits)          # 1
print(qk.from_cirq(cirq.Circuit([cirq.X(cirq.LineQubit(0)),
                                 cirq.I(cirq.LineQubit(1))])).n_qubits)          # 2
```

### The convention that matters

Importing is where qubit order goes wrong quietly. Qiskit and QASM are little-endian,
so their qubit `j` becomes qmlkit's `n-1-j` — the exact inverse of what `to_qiskit`
does on the way out. PennyLane and Cirq are big-endian like qmlkit, so their wires
pass through.

Neither claim is taken on trust. `tests/test_import.py` asserts the *statevector*
after `from_qiskit(to_qiskit(spec))` over randomly generated circuits at `1e-12`, and
checks the PennyLane and Cirq importers against those libraries' own simulators. It
also asserts that every gate `to_cirq` emits comes back through `from_cirq`, since a
gate qmlkit can write but not read is a one-way door. A circuit whose gates
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
