# Backends and conventions

One circuit, ten backends, one answer. `tests/test_cross_backend.py` runs the same
circuit zoo through every installed backend and asserts agreement with the NumPy
reference on statevectors, probabilities, expectations over X/Y/Z and two-body terms,
seeded sampling, and parameter-shift gradients.

| | simulator | when |
|---|---|---|
| `numpy` | the built-in reference | the default. Exact, vectorised, fastest up to ~13 qubits |
| `aer` | `AerSimulator(method="statevector")` | **above ~13 qubits** — C++, and it does not degrade the way a Python gate loop does |
| `qiskit` | `quantum_info.Statevector` | Qiskit's reference. Exact, no extra install, and the slowest of the three |
| `cirq` | `cirq.Simulator` | Cirq's reference |
| `openqarp` | OpenQARP's `QarpSimulator` | **batched expectations** — swept in C++, several times faster than anything else here above ~8 qubits |
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

### When the answer is an expectation

Every backend above answers an expectation the same way: produce the statevector, hand
it back, contract it here. `openqarp` is the one that does not. It contracts inside the
simulator, and it will sweep a whole batch of parameter vectors without returning to
Python between rows — which is the shape of nearly everything this library asks a
backend for. A batched parameter-shift gradient is `2P × batch` circuits behind one
call, and so is every forward pass of `QuantumLayer`.

| 256 rows, 3-layer `ry`/`rz` ring | `numpy` | `aer` | `openqarp` |
|---|---|---|---|
| 8 qubits | 52 ms | 556 ms | **17 ms** |
| 10 qubits | 217 ms | 723 ms | **38 ms** |
| 12 qubits | 1043 ms | 972 ms | **156 ms** |
| 14 qubits | 2591 ms | 1498 ms | **626 ms** |

*(one `expectation_over_slots` call on `Z0 + Z0·Z(n−1) + 0.3·X1`; exact throughout, all
three agreeing to `1e-15`.)*

For a single statevector it is a different picture — competitive to about 16 qubits and
behind Aer above that — so this is the backend to reach for when the work is
expectations, which in this library it usually is.

```python
# docs: requires qarp
import numpy as np

import qmlkit as qk

spec = qk.hardware_efficient(4, 2).build()
thetas = np.random.default_rng(0).uniform(-np.pi, np.pi, (32, spec.n_params))
print(qk.get_backend("openqarp").expectation_over(spec, thetas, qk.Z(0)).shape)
```

Where it does not win is worth knowing too, and both places are structural. The NumPy
reference carries a batch as a *leading axis* and contracts gate by gate, so anything
that asks for many statevectors at once — an adjoint gradient, a fidelity kernel —
gets a whole batch in one contraction there and one row at a time here: 256 states of
an eight-qubit ansatz cost 104 ms on this backend against 45 ms on the reference. That
reverses between eight and ten qubits (150 ms against 212 ms at ten), for the same
reason the Aer crossover exists. And at four or five qubits nothing here matters next
to Python overhead per call, so a small `VQC` trains no faster — with a second of SDK
import before the first circuit runs.

So: reach for it when the answer is an expectation and the register is wide enough to
notice. `grad_method="parameter-shift"` is where it shows most, because that method is
made of expectations; `adjoint`, which is the default, is made of statevectors and
lands roughly where the reference does.

Because the number now comes out of the SDK rather than out of this library's own
contraction, there are two routes to it and `native_expectations=False` is the other
one — same backend, same circuit, expectation derived from the statevector instead.
`tests/test_openqarp_backend.py` runs both and compares, which is the only reason to
trust the fast one.

The install is `pip install "qmlkit[openqarp]"`, Python 3.11+ — OpenQARP publishes no
wheel below it, so the extra is gated by an environment marker and resolves to nothing
on 3.10 rather than failing. It is also the heaviest dependency here: the compiled core
arrives with SciPy, SymPy, NetworkX, Matplotlib and IPython behind it.

**[Integrating OpenQARP](openqarp.md)** is the long form — what the translation is made
of, where the endianness map shows up in the block's own QASM, how a registered gate
survives the crossing, and every measurement above with the script that produced it.

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

Qiskit and OpenQARP are little-endian. Rather than reversing statevectors after
the fact, both backends map qmlkit qubit `i` to that SDK's qubit `n−1−i` **at build
time**, so the index conventions coincide and no reversal is needed anywhere
downstream. On OpenQARP that is also the faster of the two: its own
`lsb_to_msb_statevector` builds the permutation from a Python loop over `2**n`
formatted strings, 40 ms at 16 qubits against the 11 ms simulation it would be
decorating.

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
TOLERANCE = {"spinqit": 1e-7, "qiskit": 1e-9, "cirq": 1e-9, "openqarp": 1e-9}
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

`to_qiskit`, `to_cirq`, `to_spinqit` and `to_openqarp` are the four.

An OpenQARP block brings its own exporters with it, which makes the way back a
one-liner:

```python
# docs: requires qarp
import numpy as np

import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1).ry(1, 0.7)
spec = qc.to_spec()
block = qk.get_backend("openqarp").to_openqarp(spec)
back = qk.from_qasm(block.to_qasm2())
print(np.allclose(qk.statevector(spec), qk.statevector(back)))
```

qarp writes little-endian QASM, the same convention Qiskit's exporter uses, and
`from_qasm` already inverts that — so the `n−1−i` map applied on the way out is undone
on the way back, and the circuit returns as the one you started with rather than as its
mirror image.

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
