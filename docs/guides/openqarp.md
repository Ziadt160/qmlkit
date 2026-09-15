# Integrating OpenQARP

[OpenQARP](https://github.com/OpenQARP/openqarp) — Fujitsu's Open Quantum Application
Research Package, open-sourced in September 2026 under Apache-2.0 — is a Python
framework over a compiled C++ core. It is assembled from three interchangeable layers:
**blocks** describe circuits, **primitives** say what to extract from them, and
**engines** say how they run, with some seventy building blocks and twenty-odd
ready-to-run algorithms on top.

qmlkit takes one layer of that: the **engine**, `qarpx.QarpSimulator`. This library
already has the feature maps, the ansatz vocabulary, the kernels, the gradients and the
torch bridge; what it wanted from OpenQARP is a fast place to run them — and two entry
points that no other backend here offers.

```bash
pip install "qmlkit[openqarp]"
```

```python
# docs: requires qarp
import numpy as np

import qmlkit as qk

spec = qk.hardware_efficient(6, 2).build()
thetas = np.random.default_rng(0).uniform(-np.pi, np.pi, (64, spec.n_params))
print(qk.get_backend("openqarp").expectation_over(spec, thetas, qk.Z(0)).shape)
```

Nothing else in a script changes: `backend="openqarp"` per call, `QMLKIT_BACKEND` per
session, `qk.set_default_backend("openqarp")` for the process. Python 3.11+, because
OpenQARP publishes no wheel below it.

## What a backend owes qmlkit, and what this one supplies

A backend supplies *primitives*; [`base.py`](../reference/backends.md) supplies
*semantics*. Sampling, basis rotation, qubit-wise-commuting grouping, expectation and
the whole batch stack are derived **once**, above the SDK. That is what makes agreement
between ten backends a property rather than a coincidence, and it is why most backends
here implement exactly one method.

This one implements three, because the engine can do two things the protocol otherwise
derives:

| qmlkit asks for | OpenQARP answers with | why it is worth crossing the boundary |
|---|---|---|
| `statevector(spec)` | `QarpSimulator.statevector(commands, n)` | the primitive every simulator backend supplies |
| `expectation(spec, obs)` | `QarpSimulator.expectation(commands, n, terms)` | ⟨O⟩ contracted in C++ — the `2**n` vector is never built on the Python side |
| `expectation_over_slots(spec, rows, obs)` | `QarpSimulator.batch_expectation(commands, n, terms, param_sets)` | one circuit, many parameter vectors, no return to Python between rows |
| `counts(spec, shots, seed)` | **nothing — deliberately** | shots stay with the shared estimator, so a seed reproduces the *same* counts on every backend |

The last row is the important one. It would have been easy to hand sampling to
OpenQARP as well, and the cost would have been silent: `tests/test_cross_backend.py`
asserts that `counts(spec, shots=2048, seed=42)` is identical on every backend, and a
second shot-drawing implementation cannot keep that promise.

## The translation

### Gates, one table, no transpiler

Twenty qmlkit gate names map one-to-one onto `SimpleBlock` methods — `phase` is qarp's
`p`, `i` is its `id`, and the rest carry their own names. Wires come first and the
angle last, which is the opposite of Qiskit's order and the kind of thing that is right
until it is not, so every entry is asserted against the NumPy reference rather than
argued for.

Nothing is transpiled on the way in. OpenQARP simulates this library's gate set
directly, so a transpiler pass would cost more than it could save.

### Endianness: the map, not the reversal

qarp indexes amplitudes **LSB-first** — qubit `q` carries bit `q` — which is the
opposite of qmlkit. The fix is the one the Qiskit backend uses: qmlkit qubit `i` is
emitted as qarp qubit `n−1−i` **at build time**, so the two index conventions coincide
and nothing downstream reverses anything. You can see the map in the block's own QASM:

```python
# docs: requires qarp
qc = qk.QCircuit(3)
qc.h(0).cx(0, 1).ry(2, 0.7)
block = qk.get_backend("openqarp").to_openqarp(qc.to_spec())
print(block.to_qasm2())
```

```text
// qmlkit
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];

h q[2];
cx q[2], q[1];
ry(0.7) q[0];
```

qmlkit's qubit 0 is qarp's `q[2]`, and the statevector index is then the same integer
on both sides. The alternative — simulate, then bit-reverse the vector — is both easier
to get quietly wrong and slower: `qarp.endianness.lsb_to_msb_statevector` builds its
permutation from a Python loop over `2**n` formatted strings, which costs 40 ms at 16
qubits against the 11 ms simulation it would be decorating.

The same `n−1−i` map is applied to observable terms, which is the only other place a
qubit index crosses the boundary.

### A gate no SDK has heard of

`qk.register_gate` promises that a gate you invent works everywhere. On OpenQARP it
arrives as a **matrix**: the run of named gates around it closes into one `SimpleBlock`,
the custom gate becomes a `SynthesizedUnitaryBlock` — qarp's C++ Quantum Shannon
decomposition, exact including global phase — and the two are wired together as a
`CompositeBlock`.

```python
# docs: skip
qc = qk.QCircuit(3)
qc.h(0).apply("xy", (2, 0), 0.7)          # "xy" registered with register_gate
block = qk.get_backend("openqarp").to_openqarp(qc.to_spec())

print(type(block).__name__, block.n_gates())            # CompositeBlock 25
print([type(c).__name__ for c in block.children()])     # SimpleBlock, SynthesizedUnitaryBlock
```

Two operations became twenty-five gates, which is what a Shannon decomposition of an
arbitrary two-qubit unitary costs. The subtle part is not the decomposition but the
**wire order**: a matrix carries its qubit order in its *basis* rather than in a wire
list, qmlkit writes the op's first wire as the most significant bit, and a qarp
sub-block reads its own local qubit 0 as the least significant — so the wire list is
reversed on top of the `n−1−i` map. Get that wrong and the circuit still runs, with the
operands swapped. `tests/test_cross_backend.py` pins it against the reference over
ascending, descending and non-adjacent wire orders, and over a three-qubit custom gate.

### Parameters: slots, not parameters

The sweep is where the integration gets interesting. A qmlkit circuit has `n_params`
logical parameters, but those map onto **slots** — one per (operation, parameter
position) — and a weight-tied parameter fills several. A shift rule moves one *slot* at
a time, so the batch a gradient hands the backend is a batch of slot-angle vectors, not
parameter vectors.

The translation therefore names one qarp symbol per slot, `qmlkit_slot_0`,
`qmlkit_slot_1`, …, binds them through `batch_expectation`'s parameter sets, and lets
the C++ kernel do the substitution. Naming them per *logical parameter* instead would
still build, still run, and still return numbers in range — with every occurrence of a
tied weight forced to the same shift. That is precisely the failure the slot
abstraction exists to prevent, so it is asserted where the binding happens, in
`tests/test_openqarp_backend.py`.

## Does it agree?

Speed is worth nothing if the answers move. Every number below is `max |difference|`
against the NumPy reference, over a three-layer `ry`/`rz` ring:

| qubits | statevector | ⟨O⟩ contracted | ⟨O⟩ swept | parameter-shift gradient |
|---|---|---|---|---|
| 4 | 1.7e-16 | 2.8e-17 | 6.7e-16 | 4.6e-16 |
| 6 | 1.2e-16 | 1.1e-16 | 7.8e-16 | 4.0e-16 |
| 8 | 1.1e-16 | 4.3e-16 | 9.1e-16 | 3.3e-16 |
| 10 | 8.4e-17 | 2.8e-16 | 1.2e-15 | 7.4e-16 |

That is machine precision, not a tolerance chosen to pass. The backend also runs the
whole cross-backend zoo — endianness probes, every non-parametric gate, controlled
rotations, idle qubits, basis rotations, seeded sampling, registered gates at six wire
orders — 33 cases, plus 12 of its own.

## What it buys

### The call it exists for

One `expectation_over_slots` of 256 rows, `Z0 + Z0·Z(n−1) + 0.3·X1`:

| qubits | `openqarp` | `numpy` | `aer` | vs the reference |
|---|---|---|---|---|
| 4 | 5.3 ms | **3.1 ms** | 316 ms | 0.6× |
| 6 | 11.7 ms | **10.0 ms** | 397 ms | 0.9× |
| 8 | **20.8 ms** | 48.3 ms | 523 ms | 2.3× |
| 10 | **37.3 ms** | 237 ms | 666 ms | 6.4× |
| 12 | **164 ms** | 1085 ms | 874 ms | 6.6× |
| 14 | **646 ms** | 2687 ms | 1484 ms | 4.2× |

The crossover is between six and eight qubits, and the reference wins below it — which
is the same story as Aer's crossover and for the same reason: below it, this library is
bound by per-call overhead rather than arithmetic.

### Where the win comes from

Not from the C++ kernel alone. Against this backend's *own* row-by-row path — same
engine, same circuit, binding done in Python and the block rebuilt per row — the sweep
is worth:

| qubits | swept | row by row | ratio | |
|---|---|---|---|---|
| 8 | 16.4 ms | 107 ms | 6.5× | bit-identical |
| 12 | 165 ms | 297 ms | 1.8× | bit-identical |
| 14 | 635 ms | 786 ms | 1.2× | bit-identical |

So most of the eight-qubit win is *translation* that the sweep pays once instead of 256
times, and by fourteen qubits the simulation dominates and the two converge. The two
paths agree to the last bit, because it is the same circuit either way — asserted with
`np.array_equal`, not a tolerance.

### One expectation at a time

| qubits | contracted | derived from the statevector | `numpy` | `aer` |
|---|---|---|---|---|
| 10 | **0.39 ms** | 0.54 ms | 2.05 ms | 3.14 ms |
| 12 | **0.91 ms** | 1.09 ms | 3.72 ms | 4.75 ms |
| 14 | **2.77 ms** | 3.45 ms | 9.15 ms | 10.32 ms |
| 16 | 16.96 ms | 13.92 ms | 26.94 ms | **13.39 ms** |

Contracting inside the kernel is worth about 25% up to fourteen qubits and stops paying
by sixteen, where Aer catches both. The other reason to contract there is not in the
table: nothing allocates `2**n` amplitudes on the Python side.

## Where it does not win

The NumPy reference carries a batch as a **leading axis** and contracts gate by gate,
which is a different kind of fast. Anything asking for many statevectors at once — an
adjoint gradient, a fidelity kernel — gets a whole batch in one contraction there and
one row at a time here:

| qubits | 256 statevectors, `openqarp` | `numpy` |
|---|---|---|
| 6 | 78.1 ms | **6.7 ms** |
| 8 | 104 ms | **45.3 ms** |
| 10 | **150 ms** | 212 ms |
| 12 | **284 ms** | 975 ms |

Which shows up exactly where you would expect it to, in a batched gradient of 32
samples:

| qubits | method | `openqarp` | `numpy` | |
|---|---|---|---|---|
| 8 | parameter-shift | **93 ms** | 322 ms | 3.5× |
| 8 | adjoint | 17.4 ms | **10.8 ms** | 0.6× |
| 10 | parameter-shift | **263 ms** | 1779 ms | 6.8× |
| 10 | adjoint | 37.0 ms | 33.5 ms | 0.9× |
| 12 | parameter-shift | **1260 ms** | 8589 ms | 6.8× |
| 12 | adjoint | **301 ms** | 356 ms | 1.2× |

`parameter-shift` is made of expectations and gains the whole factor; `adjoint`, which
is the default, is made of statevectors and lands where the reference does. Both return
the same gradient to `1e-9` — the probe asserts that before it times anything.

**The rule of thumb.** Reach for this backend when the answer is an expectation and the
register is wide enough to notice — eight qubits and up. Below that, and for
statevector-shaped work under about ten qubits, the reference is still the right
default. And the first `get_backend("openqarp")` pays a second of SDK import, which
matters for a script that evaluates one circuit and exits.

## Two routes to every number

Handing an observable to an SDK is a second implementation of something this library
otherwise defines once, and a second implementation is a second chance to be wrong. So
the backend keeps both:

```python
# docs: requires qarp
native = qk.get_backend("openqarp")                              # contracted in C++
derived = qk.get_backend("openqarp", native_expectations=False)  # from the statevector

bound = spec.bind(thetas[0])
obs = qk.Z(0) + 0.5 * qk.X(1)
print(abs(native.expectation(bound, obs) - derived.expectation(bound, obs)) < 1e-12)
```

`native_expectations=False` is not a fallback for when something breaks. It is how the
suite asks one backend the same question by two routes that share the circuit
translation and no arithmetic, and compares the answers — and it is there for you to do
the same when a result surprises you. `qk.selfcheck` reaches for it too, since it runs
every installed backend against the reference.

## The block is yours

`to_openqarp` hands back a real qarp block, not an internal handle, so OpenQARP's own
tooling — its plotting, its exporters, its algorithm library — is one call away. And
because qarp writes little-endian QASM, the same convention Qiskit's exporter uses,
`from_qasm` inverts the build-time map and the circuit comes back as itself:

```python
# docs: requires qarp
back = qk.from_qasm(block.to_qasm2())
print(np.allclose(qk.statevector(qc.to_spec()), qk.statevector(back)))
```

## Reproduce it

Every number on this page comes out of one script, on one machine (Windows, Python
3.14, OpenQARP 0.1.0). The ratios travel; the milliseconds do not.

```bash
python scripts/probe_openqarp.py
```

It measures agreement first and speed second, and it refuses to report a timing for a
route whose gradient does not already match the reference.
