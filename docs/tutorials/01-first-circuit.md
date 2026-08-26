# 1. Circuits are data

Most quantum SDKs give you a circuit *object* that belongs to a particular simulator.
qmlkit gives you a `CircuitSpec`: an immutable description of operations and parameter
slots that belongs to nobody. Backends compile it; gradients, resource counting and
drawing all read it.

That is not a stylistic preference. It is the reason one gradient implementation
serves five backends, and the reason the later tutorials work at all.

## Build one

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

print(qk.draw(bell))
```

```text
q0: ─H──@──
q1: ────X──
```

`QCircuit` is a builder — mutable, chainable, convenient. `to_spec()` freezes it into
the immutable thing everything else consumes.

## Run it three ways

Exactly, as a statevector:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

print(qk.statevector(bell))
print(qk.probabilities(bell))
```

```text
[0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
[0.5 0.  0.  0.5]
```

Or by sampling, which is opt-in:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

print(qk.run_counts(bell, shots=1000, seed=0))
```

```text
{'00': 521, '11': 479}
```

!!! note "Qubit 0 is the most significant bit"
    `'01'` means qubit 0 is `|0⟩` and qubit 1 is `|1⟩`. This is big-endian, matching
    SpinQit and PennyLane. Qiskit is little-endian, and the Qiskit backend handles the
    reversal at build time so indices mean the same thing everywhere — see
    [Backends and conventions](../guides/backends.md).

## Inspect it

A spec knows its own cost, because it is just data:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

print(bell)
print(f"qubits {bell.n_qubits} · depth {bell.depth()} · gates {bell.gate_counts()}")
```

```text
CircuitSpec(n_qubits=2, n_ops=2, n_params=0, depth=2)
qubits 2 · depth 2 · gates {'cx': 1, 'h': 1}
```

## Parameters, and the slot idea

A parameter is a *reference*, not a number. `ParamRef(0)` means "whatever ends up at
index 0 of the parameter vector".

```python
import numpy as np
import qmlkit as qk

qc = qk.QCircuit(1)
qc.ry(0, qk.ParamRef(0))
spec = qc.to_spec()

print(f"parameters: {spec.n_params}")
print(f"<Z> at θ=0.7: {qk.expval(spec, qk.Z(0), theta=[0.7]):.15f}")
print(f"cos(0.7):     {np.cos(0.7):.15f}")
```

```text
parameters: 1
<Z> at θ=0.7: 0.764842187284488
cos(0.7):     0.764842187284489
```

The piece that matters later is the **slot**. A slot is one angle site — one
(operation, parameter position) pair. Several slots may point at the *same* logical
parameter, which is how weight tying is expressed:

```python
import qmlkit as qk

a = qk.Ansatz(1, qk.share(3, qk.RotationLayer("ry")))
spec = a.build()

print(f"logical parameters: {spec.n_params}")
print(f"slots:              {len(spec.slots())}")
print(f"occurrences of θ0:  {len(spec.occurrences_of(0))}")
```

```text
logical parameters: 1
slots:              3
occurrences of θ0:  3
```

Three `Ry(θ₀)` on one qubit is `Ry(3θ₀)`, so the derivative is three times what a
single rotation would give. Keeping slots and parameters distinct is what lets the
gradient code get that right — [tutorial 3](03-gradients.md) shows what happens when a
library conflates them.

## Observables

Observables are Pauli sums, built with `+` and `*`:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

observable = qk.Z(0) + 0.5 * qk.ZZ(0, 1) + 0.3 * qk.X(1)
print(observable)
print(f"<Z0>    = {qk.expval(bell, qk.Z(0)):.6f}")
print(f"<Z0 Z1> = {qk.expval(bell, qk.ZZ(0, 1)):.6f}")
print(f"<O>     = {qk.expval(bell, observable):.6f}")
```

```text
Z0 + 0.5*Z0 Z1 + 0.3*X1
<Z0>    = 0.000000
<Z0 Z1> = 1.000000
<O>     = 0.500000
```

`<Z0 Z1> = 1` for a Bell state — the qubits always agree — while `<Z0> = 0`, because
each on its own is a fair coin. That is entanglement in two numbers.

## Shots are opt-in, and come with an error bar

`shots=None` is the default and returns the exact value. When you do sample, ask for
the standard error too — a sampled number without one is not a measurement, it is a
guess:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
bell = qc.to_spec()

for shots in (100, 10_000, 1_000_000):
    value, err = qk.expectation(bell, qk.Z(0), shots=shots, seed=0, return_std=True)
    print(f"{shots:>9,} shots: {value:+.5f} ± {err:.5f}")

print(f"exact:            {qk.expval(bell, qk.Z(0)):+.5f}")
print(f"shots for ±0.01:  {qk.shots_for_precision(0.01):,}")
```

```text
      100 shots: +0.02000 ± 0.09998
   10,000 shots: +0.01240 ± 0.01000
1,000,000 shots: +0.00126 ± 0.00100
exact:            +0.00000
shots for ±0.01:  10,000
```

The error falls as `1/√N`: a hundred times more shots buys ten times the precision.
That is the whole economics of measurement, and it is why the [gradient
tutorial](03-gradients.md) cares so much about how many circuits a method needs.

## Composing

Specs compose and invert, which is all a fidelity kernel really needs:

```python
import qmlkit as qk

first = qk.QCircuit(1)
first.ry(0, 0.4)
second = qk.QCircuit(1)
second.rz(0, 1.1)

both = first.to_spec().compose(second.to_spec())
identity = both.compose(both.adjoint())

print(qk.draw(both))
print(f"U U† is the identity: {qk.expval(identity, qk.Z(0)):.12f}")
```

```text
q0: ─RY(0.40)──RZ(1.10)──
U U† is the identity: 1.000000000000
```

---

**Next:** [Getting data in](02-encoding-data.md) — the choice that decides what your
model can represent, before any training happens.
