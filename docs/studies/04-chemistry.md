# Study 4 — H₂, and an ansatz that converges to the wrong energy

Chemistry is the one place quantum computing has a target it can be scored against
exactly: diagonalise the Hamiltonian and compare. That makes it the best available
test of whether a variational method is working, and the worst place to be
approximately right without knowing it.

Chemical accuracy is 1 kcal/mol — **1.594 mHa**. Anything outside that is not a
chemistry result.

## The molecule

```python
import qmlkit as qk
from qmlkit.algorithms import VQE, exact_ground_energy, h2_hamiltonian

hamiltonian, info = h2_hamiltonian()
exact = exact_ground_energy(hamiltonian, 4)
print(f"{info['n_terms']} Pauli terms on {info['n_qubits']} qubits")
print(f"exact ground state {exact:.6f} Ha")
```

The Hartree–Fock reference is the occupation `[1, 1, 0, 0]` — the two electrons in the
two lowest spin-orbitals. An `Ansatz` starts from `|0000>`, so that state has to be
prepared first:

```python
occupation = info["hartree_fock_occupation"]

def hartree_fock(inner):
    """Prepend the HF occupation to any ansatz."""
    def build(circuit, context):
        for wire, occupied in enumerate(occupation):
            if occupied:
                circuit.x(wire)
        inner.block.emit(circuit, context)
    return qk.Ansatz(4, qk.Custom(build, "hf"), f"hf_{inner.name}")

reference = qk.QCircuit(4)
reference.x(0)
reference.x(1)
print(f"Hartree-Fock energy {qk.expectation(reference.to_spec(), hamiltonian):.6f} Ha")
```

That prints `-1.116999`. So HF already has most of the answer, and the whole job of
the VQE is the **20.3 mHa of correlation energy** between it and the exact value. That
framing matters: a method that returns something near `-1.1` has not necessarily done
anything at all.

## The failure

```python
shallow = hartree_fock(qk.hardware_efficient(4, 2))
result = VQE(hamiltonian, ansatz=shallow).run(seed=0)
print(f"VQE {result.energy:.6f} Ha   error {abs(result.energy - exact) * 1000:.1f} mHa")
```

`-0.536370 Ha`, which is **601 mHa** out — off by thirty times the entire correlation
energy, and *worse than doing nothing*, since Hartree–Fock alone was 20 mHa away.

Nothing raised. The optimiser converged. `-0.53` is a plausible-looking number in
Hartree, and it is the energy of the `|1000>` state — the circuit found a comfortable
minimum in the wrong particle-number sector and settled there.

## What the library says about it

Depth is the problem, and it is visible before the energy is:

```python
for layers in (2, 4):
    ansatz = hartree_fock(qk.hardware_efficient(4, layers))
    energy = VQE(hamiltonian, ansatz=ansatz).run(seed=0).energy
    error = abs(energy - exact) * 1000
    verdict = "chemical accuracy" if error < 1.594 else "NOT chemical accuracy"
    print(f"  {layers} layers, {ansatz.n_params:2d} params: {energy:+.6f} Ha"
          f"  ({error:7.2f} mHa)  {verdict}")
```

Four layers reaches `-1.137306` — **0.00 mHa**, exact to the printed precision. Two
layers cannot represent the state at all, and says so only through an energy you have
to already know the answer to recognise as wrong.

`strongly_entangling(4, 3)` also reaches it, so this is about expressive capacity
rather than about one particular template.

## Two traps this study sits on top of

**Rotosolve is not always valid.** `VQE` defaults to it, and its three-point fit
assumes the loss is a single sinusoid in each angle. That fails when one angle drives
several gates that do not compose — QAOA's cost angle drives one `rz` per edge, five
frequencies — and it then converges instantly to the wrong point and reports it. Check
before trusting it:

```python
from qmlkit.optim import supports_rotosolve

print(supports_rotosolve(hartree_fock(qk.hardware_efficient(4, 4)).build()))
```

**Particle number is conserved.** A molecular Hamiltonian commutes with the number
operator, so any excitation that does not conserve it has *exactly zero* gradient at
Hartree–Fock. An ADAPT run with a generic operator pool grows an empty circuit and
reports convergence. Use `chemistry_operator_pool`; this is physics, not a bug, and a
test pins it.

## The verdict

The quantum method wins this one outright — `-1.137306` against an exact
`-1.137306` — which is worth stating plainly given how the other studies end. H₂ in a
minimal basis is four qubits and sixteen amplitudes, so a laptop diagonalises it
instantly and no advantage is claimed. What the study demonstrates is narrower and
more useful: **the same code, one layer count apart, produces an answer that is exact
and an answer that is 601 mHa wrong, and only one of them looks different from the
outside.**
