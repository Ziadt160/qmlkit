# Algorithms

Variational algorithms built on the same IR, ansatz vocabulary and gradients as
everything else — so an ansatz you registered, a gate you defined, or a backend you
wrote works in all of them without any of them knowing about it.

Every one of these takes its ansatz, feature map or operator pool as an argument and
must actually use it. `tests/test_injection.py` injects two of different sizes and
asserts the parameter count follows, because a constructor that accepts `ansatz=` and
silently ignores it looks identical from the outside.

## `qmlkit.algorithms.vqe`

Ground-state energy by variational minimisation. Worked end to end in
[study 4](../studies/04-chemistry.md), including the case where a too-shallow ansatz
converges confidently to an energy 601 mHa wrong.

::: qmlkit.algorithms.vqe

## `qmlkit.algorithms.adapt`

ADAPT-VQE: grow the ansatz one operator at a time, chosen by gradient magnitude.

**The trap worth knowing.** A molecular Hamiltonian conserves particle number, so any
operator that does not has *exactly zero* gradient at Hartree–Fock — the generic pool
grows an empty circuit and reports convergence. Use `chemistry_operator_pool`. This is
physics, not a bug, and a test pins it.

::: qmlkit.algorithms.adapt

## `qmlkit.algorithms.qaoa`

Quantum approximate optimisation.

**Rotosolve is not valid here.** QAOA's cost angle drives one `rz` per edge, and those
do not compose into a single sinusoid — measured: five frequencies. Rotosolve's
three-point fit then converges instantly to the wrong point and reports it as a result.
Check with `qmlkit.optim.supports_rotosolve` before trusting it.

::: qmlkit.algorithms.qaoa

## `qmlkit.algorithms.molecule` and `qmlkit.algorithms.chemistry`

Molecular Hamiltonians. Two routes, deliberately: `from_integrals` is the general one
and takes PySCF or OpenFermion output for any molecule, while the built-in SCF handles
s-orbital elements only. qmlkit is not a quantum chemistry package and does not try to
become one.

::: qmlkit.algorithms.molecule

::: qmlkit.algorithms.chemistry

## `qmlkit.algorithms.hamiltonians`

Standard model Hamiltonians — Ising, Heisenberg, and the rest — as `PauliSum`s.

::: qmlkit.algorithms.hamiltonians

## `qmlkit.algorithms.autoencoder`

Quantum autoencoders: compress a state onto fewer qubits and measure what the discarded
"trash" qubits retain.

::: qmlkit.algorithms.autoencoder

## `qmlkit.algorithms.clustering`

`QMeans`. Scored with `qmlkit.evaluate.clustering`, which reports internal *and*
external quality because they routinely disagree — see
[study 5](../studies/05-beyond-classification.md).

::: qmlkit.algorithms.clustering

## `qmlkit.algorithms.rl`

Variational policies for reinforcement learning.

::: qmlkit.algorithms.rl
