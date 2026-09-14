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

**The trap worth knowing, and it has two halves.** A molecular Hamiltonian conserves
particle number, so any operator that does not has *exactly zero* gradient at
Hartree–Fock — the generic pool grows an empty circuit and reports convergence. Use
`chemistry_operator_pool`. This is physics, not a bug, and a test pins it.

The second half is that the pool alone is not enough. A particle-conserving pool cannot
*change* the particle number either, so starting from the default vacuum every
candidate has zero gradient too, and you get the same empty circuit — reporting
`+0.720` Ha against a true `-1.137`. Hartree–Fock is where `reference=` comes in:

```python
import qmlkit as qk
from qmlkit.algorithms import AdaptVQE, chemistry_operator_pool, h2_hamiltonian

hamiltonian, info = h2_hamiltonian()          # H2 at 0.735 A, 4 qubits
result = AdaptVQE(
    hamiltonian,
    n_qubits=4,
    pool=chemistry_operator_pool(4),
    reference=[0, 1],                          # Hartree-Fock: two occupied spin-orbitals
).run(seed=0)

print(f"{result.energy:.8f} Ha from {len(result.operators)} operator(s)")
# -1.13730603 Ha from 1 operator(s)
```

One operator reaches the exact ground state here, to `1e-11`. That is ADAPT's point:
it grows only what the gradient asks for. Selecting *zero* operators now warns, because
it is never convergence — it means nothing in the pool can move the energy off the
reference at all.

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

Variational policies for reinforcement learning. The circuit *is* the policy: an
observation is encoded, the trainable block runs, and one expectation per action
becomes a logit that a softmax turns into action probabilities.

```python
from qmlkit.algorithms import ContextualBandit, QuantumPolicy, train_reinforce

env = ContextualBandit(seed=0)                 # 2 observations, 2 actions, reward 0 or 1
policy = QuantumPolicy(n_observations=env.n_observations, n_actions=env.n_actions, seed=0)
result = train_reinforce(policy, env, n_episodes=200, seed=0)

print(f"{result.mean_return(50):.2f}")         # mean return over the last 50 episodes
# 0.70
```

Read that against the bar, not in isolation: two actions and a 0/1 reward means a random
policy averages **0.50**, and the first fifty episodes of this run do exactly that. The
last fifty average **0.70**. That gap is the learning, and it is the number to report —
a mean return quoted without the random baseline beside it says nothing at all.

`beta` is the knob most worth understanding. Expectations live in `[-1, 1]`, which is a
narrow range to softmax over, so `beta` sets how sharp the policy is: too low and it
never commits, too high and it stops exploring before it has learned anything.

Any environment satisfying the `Environment` protocol works — `n_observations`,
`n_actions`, `reset()` and `step(action)`. `ContextualBandit` is the toy that keeps the
test suite honest, not the interesting case.

::: qmlkit.algorithms.rl
