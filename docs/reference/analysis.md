# Analysis

Measuring an ansatz rather than asserting things about it: expressibility, entanglement, spectra, geometry.

## `qmlkit.diagnostics`

::: qmlkit.diagnostics

## `qmlkit.metrics`

Two halves. Expressibility, entangling capability and gradient variance describe the
*circuit*, and [Designing an ansatz](../tutorials/04-ansatz-design.md) and
[Trainability](../tutorials/08-trainability.md) work through them.

The other half describes what the model can be expected to *learn*, and is easy to
miss because nothing else links to it:

| | |
|---|---|
| `generalization_bound(T, N)` | the expected train-test gap, `O(sqrt(T log T / N))` (Caro et al. 2022), for `T` trainable gates and `N` samples |
| `samples_for_gap(T, gap)` | the same bound inverted: how much data a target gap needs |
| `fisher_information` / `effective_dimension` | how many parameters are *usefully* independent, which is generally far fewer than the raw count |
| `noise_survival(f, depth)` | `f**depth`, independent per-gate success compounding over a circuit |

These are worth running before a long training job rather than after.
`hardware_efficient(4, 3)` carries 24 trainable parameters; against 100 samples the
bound is **0.87**, which for a metric in `[0, 1]` is no constraint at all. Reaching a
gap of 0.1 would take **7,628 samples**. Go to `hardware_efficient(6, 8)` — 96
parameters — and the bound is **2.09** and the requirement **43,818 samples**. Quantum
datasets are rarely that large, and that arithmetic is usually the real reason a model
that trains beautifully does not generalise.

`noise_survival` is the same kind of sobriety on the hardware side: a 200-gate circuit
at 99.9% per-gate fidelity keeps **0.819** of its amplitude, and a 2,000-gate one keeps
0.135.

::: qmlkit.metrics

## `qmlkit.landscape`

Why a model is not training: gradients over every parameter, which of the four
barren-plateau mechanisms is responsible, curvature at a stationary point, whether
random starts agree, and where the QFIM rank saturates.

::: qmlkit.landscape

## `qmlkit.fourier`

::: qmlkit.fourier

## `qmlkit.info`

::: qmlkit.info

## `qmlkit.optim`

::: qmlkit.optim

## `qmlkit.datasets`

::: qmlkit.datasets

## `qmlkit.draw`

::: qmlkit.draw

## `qmlkit.generative`

Two families, and the difference decides what you can ask of them. A **Born machine**
(`QCBM`, `QGAN`) reads its distribution off measurement probabilities directly, so it
samples easily and cannot score an arbitrary point. An **energy-based** model
(`QuantumBoltzmannMachine`, `QuantumHopfield`) defines a distribution through an energy
and a partition function, so it scores easily and samples only through a chain.

`QCBM` trains against a target distribution through a sample-based loss —
`mmd_squared`, `kl_divergence` or `total_variation`. `mmd_squared` is the usual choice:
the other two need the target's support to cover the model's, which an untrained model
rarely arranges.

`QuantumBoltzmannMachine.grad` returns a **lower-bound** gradient, not an exact
log-likelihood gradient — the exact one needs the quantum relative entropy's derivative,
which is not sampled cheaply. The bound is what makes the model trainable at all; it
also means the loss it reports is a bound rather than a likelihood, and a run that
stops improving may have hit the bound rather than the optimum.

::: qmlkit.generative

## `qmlkit.shadows`

Classical shadows: many observables from few measurements.

The trade is worth making explicit, because it is the opposite of the usual one.
Measuring `M` observables the direct way costs one measurement setting each, and each
setting needs enough shots to resolve its own value. A shadow takes randomised
single-qubit measurements *once* and predicts all `M` afterwards, at a cost that grows
with `log M` rather than `M` — but with a variance penalty that grows with each
observable's locality, sharply. Shadows win on many *local* observables and lose on a
few global ones.

`shadow_shot_cost` prices the shadow route; `qmlkit.kernels.kernel_shot_cost` prices
the direct one. Compare them before committing a device run, because the crossover
moves with locality and there is no rule of thumb that survives it.

::: qmlkit.shadows

## `qmlkit.utils.shots`

::: qmlkit.utils.shots
