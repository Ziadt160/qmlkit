# Running under noise

qmlkit is simulator-only, but a simulator does not have to be a *perfect* one. Two
backends evolve a density matrix and take a noise model, so you can ask what a
circuit does on a device that makes mistakes.

| Backend | Simulator | Noise model | Install |
|---|---|---|---|
| `cirq-density` | `cirq.DensityMatrixSimulator` | any `cirq` channel or `cirq.NoiseModel` | `pip install 'qmlkit[cirq]'` |
| `qiskit-aer` | `AerSimulator(method="density_matrix")` | `qiskit_aer.noise.NoiseModel`, including `NoiseModel.from_backend(...)` | `pip install 'qmlkit[aer]'` |

`qiskit-aer` is a separate distribution from `qiskit`: installing the Qiskit backend
does not give you this one. SpinQit has no noisy simulator — its noisy path is the
real NMR hardware, which `0.x` does not target.

## You always name the backend

Noise never selects a simulator for you.

```python
# docs: requires cirq
import cirq
import qmlkit as qk

backend = qk.get_backend("cirq-density", noise=cirq.depolarize(0.01))
```

Asking for noise without naming a mixed-state backend is an error, not a guess:

```python
# docs: requires cirq
try:
    qk.get_backend(noise=cirq.depolarize(0.01))
except ValueError as exc:
    print(str(exc).splitlines()[0])
# noise was given, but the default backend evolves a pure state and cannot carry it.
```

This is deliberate. A mixed-state run costs more, refuses two of the gradient
methods, and answers a *different question* — so which simulator produced a number
should be visible in the code that produced it, not inferred from a global default.

## Noise and shot noise are separate, and stay separate

`shots=None` on a noisy backend is not a contradiction. The density matrix is
evolved exactly; what is exact is the answer *given the noise model*.

```python
# docs: requires cirq
import numpy as np

spec = qk.angle_encode([0.7])
noisy = qk.get_backend("cirq-density", noise=cirq.depolarize(0.05))

exact_pure = qk.expectation(spec, qk.Z(0), backend="numpy")
exact_noisy = qk.expectation(spec, qk.Z(0), backend=noisy)
sampled = qk.expectation(spec, qk.Z(0), backend=noisy, shots=10_000, seed=0)

print(f"no noise, no shots  {exact_pure:+.4f}")   # cos(0.7)
print(f"noise, no shots     {exact_noisy:+.4f}")  # cos(0.7) * (1 - 4p/3)
print(f"noise and shots     {sampled:+.4f}")
```

Being able to turn one off is the point. Decoherence and sampling error both pull a
number around, and studying either one with the other layered on top means never
knowing which you are looking at. Ask for `shots=N` when you want both — that is
what a device would give you.

With no noise model at all, these backends reproduce the pure-state ones to machine
precision. That is not a trivial case; it is what makes the noisy numbers
trustworthy, and `tests/test_noisy_backends.py` asserts it across circuits,
observables and both SDKs.

## What is refused

There is no statevector, so the two gradient methods that differentiate one decline:

```python
# docs: requires cirq
ansatz = qk.hardware_efficient(2, 2)
theta = np.linspace(0.1, 1.2, ansatz.n_params)

try:
    qk.grad(ansatz.build(), theta, qk.Z(0), method="adjoint", backend=noisy)
except ValueError as exc:
    print(str(exc)[:60])
```

Refusing is the whole point. `adjoint` and `backprop` would have quietly
differentiated a *noiseless* circuit and handed back a machine-precision gradient to
someone who asked about a noisy one — a number that is wrong in a way no assertion
would catch.

`parameter-shift` works, because a shift rule never inspects a state — it evaluates
the same circuit at shifted angles, which is exactly what a device does:

```python
# docs: requires cirq
g = qk.grad(ansatz.build(), theta, qk.Z(0), method="parameter-shift", backend=noisy)
print(g.shape)
```

`qk.grad_batch` routes through the same path, so a batched training step works under
noise too.

## What noise does to trainability

This is the number worth knowing before you spend a week on an experiment. A
3-qubit, 3-layer hardware-efficient ansatz, gradient norm averaged over 8 random
parameter vectors, against depolarizing strength:

| `p` | mean ‖∇‖ | vs noiseless | purity |
|---|---|---|---|
| 0 | 0.9145 | 1.000 | 1.000 |
| 0.005 | 0.8257 | 0.903 | 0.753 |
| 0.01 | 0.7425 | 0.812 | 0.585 |
| 0.02 | 0.5052 | 0.552 | 0.334 |
| 0.05 | 0.3243 | 0.355 | 0.151 |
| 0.1 | 0.1270 | 0.139 | 0.126 |

The gradient direction survives — it correlates above 0.99 with the noiseless one at
`p = 0.05` — but the *magnitude* collapses. That is the failure mode: the cost
landscape flattens toward the maximally mixed state, and it flattens faster with
depth. Past some point the gradient is smaller than the standard error of the shot
budget you can afford, and training stops working for a reason no amount of tuning
fixes.

`qk.plan()` will tell you the standard error your shot budget buys. Comparing it to
the gradient norm above is the arithmetic that decides whether an experiment is
feasible at all.

```python
# docs: requires cirq
print(f"purity under 5% depolarizing: {noisy.purity(ansatz.build(theta)):.4f}")
```

`purity()` is the cheapest single number that says how much the noise model actually
did. `1.0` is a pure state; `1/2**n` is maximally mixed.

## Traps

**A fidelity kernel loses its unit diagonal.** `k(x, x) = 1` is a fact about a
noiseless compute-uncompute circuit. Under noise the circuit does not return to
`|0⟩`, so the diagonal drops below one — and anything that fills the diagonal in by
assumption is writing down a number it did not measure.

```python
# docs: requires cirq
x = np.array([0.4, 0.5])
kernel = qk.QuantumKernel(qk.ZZFeatureMap(2), backend=noisy)
print(f"k(x, x) under noise: {float(np.atleast_2d(kernel(x, x))[0, 0]):.4f}")  # < 1
```

**The diagnostic thresholds were calibrated on exact gradients.** `qk.diagnose()`
reports `FLAT_GRADIENTS` against a cutoff chosen for shot-free, noiseless runs. Under
noise, gradients are genuinely smaller, so that finding fires more readily and
carries less information than it does on an exact backend. Read it as "the gradient
is small here", which is true, rather than "the ansatz is badly designed", which may
not be.

**Cirq applies a bare channel after every moment.** `cirq.depolarize(p)` passed as a
noise model becomes a `ConstantQubitNoiseModel` — every qubit, every moment,
including idle ones. That is a reasonable first model and a poor imitation of a real
device. Pass a `cirq.NoiseModel`, or lift one off hardware with Qiskit's
`NoiseModel.from_backend(...)`, when the answer needs to mean something.

## What qmlkit does not do

**Error mitigation.** No ZNE, PEC or CDR here. [Mitiq](https://mitiq.readthedocs.io)
is backend-agnostic and already good at this, and a qmlkit backend is an executor
function away from it. Reimplementing it would be a worse copy of a solved problem.

**Error correction.** Simulating a code with a QML circuit inside it is not a
library gap — an encoded VQC is non-Clifford, so the stabilizer simulators that make
QEC tractable at scale do not apply, and dense simulation of the encoded register is
out of reach. [Stim](https://github.com/quantumlib/Stim) is the tool for codes.

What is missing and worth having is the honest comparison: whether mitigation
actually *improved* an estimate, or only traded bias for variance, measured on
identical seeds with the shot cost stated. That is the shape of `qk.baseline`, and
it is where this will go next.
