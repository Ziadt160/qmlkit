# 3. Gradients

Six methods, one function. They differ in cost, in whether they are exact, and in
whether they could run on real hardware — not in the answer.

```python
import numpy as np
import qmlkit as qk

ansatz = qk.hardware_efficient(3, n_layers=2)
spec, theta = ansatz.build(), ansatz.init(seed=0)
obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
reference = qk.grad(spec, theta, obs, method="adjoint")

for method in qk.list_gradient_methods():
    kwargs = {"seed": 0, "n_avg": 50} if method == "spsa" else {}
    g = qk.grad(spec, theta, obs, method=method, **kwargs)
    cost = qk.gradient_cost(spec, method)
    print(f"{method:<17}{str(cost):>4} circuits   max error {np.abs(g - reference).max():.2e}")
```

```text
adjoint             1 circuits   max error 0.00e+00
backprop            1 circuits   max error 6.94e-17
finite-diff        24 circuits   max error 5.14e-10
hadamard           12 circuits   max error 1.11e-16
parameter-shift    24 circuits   max error 7.67e-16
spsa                2 circuits   max error 4.29e-02
```

`qk.grad(spec, theta, obs)` with no `method` picks for you: adjoint when every gate
has a closed-form derivative and the backend can produce a statevector,
parameter-shift otherwise. Asking for `shots` forces parameter-shift, because
sampling rules out reading the statevector by definition.

Which to reach for is its own page: [Choosing a gradient
method](../guides/choosing-a-gradient.md).

## Two ways to get a plausible wrong number

Parameter-shift is the method everyone implements themselves, and there are two
places where a natural implementation returns a smooth, believable, wrong answer.
Neither raises. Both are why this library derives shift rules instead of
transcribing one.

### The rule is per gate, not per library

The famous two-term rule `[E(θ+π/2) − E(θ−π/2)] / 2` is correct for a gate whose
generator has a single frequency. It is *not* correct for a controlled rotation.

```python
import qmlkit as qk

for gate in ("ry", "rz", "crz", "phase"):
    rule = qk.rule_for_gate(gate)
    print(f"{gate:<7} frequencies {qk.get_gate(gate).frequencies}  ->  {len(rule.shifts)}-term rule")
```

```text
ry      frequencies (1.0,)  ->  2-term rule
rz      frequencies (1.0,)  ->  2-term rule
crz     frequencies (0.5, 1.0)  ->  4-term rule
phase   frequencies (1.0,)  ->  2-term rule
```

Here is a circuit where it bites. `H⊗H` then `CRZ(θ)`, measuring `X₀X₁`, gives
exactly `cos(θ/2)` — pure frequency ½:

```python
import numpy as np
import qmlkit as qk

qc = qk.QCircuit(2)
qc.h(0).h(1).crz(0, 1, qk.ParamRef(0))
spec = qc.to_spec()
XX = qk.PauliString(((0, "X"), (1, "X")), 1.0)

theta = 1.1
correct = qk.grad(spec, np.array([theta]), XX, method="parameter-shift")[0]
naive = (
    qk.expval(spec, XX, theta=[theta + np.pi / 2])
    - qk.expval(spec, XX, theta=[theta - np.pi / 2])
) / 2

print(f"E(θ)            = cos(θ/2), checked: {qk.expval(spec, XX, theta=[theta]):.10f} vs {np.cos(theta / 2):.10f}")
print(f"correct  (4-term) {correct:+.10f}   = -sin(θ/2)/2")
print(f"naive    (2-term) {naive:+.10f}   = -sin(θ/2)/√2")
print(f"ratio             {naive / correct:.10f}   (that is √2)")
```

```text
E(θ)            = cos(θ/2), checked: 0.8525245221 vs 0.8525245221
correct  (4-term) -0.2613436145   = -sin(θ/2)/2
naive    (2-term) -0.3695956840   = -sin(θ/2)/√2
ratio             1.4142135624   (that is √2)
```

The naive answer is exactly **√2 times** the right one. Same sign, same shape, so
training still descends — with a step size that is silently wrong by 41%. Nothing
about the loss curve would tell you.

!!! note "Why measuring the target qubit hides it"
    Swap `X₀X₁` for `Z₁` and both rules agree perfectly, because `Z` commutes with
    `CRZ` and the expectation does not depend on `θ` at all. A test built on that
    circuit passes against a broken implementation. This is not hypothetical — an
    early version of this library's own test suite made exactly that mistake.

### Tied parameters shift one at a time

When one logical parameter fills several slots, its derivative is the **sum** over
occurrences. Shifting them all together computes something else entirely:

```python
import numpy as np
import qmlkit as qk

ansatz = qk.Ansatz(1, qk.share(3, qk.RotationLayer("ry")))
spec, theta = ansatz.build(), np.array([0.4])

correct = qk.grad(spec, theta, qk.Z(0), method="parameter-shift")[0]
naive = (
    qk.expval(spec, qk.Z(0), theta=theta + np.pi / 2)
    - qk.expval(spec, qk.Z(0), theta=theta - np.pi / 2)
) / 2

print(f"three tied Ry(θ) on one qubit is Ry(3θ), so E(θ) = cos(3θ)")
print(f"analytic  -3·sin(1.2) = {-3 * np.sin(1.2):+.10f}")
print(f"correct               = {correct:+.10f}")
print(f"naive (shifted as one)= {naive:+.10f}")
```

```text
three tied Ry(θ) on one qubit is Ry(3θ), so E(θ) = cos(3θ)
analytic  -3·sin(1.2) = -2.7961172579
correct               = -2.7961172579
naive (shifted as one)= +0.9320390860
```

Wrong by a factor of −3 here, and it is the case that matters for QCNNs and any
convolutional ansatz, where weight tying is the whole point.

## Cost, and why adjoint is the default

Parameter-shift costs `2P` circuits — more when a gate needs a four-term rule.
Adjoint costs one pass regardless of `P`:

```python
import qmlkit as qk

for n_layers in (2, 6, 12):
    a = qk.hardware_efficient(5, n_layers)
    spec = a.build()
    print(
        f"P={a.n_params:>3}   adjoint 1 pass   "
        f"hadamard {qk.gradient_cost(spec, 'hadamard'):>4}   "
        f"parameter-shift {qk.gradient_cost(spec, 'parameter-shift'):>4}"
    )
```

```text
P= 20   adjoint 1 pass   hadamard   20   parameter-shift   40
P= 60   adjoint 1 pass   hadamard   60   parameter-shift  120
P=120   adjoint 1 pass   hadamard  120   parameter-shift  240
```

Measured on a 5-qubit ansatz with a two-term observable, that is **12.6 ms for
adjoint against 823 ms for parameter-shift** at `P=120`. Adjoint is not more accurate
— both are exact — it is just cheaper on a simulator, where reading the statevector
is allowed.

`hadamard` sits between them: one circuit per parameter instead of two, using an
ancilla and controlled generators, and unlike adjoint it is a real measurement, so it
stays valid on hardware.

## Sampling

Ask for `shots` and you get parameter-shift with shot noise, which is what a device
would give you:

```python
import numpy as np
import qmlkit as qk

ansatz = qk.hardware_efficient(3, n_layers=2)
spec, theta = ansatz.build(), ansatz.init(seed=0)
obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)

exact = qk.grad(spec, theta, obs, method="parameter-shift")
sampled = qk.grad(spec, theta, obs, method="parameter-shift", shots=4096, seed=0)
print(f"4096 shots per circuit: max deviation {np.abs(sampled - exact).max():.4f}")
```

```text
4096 shots per circuit: max deviation 0.0211
```

## Second derivatives

`hessian` differences the *exact* gradient, so only the outer derivative is
approximate:

```python
import numpy as np
import qmlkit as qk

qc = qk.QCircuit(1)
qc.ry(0, qk.ParamRef(0))

h = qk.hessian(qc.to_spec(), np.array([0.8]), qk.Z(0))
print(f"hessian    {h[0, 0]:+.8f}")
print(f"-cos(0.8)  {-np.cos(0.8):+.8f}")
```

```text
hessian    -0.69670671
-cos(0.8)  -0.69670671
```

## Bringing your own

The gradient dispatcher is a registry, so a new estimator becomes a keyword
everywhere the library takes `method=`:

```python
import numpy as np
import qmlkit as qk

@qk.register_gradient("central_diff_demo")
def central_diff(spec, theta, obs, *, backend=None, shots=None, eps=1e-5, **kw):
    out = np.zeros(spec.n_params)
    for k in range(spec.n_params):
        step = np.zeros_like(theta)
        step[k] = eps
        plus = qk.expval(spec, obs, theta=theta + step, backend=backend, shots=shots)
        minus = qk.expval(spec, obs, theta=theta - step, backend=backend, shots=shots)
        out[k] = (plus - minus) / (2 * eps)
    return out

ansatz = qk.hardware_efficient(2, n_layers=1)
spec, theta = ansatz.build(), ansatz.init(seed=0)
mine = qk.grad(spec, theta, qk.Z(0), method="central_diff_demo")
exact = qk.grad(spec, theta, qk.Z(0), method="adjoint")
print(f"agrees with adjoint to {np.abs(mine - exact).max():.2e}")
```

---

**Next:** [Designing an ansatz](04-ansatz-design.md) — one line, and it inherits all
of the above for free.
