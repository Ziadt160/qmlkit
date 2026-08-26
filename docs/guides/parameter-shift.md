# The parameter-shift rule

Every QML course teaches this formula:

$$\frac{\partial E}{\partial \theta} = \frac{E(\theta + \pi/2) - E(\theta - \pi/2)}{2}$$

It is correct — for a gate whose generator has a single frequency, differentiated one
occurrence at a time. Both of those conditions are easy to violate without noticing,
and violating either returns a smooth, finite, believable number rather than an
error.

qmlkit therefore **derives** each rule from the gate's declared generator
frequencies. Nothing is transcribed.

## Where the rule comes from

For a gate $U(\theta) = e^{-i\theta G/2}$, the expectation $E(\theta)$ is a
trigonometric polynomial whose frequencies are the gaps between eigenvalues of $G$.
A gate that declares frequencies $\{\Omega_1, \dots, \Omega_k\}$ has

$$E(\theta) = a_0 + \sum_{j=1}^{k} \left[ a_j \cos(\Omega_j \theta) + b_j \sin(\Omega_j \theta) \right]$$

Reconstructing $E'(\theta)$ exactly from finitely many evaluations of $E$ is then a
linear algebra problem: choose $2k$ shifts, write down what each evaluation
contributes, and solve for the coefficients. `general_shift_rule` does exactly that.

```python
import qmlkit as qk

rule = qk.general_shift_rule((0.5, 1.0))
for shift, coeff in zip(rule.shifts, rule.coeffs):
    print(f"shift {shift:+.6f}   coefficient {coeff:+.6f}")
```

```text
shift +1.570796   coefficient +0.426777
shift -1.570796   coefficient -0.426777
shift +4.712389   coefficient -0.073223
shift -4.712389   coefficient +0.073223
```

Solving a linear system rather than looking up a formula means a gate you register
yourself gets a correct rule automatically, as long as you declare its frequencies.

## Rules are per gate

```python
import qmlkit as qk

for gate in ("rx", "ry", "rz", "phase", "crx", "cry", "crz"):
    rule = qk.rule_for_gate(gate)
    print(f"{gate:<7} frequencies {str(qk.get_gate(gate).frequencies):<14} {len(rule.shifts)}-term rule")
```

```text
rx      frequencies (1.0,)        2-term rule
ry      frequencies (1.0,)        2-term rule
rz      frequencies (1.0,)        2-term rule
phase   frequencies (1.0,)        2-term rule
crx     frequencies (0.5, 1.0)    4-term rule
cry     frequencies (0.5, 1.0)    4-term rule
crz     frequencies (0.5, 1.0)    4-term rule
```

A controlled rotation's generator is a projector times a Pauli. Its eigenvalue gaps
include both ½ and 1, so a two-term rule cannot reconstruct the derivative.
[Tutorial 3](../tutorials/03-gradients.md) has a worked case where the naive rule
returns exactly √2 times the right answer — same sign, same shape, wrong magnitude.

## Occurrences are shifted one at a time

When one logical parameter fills several slots, the chain rule gives a **sum** over
occurrences:

$$\frac{\partial E}{\partial \theta_k} = \sum_{\text{slots } s \text{ using } \theta_k} \frac{\partial E}{\partial \phi_s}$$

Shifting every occurrence simultaneously computes a directional derivative along a
different direction entirely. For three tied `Ry(θ)` on one qubit — which is
`Ry(3θ)` — it is wrong by a factor of −3.

This is not an exotic case. It is exactly what a QCNN does, and weight tying is the
entire point of a convolutional ansatz.

## Cost is not `2P`

Because rules differ per gate, the real cost is a sum over slots, not a multiple of
the parameter count:

```python
import qmlkit as qk

qc = qk.QCircuit(2)
qc.ry(0, qk.ParamRef(0)).crz(0, 1, qk.ParamRef(1))
spec = qc.to_spec()

print(f"P = {spec.n_params}")
print(f"naive 2P      = {2 * spec.n_params}")
print(f"actual cost   = {qk.grad_circuit_cost(spec)}   (2 for the ry, 4 for the crz)")
```

```text
P = 2
naive 2P      = 4
actual cost   = 6   (2 for the ry, 4 for the crz)
```

`grad_circuit_cost` sums the real per-slot rule cost. On hardware, where each circuit
carries fixed latency, that difference is the difference between a job that fits in
your queue allocation and one that does not.

## Scaling and offsets

A `ParamRef` may carry a scale and an offset — `φ = scale·θ + offset`. The chain rule
then multiplies the slot's contribution by `scale`:

```python
import numpy as np
import qmlkit as qk

qc = qk.QCircuit(1)
qc.ry(0, qk.ParamRef(0, scale=2.0))
spec = qc.to_spec()

g = qk.grad(spec, np.array([0.3]), qk.Z(0), method="parameter-shift")
print(f"gradient           {g[0]:+.10f}")
print(f"2 · -sin(2·0.3)    {-2 * np.sin(0.6):+.10f}")
print(f"without the scale  {-np.sin(0.6):+.10f}")
```

```text
gradient           -1.1292849468
2 · -sin(2·0.3)    -1.1292849468
without the scale  -0.5646424734
```

Dropping the `scale` factor halves the gradient here. Like the other two failure
modes on this page, the result stays smooth and finite — training still descends,
just at the wrong rate.

## When it still does not apply

A generator with a continuum of frequencies — a general time-evolution operator, say
— has no finite shift rule. The literature's answer is the **stochastic**
parameter-shift rule (Banchi & Crooks 2020), which samples an integral instead. It is
not implemented here; `method="adjoint"` covers those cases on a simulator, and
`register_gradient` is the hook if you need the hardware-valid version.
