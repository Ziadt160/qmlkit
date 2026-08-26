# 7. Re-uploading and the Fourier picture

Encode the data once and a circuit is a fairly limited function of it. Encode it
again between trainable blocks and the model becomes a **truncated Fourier series**,
where each upload buys one more frequency. That is the single clearest theoretical
statement in variational QML, and it is directly measurable.

## Re-uploading is a pattern, not a structure

There is no single "the re-uploading ansatz". It is *any* interleaving of *any*
encoding with *any* trainable block, so qmlkit treats it as a composition rather than
a class:

```python
import qmlkit as qk

variants = [
    ("default (S then W)", qk.reupload(qk.AngleFeatureMap(2), n_layers=3)),
    ("order='WS'", qk.reupload(qk.AngleFeatureMap(2), n_layers=3, order="WS")),
    ("shared weights", qk.reupload(qk.AngleFeatureMap(2), n_layers=3, share_weights=True)),
    ("ZZ feature map", qk.reupload(qk.ZZFeatureMap(2, reps=1), n_layers=2)),
]
for label, model in variants:
    r = model.resources()
    print(f"{label:<20} inputs {model.n_inputs}  weights {model.n_weights:>3}  depth {r['depth']:>3}")
```

```text
default (S then W)   inputs 2  weights  18  depth  18
order='WS'           inputs 2  weights  18  depth  18
shared weights       inputs 2  weights   6  depth  18
ZZ feature map       inputs 3  weights  12  depth  18
```

`reupload()` covers the common shapes. Anything else composes directly from the block
vocabulary — including **two different feature maps in one model**, which a fixed
class cannot express at all.

Notice `share_weights=True` gives 6 weights instead of 18, at identical depth: the
same trainable block reused at every upload.

## The circuit

```python
import qmlkit as qk

model = qk.reupload(
    qk.AngleFeatureMap(1, entangle=False),
    n_layers=2,
    rotations=("rz", "ry", "rz"),
    entangler=None,
)
print(qk.draw(model.build()))
```

```text
q0: ─RY(θ0)──RZ(θ1)──RY(θ2)──RZ(θ3)──RY(θ0)──RZ(θ4)──RY(θ5)──RZ(θ6)──
```

`θ0` appears twice — that is the data, uploaded twice. The rest are weights. Data and
weights occupy separate ranges of one flat vector, which is what keeps `∂f/∂x` and
`∂f/∂θ` separable while the model still drops straight into a `QuantumLayer`.

## Measuring the claim

`L` uploads should reach frequencies `0…L`. Do not take that on faith — extract the
spectrum:

```python
import numpy as np
import qmlkit as qk

for n_layers in (1, 2, 3, 4):
    model = qk.reupload(
        qk.AngleFeatureMap(1, entangle=False),
        n_layers=n_layers,
        rotations=("rz", "ry", "rz"),
        entangler=None,
    )
    weights = model.init("uniform", seed=1)
    bound = model.build()

    def f(x, bound=bound, model=model, weights=weights):
        return qk.expval(bound, qk.Z(0), theta=np.concatenate([model.angles([x]), weights]))

    spectrum = qk.fourier.spectrum(f, n_layers + 3)
    print(f"L={n_layers}: frequencies {sorted(spectrum)}"
          f"  amplitudes {[round(spectrum[k], 4) for k in sorted(spectrum)]}")
```

```text
L=1: frequencies [1]  amplitudes [0.9997]
L=2: frequencies [0, 1, 2]  amplitudes [0.0853, 0.2176, 0.1573]
L=3: frequencies [0, 1, 2, 3]  amplitudes [0.4162, 0.5004, 0.1654, 0.0156]
L=4: frequencies [0, 1, 2, 3, 4]  amplitudes [0.4088, 0.5291, 0.2017, 0.0506, 0.0028]
```

Exactly `0…L`, and nothing above it. Frequencies the architecture cannot reach are
not "hard to learn" — they are unreachable, and no training run will find them. That
makes this a design question, decided before the first gradient step.

Note the amplitudes fall off sharply at the top end. Reaching a frequency is not the
same as having much of it: `L=4` reaches frequency 4 with amplitude 0.0028. Depth
buys bandwidth, not power.

## The trap: a trainable block that commutes

If the trainable block commutes with the encoding, the uploads collapse.
`Ry(x)Ry(θ₁)Ry(x)Ry(θ₂)` is just `Ry(2x + θ₁ + θ₂)` — one frequency, and the weights
do nothing but shift a phase.

```python
import numpy as np
import qmlkit as qk

for label, rotations in (("Ry only (commutes)", ("ry",)), ("Rz Ry Rz (does not)", ("rz", "ry", "rz"))):
    model = qk.reupload(
        qk.AngleFeatureMap(1, entangle=False), n_layers=3, rotations=rotations, entangler=None
    )
    weights = model.init("uniform", seed=1)
    bound = model.build()

    def f(x, bound=bound, model=model, weights=weights):
        return qk.expval(bound, qk.Z(0), theta=np.concatenate([model.angles([x]), weights]))

    spectrum = qk.fourier.spectrum(f, 6)
    print(f"{label:<22} frequencies {sorted(spectrum)}"
          f"  amplitudes {[round(spectrum[k], 4) for k in sorted(spectrum)]}")
```

```text
Ry only (commutes)     frequencies [3]  amplitudes [1.0]
Rz Ry Rz (does not)    frequencies [0, 1, 2, 3]  amplitudes [0.4162, 0.5004, 0.1654, 0.0156]
```

A three-upload model that reaches **one** frequency, with weights that cannot change
the function's shape. It trains, it produces a loss curve, and it is architecturally
incapable of the thing it was built for.

!!! warning "The library warns about this one"
    `DataReuploadEncoder` raises a `UserWarning` when the trainable block commutes
    with the encoding rotation, because the failure is otherwise invisible — the
    model looks fine and simply cannot represent anything.

## Using it

A re-uploading model is an `Ansatz`, so it drops into everything else. Bind data and
weights separately:

```python
import numpy as np
import qmlkit as qk

model = qk.reupload(qk.AngleFeatureMap(2), n_layers=2)
weights = model.init(seed=0)
x = np.array([0.3, 0.8])

spec = model.bind(x, weights)
print(f"<Z0> = {qk.expval(spec, qk.Z(0)):+.6f}")
print(f"inputs {model.n_inputs}, weights {model.n_weights}")
```

`build(theta)` still takes the full concatenated vector, and says so clearly when the
sizes disagree — the two calling conventions were a real source of confusion, so the
error message names both sizes.

---

**Next:** [Trainability](08-trainability.md) — what happens when the gradient is
there but vanishingly small.
