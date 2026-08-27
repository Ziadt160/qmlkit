# Working with a coding agent

Most code written against a library now is written by a model, and a model does not
read this page before it types. It guesses a name from what it has read elsewhere,
runs it, reads the traceback, and tries again.

That loop is the real interface. Three things follow from taking it seriously, and
all three help a human equally — none of this is a special mode.

## A wrong name answers with the right one

The training data for any model contains far more PennyLane and Qiskit than qmlkit,
so the first guess at a name is usually theirs. That guess now costs one line
instead of a search:

```python
import qmlkit as qk

try:
    qk.AngleEmbedding
except AttributeError as exc:
    print(exc)
```

```
module 'qmlkit' has no attribute 'AngleEmbedding'. 'AngleEmbedding' is PennyLane's
name for qmlkit.AngleFeatureMap (or angle_encode(x) for a one-shot circuit).
```

It is a **translation, not an alias** — the foreign name still raises. Two reasons.
Anything importable is something somebody will depend on, and a shadow vocabulary of
thirty PennyLane spellings is a second public surface to keep working forever. More
importantly, an alias would hide semantic drift: `qml.expval` takes a QNode where
`qk.expectation` takes a circuit and an observable, so a name that silently resolved
would fail later and further from its cause.

Names that are merely misspelled get the same treatment from every registry, and
case or separator drift resolves to a single certain suggestion rather than a fuzzy
one — `parameter_shift` for `parameter-shift` is what half-remembering another
library looks like, not a typo:

```python
try:
    qk.get_ansatz("hardware-efficient")
except KeyError as exc:
    print(exc)
```

```
unknown ansatz 'hardware-efficient'. Did you mean 'hardware_efficient'? Valid:
basic_entangler, hardware_efficient, mps, qaoa, qcnn, random_layers,
simplified_two_design, strongly_entangling, tree_tensor_network, two_local. Add
your own with register_ansatz(name, factory).
```

Every gate, backend, gradient method, ansatz and conv filter behaves this way. So
does asking for a backend you have not installed — but that is a *different* answer,
because the name was right and the environment was wrong, and it names the install
command instead.

## `diagnose()` catches what does not raise

This is the part that has nothing to do with names, and it is the one that matters
most. In this field a mistake usually returns a number of the right shape and the
right range, and the model trains, converges and reports an accuracy.

Here is a re-uploading model composed by hand out of blocks. It has three uploads,
six weights, and one frequency:

```python
from qmlkit.ansatz import Ansatz, EncodingLayer, RotationLayer, repeat

fmap = qk.AngleFeatureMap(2, rotation="ry")
model = Ansatz(2, repeat(3, EncodingLayer(fmap) + RotationLayer("ry")), n_inputs=2)

print(qk.diagnose(model))
```

```
ansatz on 2 qubits: 1 finding(s)
  [error] ENCODING_COMMUTES: 3 uploads, but every trainable rotation is 'ry', the
  same generator the encoding uses. RY(x) RY(t) composes into one rotation, so the
  model reaches 1 frequency rather than 0..3, and its weights do nothing beyond a
  phase.  Fix: Use a non-commuting block, e.g. RotationLayer(('rz', 'ry', 'rz')).
```

Nothing about that model raises. It builds, binds, differentiates and trains; it is
simply not the model that was intended. `reupload()` warns about this case at
construction, but a model composed directly out of blocks has no constructor to warn
from — and composing directly is the whole point of the block vocabulary.

Take the fix and the report goes quiet:

```python
fixed = Ansatz(2, repeat(3, EncodingLayer(fmap) + RotationLayer(("rz", "ry", "rz"))), n_inputs=2)
report = qk.diagnose(fixed)
print(bool(report), report.codes)
```

```
False ()
```

A report is falsy when it found nothing, so `if qk.diagnose(model): ...` reads the
way it should. Each finding carries a stable `code` to branch on, the number that
was measured, and the edit that resolves it:

```python
finding = qk.diagnose(model)[0]
print(finding.code, finding.severity, finding.value)
```

```
ENCODING_COMMUTES error 3.0
```

It takes an `Ansatz`, anything holding one — a `QuantumLayer`, a `VQC`, an
`nn.Sequential` with a quantum layer somewhere inside it — or a Gram matrix, where
it checks for concentration, for a signal below the shot noise, and for a matrix
that has stopped being positive semi-definite:

```python
import numpy as np

flat = np.full((8, 8), 0.5)
np.fill_diagonal(flat, 1.0)
print(qk.diagnose(flat).codes)
```

```
('KERNEL_CONCENTRATED',)
```

What it checks, and on what evidence, is in
[the API reference](../reference/analysis.md#qmlkit.diagnostics.diagnose). Checks
that can be exact are exact: a parameter is dead if shifting it cannot change the
state at all. Checks that are statistical report the number they measured, so the
threshold can be argued with rather than trusted.

## The whole library in one fetch

A model that has to click through a documentation site mostly does not. Two files
are generated from these pages and from the package itself, and served at the root
of the site:

- **[`/llms.txt`](https://ziadt160.github.io/qmlkit/llms.txt)** — what is here, where
  it is, and the handful of constraints that are not inferable from the API.
- **[`/llms-full.txt`](https://ziadt160.github.io/qmlkit/llms-full.txt)** — every
  tutorial and guide in full, then the entire public API with signatures and summary
  lines. One fetch, no navigation.

Both are generated by `scripts/generate_llms_txt.py`, committed, and checked in CI,
for the same reason `tests/test_docs.py` executes every snippet on every page: a
summary of an API written by hand is a second copy of the truth, and second copies
rot silently. Change a page or a public signature without regenerating and the build
goes red.

For working *on* qmlkit rather than with it, [`AGENTS.md`](https://github.com/Ziadt160/qmlkit/blob/main/AGENTS.md)
carries the commands, the conventions the tests enforce, and the traps that have
already cost time.
