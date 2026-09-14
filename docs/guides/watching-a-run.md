# Watching a run

`qk.plan` says what a run will cost before it starts. `qk.diagnose` says what went
wrong after it finishes. In between there was nothing, and in between is where the
hours go.

A quantum kernel on a few hundred points is tens of thousands of circuits. Every
library in this field runs them behind a single call that returns when it returns,
and *how much is left* is one of the most common questions asked about all of them.

```python
# docs: run
import numpy as np
import qmlkit as qk

rng = np.random.default_rng(0)
X = rng.uniform(0, np.pi, (24, 2))
kernel = qk.QuantumKernel(qk.ZZFeatureMap(2), shots=64, seed=0)

with qk.progress(live=False) as run:      # live=True draws the line below
    gram = kernel(X)

print(run.report())
```

```text
kernel gram  3,412/12,720   27%   14.2s elapsed  ~37s left
```

Nothing is printed unless you enter `qk.progress()`, and the live line goes to stderr,
so piping a script's stdout to a file does not collect redraws.

## Where the time actually went

```python
# docs: skip
print(run.report())
```

```text
Run finished in 51.4s
kernel gram               12,720 items    47.9s    3.77 ms/item
fit VQC                      600 items     3.2s    5.33 ms/item
  (untracked)                              0.3s   - setup, data handling, and anything outside a task
```

The `(untracked)` row is there because the first thing a timing breakdown usually gets
wrong is implying that the parts it measured are the whole run. On a short run it is
often the largest row, and that is worth seeing.

## Three properties, in this order

**It does not change any number.** A seeded kernel and a seeded circuit produce
bit-identical results watched and unwatched, and a test asserts exactly that. A
reporter that perturbed a result would be a worse defect than the silence it replaces.

**It is free when nobody is watching.** With no active reporter, the tracking calls
inside the library cost one comparison against `None`, and the live line redraws at
most ten times a second however fast the loop runs.

**It does not claim to know what it does not.** An estimate from four items in half a
second says more about scheduling noise than about the run, so it prints `estimating`
until there is evidence — the same rule the rest of the library follows about reporting
a measurement.

## The trajectory, not just the timings

`HybridModel.fit` — so `VQC` and `VQRegressor` — logs three series per epoch without
being asked:

| | |
|---|---|
| **loss** | whether it is learning |
| **gradient norm** | whether it *can*. A plateau and a solved problem look identical in loss |
| **parameter norm** | whether the weights are running away, which looks like nothing at all in a loss curve |

Computing the two norms costs a pass over the parameters, so it happens only when
something is watching. Anything else can log its own:

```python
# docs: skip
with qk.progress() as run:
    run.note(dataset="breast-cancer", seed=0)   # facts about the run
    for step, value in enumerate(history):
        run.log("validation auc", value, step)
```

## One file, no server

```python
# docs: skip
run.save_html("run.html")
```

Timings, every logged series as a chart, and the versions and settings the run was
made with — in a single self-contained page with no assets directory and nothing to
fetch.

It is deliberately a *file* rather than a dashboard. A dashboard you have to start,
connect to and keep alive is a dependency, a port and a process; a page you can open,
email, and leave next to the result in a directory is none of those and outlives all of
them. The charts are inline SVG for the same reason: nothing to pin, nothing to break
in two years.

A flat series is labelled `flat — never moved` rather than drawn as a straight line at
an arbitrary height, because a loss curve that never moved is the run you most want to
look at, and a picture that hides it is worse than no picture.

## Counting your own loop

```python
# docs: skip
for row in qk.track(rows, "rows"):
    ...
```

`qk.track` counts an iterable as it is consumed and takes its total from `len` when
there is one. Inside the library, `task` is the equivalent — it hands back a silent
stand-in when nothing is watching, so a loop never has to branch on whether anyone is
there.

Reach for either of those through a direct import, `from qmlkit.progress import log,
task`, rather than the dotted `qmlkit.progress.log`. Importing the `progress` context
manager into the top-level namespace rebinds `qmlkit.progress` to the *function*, so
the dotted path raises `AttributeError` — which is what this page used to tell you to
type.

Already instrumented: the pair-at-a-time kernel Gram matrix — the path that takes hours,
used for sampled kernels, non-inversion estimators, and any backend without a
statevector — and `HybridModel.fit`.
