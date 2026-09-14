# Evaluation

Scoring predictions, handling skewed classes, comparing against classical
baselines, costing a run before it starts, and recording what produced a number.

The guide is [Evaluating a quantum model honestly](../guides/evaluation.md).

## `qmlkit.evaluate`

::: qmlkit.evaluate

## `qmlkit.imbalance`

::: qmlkit.imbalance

## `qmlkit.search`

::: qmlkit.search

## `qmlkit.baselines`

::: qmlkit.baselines

## `qmlkit.budget`

::: qmlkit.budget

## `qmlkit.provenance`

::: qmlkit.provenance

## `qmlkit.nn.losses`

::: qmlkit.nn.losses

## `qmlkit.recommend`

Which backend to run a circuit on, from measured crossovers rather than a rule of
thumb. [Backends](backends.md) gives the static table; this computes the answer for
the circuit in front of you, and will name `mps` where a statevector would not fit.

::: qmlkit.recommend

## `qmlkit.parallel`

Independent work — folds, seeds, sweep configurations — run on a thread pool. Not for
parallelising *inside* one circuit, which is the backend's job. `qk.search(...,
n_jobs=4)` dispatches through this; so can your own loops.

::: qmlkit.parallel

## `qmlkit.progress`

See [Watching a run](../guides/watching-a-run.md) for the guide.

::: qmlkit.progress

## `qmlkit.report`

::: qmlkit.report

## `qmlkit.utils.errors`

How every "unknown gate / backend / ansatz / method" message in the library is built.
Worth reading before adding one, since the shape of the message is the contract:
what was wrong, what was probably meant, and what is allowed.

::: qmlkit.utils.errors
