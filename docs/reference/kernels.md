# Kernels

Three overlap estimators, Gram matrices that stay positive semi-definite, and the models on top.

## `qmlkit.kernels.estimators`

::: qmlkit.kernels.estimators

## `qmlkit.kernels.matrix`

::: qmlkit.kernels.matrix

## `qmlkit.kernels.models`

### What `QSVC` and `QSVR` take

Both inherit their constructor and their whole interface from a shared private base,
so the generated entries below show only `feature_map`, the solver argument (`C` or
`epsilon`) and `**kwargs`. These are the arguments hiding in that `**kwargs`, and they
are the ones worth knowing:

| Argument | Default | What it does |
|---|---|---|
| `shots` | `None` | Exact by default. A shot count makes the Gram matrix noisy, which is what pushes it out of the PSD cone |
| `backend` | `None` | Any registered backend |
| `bandwidth` | `1.0` | Scales the feature map's angles. **The first thing to try when a kernel has concentrated** — a narrower bandwidth spreads the off-diagonal entries back out |
| `estimator` | `"inversion"` | `inversion`, `swap` or `hadamard`. They agree on a simulator and differ in qubit count and circuit cost |
| `repair_psd` | `"threshold"` | Shot noise costs a Gram matrix its positive semi-definiteness; `fit` repairs it by default rather than handing an SVM something it cannot solve |
| `seed` | `None` | Seeds the sampling when `shots` is set |

They also inherit `fit`, `predict`, `decision_function`, `score`, `get_params`,
`set_params`, and the two cost counters `n_circuit_evaluations` and
`circuits_on_hardware` — budget a device run from the second, which counts the
pairwise circuits a simulator's statevector shortcut avoids.

Storing every constructor argument under its own name, unchanged, is scikit-learn's
one requirement for `clone` — and `clone` is what `Pipeline`, `GridSearchCV` and
`cross_val_score` are built on. That is what lets a quantum estimator sit inside an
ordinary scikit-learn workflow.

::: qmlkit.kernels.models
