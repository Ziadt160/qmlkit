# Gradients

Six estimators behind one `grad()`, plus the shift-rule machinery they share.

## `qmlkit.gradients.batch`

Gradients for a whole training batch in one pass. `param_shift_grad_batch` never
inspects a state, so it works on every backend including a sampling-only device, and
is the batched submission real hardware wants.

::: qmlkit.gradients.batch

## `qmlkit.gradients.dispatch`

::: qmlkit.gradients.dispatch

## `qmlkit.gradients.rules`

::: qmlkit.gradients.rules

## `qmlkit.gradients.parameter_shift`

::: qmlkit.gradients.parameter_shift

## `qmlkit.gradients.adjoint`

::: qmlkit.gradients.adjoint

## `qmlkit.gradients.hadamard`

::: qmlkit.gradients.hadamard

## `qmlkit.gradients.spsa`

::: qmlkit.gradients.spsa
