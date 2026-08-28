# Core

The circuit IR, gates, observables, execution and backends. Everything else in the library reads or writes these types.

## `qmlkit.core.ir`

::: qmlkit.core.ir

## `qmlkit.core.gates`

::: qmlkit.core.gates

## `qmlkit.core.observables`

::: qmlkit.core.observables

## `qmlkit.core.builder`

::: qmlkit.core.builder

## `qmlkit.core.execute`

::: qmlkit.core.execute

## `qmlkit.core.backends.base`

The protocol every backend implements. A simulator supplies `statevector`; a device
supplies `counts`. Everything else — sampling, basis rotation, qubit-wise-commuting
grouping, expectation values, batched execution — is derived here once, which is what
makes agreement between backends a property rather than a coincidence.

::: qmlkit.core.backends.base

## `qmlkit.core.backends.registry`

::: qmlkit.core.backends.registry

## `qmlkit.interop`

::: qmlkit.interop
