# Backends

Nine simulators behind one protocol. A backend supplies a statevector — or, for a
device, counts — and sampling, basis rotation, qubit-wise-commuting grouping,
expectation and the whole batch stack are derived once in `base.py`. That is what makes
agreement between them a property rather than a coincidence, and it is why adding one
is a single `register_backend` call.

The reference below was split out of [Core](core.md) because seven of the nine
implementation modules had no entry anywhere: their constructor options, their refusals
and their approximations were readable only in the source. A backend that silently
truncates past a bond dimension, or refuses a GPU it cannot find, should say so on the
page you look it up on.

## Which one supports what

`supports_statevector` and `supports_exact` are a contract, not metadata — they decide
which gradients, diagnostics and metrics are even legal on a given backend. Anything
needing a statevector (adjoint differentiation, the metric tensor, expressibility,
`diagnose`'s structural checks) refuses rather than substituting a sampled stand-in.

| Backend | statevector | exact | notes |
|---|---|---|---|
| `numpy` | yes | yes | the reference. Vectorised, with gate fusion; hard cap at 24 qubits |
| `aer` | yes | yes | C++ statevector, faster above ~13 qubits |
| `mps` | **no** | until truncation | matrix-product state; wide but lightly entangled circuits |
| `qiskit` | yes | yes | `quantum_info.Statevector`, the slowest of the exact three |
| `cirq` | yes | yes | Cirq's reference simulator |
| `spinqit` | yes | yes | Python 3.10 only |
| `torch` | yes | yes | differentiable — the only one `backprop` can use |
| `cirq-density` | no | yes, given the noise model | density matrix |
| `qiskit-aer` | no | yes, given the noise model | density matrix |

"Exact given the noise model" is deliberate on the two mixed-state rows: they return a
shot-free number, exact for the channel you asked for. That is not the same as the
noiseless answer, and `diagnose` separates the two rather than reporting their product.

## The protocol

::: qmlkit.core.backends.base

## The registry

::: qmlkit.core.backends.registry

## Exact statevector simulators

### `qmlkit.core.backends.numpy_backend`

::: qmlkit.core.backends.numpy_backend

### `qmlkit.core.backends.aer_backend`

::: qmlkit.core.backends.aer_backend

### `qmlkit.core.backends.qiskit_backend`

::: qmlkit.core.backends.qiskit_backend

### `qmlkit.core.backends.cirq_backend`

::: qmlkit.core.backends.cirq_backend

### `qmlkit.core.backends.spinqit_backend`

::: qmlkit.core.backends.spinqit_backend

## Differentiable

### `qmlkit.core.backends.torch_backend`

::: qmlkit.core.backends.torch_backend

## Approximate

### `qmlkit.core.backends.mps_backend`

A matrix-product-state simulator reaches widths a statevector cannot, by keeping only
as much entanglement as its bond dimension allows. It has no `statevector()`, so every
route that needs one is refused rather than approximated; and past the bond dimension
it is no longer exact, which is a different thing from noisy.

::: qmlkit.core.backends.mps_backend

## Mixed-state, for noise

### `qmlkit.core.backends.noisy`

::: qmlkit.core.backends.noisy

### `qmlkit.core.backends.cirq_density_backend`

::: qmlkit.core.backends.cirq_density_backend

### `qmlkit.core.backends.qiskit_aer_backend`

::: qmlkit.core.backends.qiskit_aer_backend
