# API stability

Research code outlives the version it was written against. A script from last year
that no longer runs is not a small inconvenience — it is a result nobody can
reproduce, and it is the most common complaint about every library in this field.

So this page is a promise, not a description.

## What is promised

Everything exported from the top-level `qmlkit` namespace — `qk.expectation`,
`qk.grad`, `qk.VQC`, `qk.QuantumKernel`, the registries, and everything else in
`qmlkit.__all__` — is **public API**. Within a major version:

- A public name will not be **removed** without a deprecation period.
- A public function will not **change what it returns**, or the meaning of an
  argument it already accepts.
- A **default will not change silently.** If a default changes, the old behaviour
  stays reachable by passing the old value explicitly, and the change is in the
  changelog under `Changed`.
- **Numerical conventions are frozen**: qubit 0 stays the most significant bit,
  angles stay in radians, divergences stay in nats, `shots=None` stays exact.

Names beginning with an underscore, and anything under `qmlkit.core.*` not
re-exported at the top level, are internal. They may change at any time.

## The deprecation period

A public name that is going away:

1. keeps working, and emits a `DeprecationWarning` naming its replacement;
2. stays that way for **two minor releases**;
3. is removed only in the release after that, and only with a changelog entry.

A `DeprecationWarning` from qmlkit always names what to use instead. If one does
not, that is a bug worth reporting.

## What is explicitly *not* promised

Being honest about the edges is what makes the rest of the promise worth having.

| | |
|---|---|
| **The `0.x` line** | Semantic versioning allows breaking changes in `0.x`, and this project uses that latitude — with the deprecation period above applied anyway wherever it is practical. The guarantees tighten at `1.0`. |
| **Exact floating-point output** | Results are exact to the tolerances the test suite asserts, not bit-for-bit across versions, platforms, or NumPy releases. An optimisation that changes the last two digits is not a breaking change. |
| **Performance** | Speed may change in either direction. Where a change makes something *slower* for a plausible workload, it is in the changelog. |
| **Simulator-only scope** | The whole `0.x` line is simulator-only. That is a scope decision, stated in the README, not something to be inferred from what happens to work. |
| **Third-party conventions** | If Qiskit, Cirq, PennyLane or SpinQit change a convention qmlkit maps onto, the mapping follows theirs. `tests/test_cross_backend.py` and `tests/test_pennylane_parity.py` exist to catch that when it happens. |

## How you can check

The promise is only as good as the evidence for it, so:

- **Executable documentation.** Every Python block on this site runs in CI
  (`tests/test_docs.py`). An API change that breaks a documented example breaks the
  build before it reaches you.
- **Cross-library parity.** 301 cases against PennyLane and every metric against
  scikit-learn, so a convention cannot drift unnoticed.
- **[`qk.fingerprint()`](../reference/evaluation.md)** records the versions that
  produced a number, so a result that stops reproducing can be traced to the layer
  that moved.

If you find something that broke without a deprecation, that is a bug — please
[open an issue](https://github.com/Ziadt160/qmlkit/issues).
