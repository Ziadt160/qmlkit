"""Which simulator, and which of its optimisations, for the circuit you have.

:func:`~qmlkit.budget.plan` answers *what will this run cost*. This answers the
question that comes immediately after it — **what should I run it on** — because the
right answer changes by an order of magnitude across the widths this library is used
at, and none of the reasons are guessable from the outside.

Three crossovers decide it, and all three were measured rather than reasoned about:

* **Batching** (one circuit at many parameter vectors, carried as a leading axis) is
  worth 30x at 4 qubits and *loses* by 11. It is off above
  ``NumpyBackend.batch_max_qubits``.
* **Gate fusion** (adjacent gates composed into one wider matrix) loses below 14
  qubits — the ``2**(2k)`` block is comparable to the ``2**n`` state, and 84% of the
  runtime goes on building blocks — and is worth 5.6x by 18.
* **Aer** is a C++ simulator whose per-call cost is Python. It loses to the fused
  NumPy reference up to about 11 qubits, then wins by 4.5x at 16 and 11.8x at 20 -
  once it is told how many circuits to run at once, which by default it is not.

The three are not independent, and that is the point: they are all the same fact seen
from different sides. Below ~11 qubits this library is bound by *per-call overhead*,
and the answer is always to make fewer, fatter calls — batch them, or fuse them. Above
it the cost is arithmetic on an array too big for cache, and C++ wins.

    >>> import qmlkit as qk
    >>> print(qk.recommend(qk.hardware_efficient(18, 3)))     # doctest: +SKIP
    18 qubits, 159 gates - 4.0 MiB per statevector
      use  backend='aer'          AerSimulator, C++ - the NumPy reference loses above ~11 qubits
      optimisations:              gate fusion off, batching off
      ...

Nothing here is a promise about *your* hardware. Every number below was measured on
one machine, single-threaded, and the honest use of it is as a starting point for
`examples/benchmark_pennylane.py` and `scripts/probe_dispatch.py` on yours.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from qmlkit.core.backends.registry import is_available

__all__ = ["Recommendation", "recommend"]

#: Bytes per amplitude: complex128, which is not negotiable — ``gradcheck`` needs
#: float64 and so does any honest claim that parameter-shift is *exact*.
_BYTES_PER_AMPLITUDE = 16

#: The width above which Aer's C++ kernel beats the fused NumPy reference. Measured on
#: a Gram matrix over a ZZFeatureMap, numpy/aer seconds: 0.59/0.70 at 10 qubits,
#: 0.86/0.88 at 11, 0.93/0.69 at 12, 0.99/0.53 at 14, 2.00/0.44 at 16, 10.56/0.89 at 20.
#:
#: This number moved twice in one afternoon and the moves are the interesting part.
#: Fusion pushed it *up* from 13 to 15 by making the reference faster; giving Aer a
#: memory-aware `max_parallel_experiments` pushed it back *down* to 11 by making Aer
#: faster still. Neither was predictable from the outside, which is the argument for
#: measuring the boundary rather than reasoning about it.
AER_CROSSOVER = 11

#: Above this a statevector is measured in gigabytes and an MPS is the only thing
#: left — 24 qubits is 256 MiB, 30 is 16 GiB, and Aer refuses 32 outright. Measured on
#: a chain circuit, seconds for one expectation: 0.279/0.002 at 24 qubits, 4.97/0.002
#: at 28, and at 32 the statevector simulator fails while MPS is unchanged. The catch
#: is that this holds only while the circuit stays lightly entangled.
MPS_CROSSOVER = 22

#: Where a statevector stops fitting a core's private cache and the run becomes
#: memory-bound. Below this, threading a single circuit is *slower* — measured 0.04x
#: on 8 threads at 8 qubits, because handing out the work costs 17x the gate.
MEMORY_BOUND_QUBITS = 16


@dataclass(frozen=True)
class Recommendation:
    """What to run this on, and what that turns on."""

    n_qubits: int
    n_gates: int
    statevector_bytes: int
    backend: str
    reason: str
    fusion: bool
    batching: bool
    alternatives: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = field(default=())

    @property
    def statevector_mib(self) -> float:
        return self.statevector_bytes / 1024 / 1024

    def __str__(self) -> str:
        size = (
            f"{self.statevector_bytes / 1024:.1f} KiB"
            if self.statevector_mib < 1
            else f"{self.statevector_mib:,.1f} MiB"
        )
        lines = [
            f"{self.n_qubits} qubits, {self.n_gates} gates - {size} per statevector",
            f"  use  backend={self.backend!r}".ljust(30) + f"  {self.reason}",
        ]
        lines.append(
            "  optimisations:".ljust(30)
            + f"  gate fusion {'on' if self.fusion else 'off'}"
            + f", batching {'on' if self.batching else 'off'}"
        )
        if self.alternatives:
            lines.append("\n  considered and not chosen:")
            lines.extend(f"    {name:<14} {why}" for name, why in self.alternatives)
        if self.notes:
            lines.append("")
            lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _as_spec(model: Any) -> Any:
    """An ``Ansatz``, a ``CircuitSpec``, or anything holding one."""
    from qmlkit.core.ir import CircuitSpec

    if isinstance(model, CircuitSpec):
        return model
    build = getattr(model, "build", None)
    if callable(build):
        return build()
    inner = getattr(model, "ansatz", None)
    if inner is not None and hasattr(inner, "build"):
        return inner.build()
    raise TypeError(
        f"recommend() cannot read a {type(model).__name__}. It takes an Ansatz, a "
        "CircuitSpec, or a model holding one (QuantumLayer, VQC, VQRegressor)."
    )


def recommend(model: Any, *, shots: int | None = None, batch: int | None = None) -> Recommendation:
    """Which backend to run ``model`` on, and what that switches on.

    Parameters
    ----------
    model
        An :class:`~qmlkit.ansatz.library.Ansatz`, a
        :class:`~qmlkit.core.ir.CircuitSpec`, or a model holding one.
    shots
        ``None`` (the default) means exact. A shot budget rules out nothing here, but
        it does change which gradient methods remain — see
        :func:`~qmlkit.gradients.dispatch.choose_method`.
    batch
        How many parameter vectors will be evaluated together, if you know. Batching
        is what decides the answer below ~10 qubits.

    Returns
    -------
    Recommendation
        Falsy-free: it always names a backend, because the NumPy reference is always
        installed and always correct. The interesting part is the alternatives and
        why each was not chosen.
    """
    from qmlkit.core.backends.numpy_backend import NumpyBackend

    spec = _as_spec(model)
    n = spec.n_qubits
    reference = NumpyBackend()
    size = 2**n * _BYTES_PER_AMPLITUDE

    aer_here = is_available("aer")
    batching = n <= reference.batch_max_qubits and (batch is None or batch > 1)
    alternatives: list[tuple[str, str]] = []
    notes: list[str] = []

    # An MPS has no 2**n array at all, so it is the only thing that reaches widths a
    # statevector cannot — but only while the circuit stays lightly entangled, and it
    # cannot answer a statevector question at any price.
    entangling = sum(1 for op in spec.ops if len(op.qubits) > 1)
    if n > MPS_CROSSOVER and aer_here:
        backend = "mps"
        reason = "no 2**n array at all - the only thing that reaches this width"
        fusion = False
        alternatives.append(
            ("aer", f"needs {size / 1024**3:.1f} GiB for one statevector at {n} qubits")
        )
        notes.append(
            "MPS is exact only while the bond dimension holds. It is at its best on "
            "shallow or chain-structured circuits and at its worst on a full entangler "
            f"ring - this one has {entangling} multi-qubit gates. Check what a run "
            "actually needed with get_backend('mps').bond_dimension(spec)."
        )
        notes.append(
            "it has no statevector, so adjoint and backprop refuse; parameter-shift "
            "works, because a shift rule never inspects a state"
        )
        return Recommendation(
            n_qubits=n,
            n_gates=len(spec.ops),
            statevector_bytes=size,
            backend=backend,
            reason=reason,
            fusion=fusion,
            batching=False,
            alternatives=tuple(alternatives),
            notes=tuple(notes),
        )

    if n > AER_CROSSOVER and aer_here:
        backend = "aer"
        reason = f"AerSimulator, C++ - the NumPy reference loses above ~{AER_CROSSOVER} qubits"
        fusion = False  # Aer does its own; qmlkit's pass is for the NumPy path
        alternatives.append(
            (
                "numpy",
                f"fused, but still a Python loop at {n} qubits: 4.5x slower at 16, 11.8x at 20",
            )
        )
        alternatives.append(
            ("qiskit", "quantum_info.Statevector is Qiskit's reference, not its fast path")
        )
    else:
        backend = "numpy"
        fusion = n >= reference.fuse_min_qubits
        if n > AER_CROSSOVER:
            reason = "the reference - and the only one installed that is worth using here"
            notes.append(
                "pip install 'qmlkit[aer]' - measured 4.5x faster than this at 16 qubits "
                "and 11.8x at 20, and the gap widens"
            )
        elif batching:
            reason = f"vectorised across the batch; measured up to 30x at {n} qubits"
        else:
            reason = (
                "exact, always present, and the reference every other backend is tested against"
            )
        if aer_here:
            alternatives.append(
                ("aer", f"C++, but its per-call cost dominates below ~{AER_CROSSOVER} qubits")
            )
        if is_available("qiskit"):
            alternatives.append(("qiskit", "Qiskit's pure-Python reference; slower than both"))

    if fusion:
        notes.append(
            f"gate fusion is on (blocks of up to {reference._fuse_width(n)} qubits): "
            "measured 2.8x at 15 qubits, 5.6x at 18"
        )
    elif n < reference.fuse_min_qubits and backend == "numpy":
        notes.append(
            f"gate fusion stays off below {reference.fuse_min_qubits} qubits - the block "
            "matrix costs more to build than the applies it replaces (0.68x at 6)"
        )

    if n >= MEMORY_BOUND_QUBITS:
        notes.append(
            f"at {n} qubits the state is {size / 1024 / 1024:,.0f} MiB and no longer "
            "cache-resident, so this is memory-bound: more threads saturate at ~1.5x "
            "rather than scaling"
        )
    else:
        notes.append(
            f"**your CPU will sit near {100 // (os.cpu_count() or 1)}% and that is correct.** "
            f"At {n} qubits the statevector is {size / 1024:,.0f} KiB and fits in one "
            "core's cache, so there is no work to spread: threading a single circuit "
            "measured 0.04x on 8 threads at 8 qubits, because handing out the work "
            "costs 17x what the gate costs. Use more cores on *independent* circuits - "
            "folds, seeds, hyperparameter configurations - not inside one."
        )

    if shots is not None:
        notes.append(f"shots={shots:,} rules out adjoint and backprop; parameter-shift still works")

    return Recommendation(
        n_qubits=n,
        n_gates=len(spec.ops),
        statevector_bytes=size,
        backend=backend,
        reason=reason,
        fusion=fusion,
        batching=batching,
        alternatives=tuple(alternatives),
        notes=tuple(notes),
    )
