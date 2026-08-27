"""What an experiment costs in circuits, before it is run rather than after.

The number that decides whether a quantum machine learning experiment is possible
is not accuracy. It is circuits: samples times steps times the cost of one
gradient, and on hardware that cost is multiplied by a queue.

Nobody computes it. People start the run, watch the first epoch take four minutes,
multiply in their head, and stop. The arithmetic is not hard — it is just spread
across :func:`~qmlkit.gradients.dispatch.gradient_cost`,
:func:`~qmlkit.core.observables.group_qubit_wise_commuting` and
:func:`~qmlkit.utils.shots.shots_for_precision`, and nobody assembles it up front::

    >>> import qmlkit as qk
    >>> plan = qk.plan(qk.hardware_efficient(4, 3), n_samples=100, steps=50)
    >>> plan.circuits > 0
    True

:class:`Plan` prints the total, the wall-clock at a given seconds-per-circuit, and
the reductions available with what each one costs in exactness. The reductions are
the point: a plan that says "24 days" and stops is a discouragement, while one that
says "24 days, or 6 hours on adjoint, or 90 minutes with SPSA at the price of an
unbiased estimate instead of an exact one" is a decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qmlkit.ansatz.library import Ansatz
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z, as_sum, group_qubit_wise_commuting
from qmlkit.gradients.dispatch import gradient_cost

__all__ = ["Plan", "Reduction", "plan"]

#: Methods worth putting in front of a caller, with what each one gives up.
_METHOD_NOTES = {
    "adjoint": "exact, simulator only - cost does not grow with the parameter count",
    "backprop": "exact, simulator only - needs torch, keeps the circuit in an autograd graph",
    "hadamard": "exact, hardware-valid - halves parameter-shift, needs an ancilla on every wire",
    "parameter-shift": "exact, hardware-valid - the default on a device",
    "spsa": "unbiased estimate, not exact - two evaluations at any parameter count",
    "finite-diff": "biased, debugging only",
}


@dataclass(frozen=True)
class Reduction:
    """One way to make the run cheaper, and what it costs to take it."""

    name: str
    circuits: int
    factor: float
    trade: str

    def __str__(self) -> str:
        return f"{self.name:<16} {self.circuits:>14,}  {self.factor:>7.1f}x   {self.trade}"


@dataclass(frozen=True)
class Plan:
    """The circuit budget for a training run, and the ways to shrink it."""

    circuits: int
    shots_total: int | None
    method: str
    n_params: int
    n_samples: int
    steps: int
    shots: int | None
    measurement_settings: int
    observable_terms: int
    reductions: tuple[Reduction, ...] = ()
    notes: tuple[str, ...] = ()

    def hours(self, seconds_per_circuit: float) -> float:
        """Wall-clock at a given per-circuit latency. A queued device is ~0.5-2 s."""
        return self.circuits * seconds_per_circuit / 3600.0

    @property
    def circuits_per_gradient(self) -> int:
        """The per-gradient factor, so the arithmetic in the printout checks out."""
        return self.circuits // max(self.n_samples * self.steps, 1)

    def __str__(self) -> str:
        shots = f"  |  {self.shots_total:,} shots" if self.shots_total else "  |  exact, no shots"
        lines = [
            f"{self.circuits:,} circuits{shots}",
            f"  {self.n_samples} samples x {self.steps} steps x "
            f"{self.circuits_per_gradient} circuit(s) per gradient",
            f"  gradient method: {self.method} - {_METHOD_NOTES.get(self.method, '')}",
        ]
        if self.observable_terms != self.measurement_settings:
            lines.append(
                f"  qubit-wise-commuting grouping: {self.observable_terms} term(s) -> "
                f"{self.measurement_settings} setting(s)"
            )
        for seconds in (0.001, 0.5, 2.0):
            label = (
                "1 ms/circuit (simulator)"
                if seconds < 0.1
                else f"{seconds:.1f} s/circuit queued"
            )
            lines.append(f"  at {label:<26} {self.hours(seconds):10.2f} hours")
        if self.reductions:
            lines.append("\n  cheaper, and what it costs:")
            lines.extend(f"    {r}" for r in self.reductions)
        lines.extend(f"  note: {note}" for note in self.notes)
        return "\n".join(lines)


def _all_z(n_qubits: int) -> Observable:
    """``Z0 + Z1 + ...`` — the default readout, and one measurement setting."""
    total: Observable = Z(0)
    for i in range(1, n_qubits):
        total = total + Z(i)
    return total


def _as_spec(model: Any) -> tuple[CircuitSpec, Observable]:
    """Accept an ansatz, a circuit, or a model that carries one."""
    if isinstance(model, Ansatz):
        spec = model.build()
        return spec, _all_z(spec.n_qubits)
    if isinstance(model, CircuitSpec):
        return model, _all_z(model.n_qubits)
    quantum = getattr(model, "quantum", None)  # HybridModel and its subclasses
    if quantum is not None and hasattr(quantum, "ansatz"):
        spec = quantum.ansatz.build()
        observables = getattr(quantum, "observables", None)
        if observables:
            total = observables[0]
            for extra in observables[1:]:
                total = total + extra
            return spec, total
        return spec, _all_z(spec.n_qubits)
    for attribute in ("spec", "circuit_spec"):
        if hasattr(model, attribute):
            spec = getattr(model, attribute)
            return spec, _all_z(spec.n_qubits)
    raise TypeError(
        "plan() needs an Ansatz, a CircuitSpec, or a model carrying one, got "
        f"{type(model).__name__}"
    )


def plan(
    model: Any,
    n_samples: int = 1,
    steps: int = 1,
    method: str = "parameter-shift",
    obs: Observable | None = None,
    shots: int | None = None,
) -> Plan:
    """Circuits, shots and wall-clock for a training run, plus the ways to shrink it.

    Parameters
    ----------
    model:
        An :class:`~qmlkit.ansatz.library.Ansatz`, a
        :class:`~qmlkit.core.ir.CircuitSpec`, or a model carrying one.
    n_samples, steps:
        The training set size and the number of optimiser steps. The product is
        how many gradients get taken.
    method:
        Which gradient rule to cost. ``"parameter-shift"`` is the default because
        it is what a device would use; pass ``"adjoint"`` to see the simulator cost.
    obs:
        The observable being measured. Its terms are grouped into qubit-wise
        commuting sets, because that grouping is the difference between a k-term
        observable costing k circuits and costing one.
    shots:
        Shots per circuit, or ``None`` for an exact simulator.

    Returns
    -------
    Plan
        Printable, and carrying the numbers so a caller can branch on them.
    """
    spec, default_obs = _as_spec(model)
    observable = obs if obs is not None else default_obs
    terms = len(as_sum(observable).terms)
    settings = max(len(group_qubit_wise_commuting(observable)), 1)

    cost = gradient_cost(spec, method)
    if isinstance(cost, str):  # a method whose cost is not a fixed circuit count
        raise ValueError(
            f"the {method!r} method reports its cost as {cost!r}, so it cannot be planned"
        )

    per_gradient = int(cost) * settings
    circuits = per_gradient * n_samples * steps
    notes: list[str] = []

    reductions: list[Reduction] = []
    for alternative in ("adjoint", "hadamard", "parameter-shift", "spsa"):
        if alternative == method:
            continue
        try:
            other = gradient_cost(spec, alternative)
        except (KeyError, ValueError):  # pragma: no cover - registry dependent
            continue
        if isinstance(other, str):
            continue
        total = int(other) * settings * n_samples * steps
        if total < circuits:
            reductions.append(
                Reduction(
                    alternative,
                    total,
                    circuits / total,
                    _METHOD_NOTES.get(alternative, ""),
                )
            )
    if terms > settings:
        notes.append(
            f"qubit-wise-commuting grouping is already saving {terms / settings:.1f}x "
            f"({terms} terms in {settings} settings)"
        )
    if spec.n_params == 0:
        notes.append("this circuit has no parameters, so a gradient costs nothing to take")

    return Plan(
        circuits=circuits,
        shots_total=circuits * shots if shots else None,
        method=method,
        n_params=spec.n_params,
        n_samples=n_samples,
        steps=steps,
        shots=shots,
        measurement_settings=settings,
        observable_terms=terms,
        reductions=tuple(sorted(reductions, key=lambda r: r.circuits)),
        notes=tuple(notes),
    )
