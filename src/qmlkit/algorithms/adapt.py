r"""ADAPT-VQE — grow the ansatz instead of guessing it.

A fixed ansatz is a bet placed before you have seen the Hamiltonian. ADAPT-VQE
(Grimsley et al. 2019) makes the circuit itself part of the optimisation: keep a pool
of candidate generators, and at each iteration append the one whose gradient is
largest, then re-optimise everything.

The gradient of appending :math:`e^{-i\theta P/2}` to the current state, evaluated at
:math:`\theta = 0`, is

.. math::  \left.\frac{\partial E}{\partial\theta}\right|_0 = -i\langle\psi|[H, P]|\psi\rangle

so ranking the pool costs one commutator expectation per candidate — no re-training
to find out which operator would have helped.

This is the algorithm that most depends on a circuit being *data*: growing an ansatz
mid-optimisation is a list append here, not a rebuild.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.algorithms.hamiltonians import exact_ground_energy
from qmlkit.algorithms.vqe import OPTIMIZERS, Optimizer
from qmlkit.ansatz.blocks import BuildContext, Custom
from qmlkit.ansatz.library import Ansatz
from qmlkit.core.builder import QCircuit
from qmlkit.core.execute import BackendLike, expectation
from qmlkit.core.observables import Observable, PauliString, PauliSum, as_sum

__all__ = ["AdaptVQE", "AdaptResult", "pauli_rotation", "default_operator_pool"]


def pauli_rotation(qc: QCircuit, term: PauliString, angle: Any) -> None:
    r"""Emit :math:`e^{-i\theta P/2}` for an arbitrary Pauli string ``P``.

    The standard construction: rotate each wire into the Z basis, run a CX ladder to
    collect the parity onto one wire, apply a single ``rz``, then undo both. Built
    from ordinary registered gates, so it runs on every backend.
    """
    wires = [q for q, p in term.paulis if p != "I"]
    if not wires:
        return
    for qubit, pauli in term.paulis:
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.sdg(qubit)
            qc.h(qubit)
    for a, b in zip(wires, wires[1:], strict=False):
        qc.cx(a, b)
    qc.rz(wires[-1], angle)
    for a, b in reversed(list(zip(wires, wires[1:], strict=False))):
        qc.cx(a, b)
    for qubit, pauli in term.paulis:
        if pauli == "X":
            qc.h(qubit)
        elif pauli == "Y":
            qc.h(qubit)
            qc.s(qubit)


def default_operator_pool(n_qubits: int) -> list[PauliString]:
    """Single-qubit ``Y`` and neighbouring ``YZ`` generators.

    Deliberately all imaginary-valued generators: those are the ones that move a real
    starting state, which is what a real-amplitude ground state needs. The pool is an
    argument, so a chemistry-flavoured (UCCSD-style) pool drops straight in.
    """
    pool = [PauliString(((q, "Y"),), 1.0) for q in range(n_qubits)]
    pool += [PauliString(((q, "Y"), (q + 1, "Z")), 1.0) for q in range(n_qubits - 1)]
    pool += [PauliString(((q, "Y"), (q + 1, "X")), 1.0) for q in range(n_qubits - 1)]
    return pool


@dataclass
class AdaptResult:
    energy: float
    theta: npt.NDArray[Any]
    operators: list[PauliString] = field(default_factory=list)
    history: list[float] = field(default_factory=list)
    gradients: list[float] = field(default_factory=list)
    exact: float | None = None

    @property
    def error_vs_exact(self) -> float | None:
        return None if self.exact is None else abs(self.energy - self.exact)

    @property
    def n_operators(self) -> int:
        return len(self.operators)

    def __repr__(self) -> str:
        tail = "" if self.exact is None else f", error={self.error_vs_exact:.2e}"
        return f"AdaptResult(energy={self.energy:.8f}, operators={self.n_operators}{tail})"


class AdaptVQE:
    """Build the ansatz one operator at a time, largest gradient first."""

    def __init__(
        self,
        hamiltonian: Observable,
        n_qubits: int,
        pool: Sequence[PauliString] | None = None,
        optimizer: str | Optimizer = "gradient-descent",
        backend: BackendLike = None,
        reference: Sequence[int] | None = None,
    ) -> None:
        self.hamiltonian = hamiltonian
        self.n_qubits = n_qubits
        self.pool = list(pool) if pool is not None else default_operator_pool(n_qubits)
        self.optimizer = optimizer
        self.backend = backend
        #: the starting state, as bits. Hartree-Fock plays this role in chemistry.
        self.reference = list(reference) if reference is not None else []

    # -------------------------------------------------------------- machinery --
    def _ansatz(self, operators: Sequence[PauliString]) -> Ansatz:
        """The circuit built from the operators chosen so far — a list, not a class."""
        reference = self.reference

        def build(qc: QCircuit, ctx: BuildContext) -> None:
            for wire in reference:
                qc.x(wire)
            for term in operators:
                pauli_rotation(qc, term, ctx.new_param())

        return Ansatz(self.n_qubits, Custom(build, "adapt"), "adapt")

    def _commutator_gradient(
        self, operators: Sequence[PauliString], theta: npt.NDArray[Any], candidate: PauliString
    ) -> float:
        r"""``|<psi|[H, P]|psi>|`` — how much appending ``P`` would move the energy."""
        commutator = _commutator(self.hamiltonian, candidate)
        if not commutator.terms:
            return 0.0
        spec = self._ansatz(operators).build(theta) if len(operators) else self._ansatz([]).build()
        return abs(float(expectation(spec, commutator, backend=self.backend)))

    # -------------------------------------------------------------------- run --
    def run(
        self,
        max_operators: int = 8,
        gradient_tol: float = 1e-3,
        seed: int | None = None,
        compare_exact: bool | None = None,
        **optimizer_kwargs: Any,
    ) -> AdaptResult:
        operators: list[PauliString] = []
        theta = np.zeros(0)
        history: list[float] = []
        picked_gradients: list[float] = []

        base = self._ansatz([]).build()
        history.append(float(expectation(base, self.hamiltonian, backend=self.backend)))

        for _ in range(max_operators):
            scores = [self._commutator_gradient(operators, theta, p) for p in self.pool]
            best = int(np.argmax(scores))
            if scores[best] < gradient_tol:
                break  # nothing left in the pool moves the energy
            operators.append(self.pool[best])
            picked_gradients.append(float(scores[best]))

            ansatz = self._ansatz(operators)
            spec = ansatz.build()
            start = np.concatenate([theta, [0.0]])  # a new operator starts at identity

            def energy(t: Sequence[float], spec: Any = spec) -> float:
                return float(
                    expectation(
                        spec, self.hamiltonian, theta=np.asarray(t, float), backend=self.backend
                    )
                )

            def gradient(t: Sequence[float], spec: Any = spec) -> npt.NDArray[Any]:
                from qmlkit.gradients.dispatch import grad

                return grad(spec, np.asarray(t, float), self.hamiltonian, backend=self.backend)

            fn = OPTIMIZERS[self.optimizer] if isinstance(self.optimizer, str) else self.optimizer
            kwargs = dict(optimizer_kwargs)
            if fn is OPTIMIZERS["gradient-descent"]:
                kwargs.setdefault("grad", gradient)
                kwargs.setdefault("n_steps", 60)
                kwargs.setdefault("lr", 0.2)
            if fn is OPTIMIZERS["spsa"]:
                kwargs.setdefault("seed", seed)
            theta, run_history = fn(energy, start, **kwargs)
            history.append(float(run_history[-1]))

        if compare_exact is None:
            compare_exact = self.n_qubits <= 12
        exact = exact_ground_energy(self.hamiltonian, self.n_qubits) if compare_exact else None

        return AdaptResult(
            energy=history[-1],
            theta=theta,
            operators=operators,
            history=history,
            gradients=picked_gradients,
            exact=exact,
        )

    def __repr__(self) -> str:
        return f"AdaptVQE(n_qubits={self.n_qubits}, pool={len(self.pool)} operators)"


# --------------------------------------------------------------------------- #
def _multiply(a: PauliString, b: PauliString) -> tuple[complex, PauliString]:
    """Product of two Pauli strings, as ``(phase, string)``."""
    table = {
        ("X", "Y"): (1j, "Z"),
        ("Y", "X"): (-1j, "Z"),
        ("Y", "Z"): (1j, "X"),
        ("Z", "Y"): (-1j, "X"),
        ("Z", "X"): (1j, "Y"),
        ("X", "Z"): (-1j, "Y"),
    }
    left, right = dict(a.paulis), dict(b.paulis)
    phase = complex(a.coeff) * complex(b.coeff)
    out: dict[int, str] = {}
    for qubit in set(left) | set(right):
        p, q = left.get(qubit), right.get(qubit)
        if p is None or q is None:
            out[qubit] = p or q  # type: ignore[assignment]
        elif p == q:
            continue  # P^2 = I
        else:
            factor, letter = table[(p, q)]
            phase *= factor
            out[qubit] = letter
    return phase, PauliString(tuple(sorted(out.items())), 1.0)


def _commutator(h: Observable, p: PauliString) -> PauliSum:
    r"""``i[H, P]`` — the *Hermitian* combination, which is what can be measured.

    ``[H, P]`` with both operands Hermitian is anti-Hermitian, so every coefficient it
    produces is purely imaginary and its expectation value is imaginary too. Taking
    the real part of that discards the whole thing: an early version of this function
    did exactly that, and ADAPT-VQE then scored every candidate at zero and grew an
    empty circuit. Multiplying by ``i`` first is what makes the result an observable.
    """
    collected: dict[tuple[tuple[int, str], ...], complex] = {}
    for term in as_sum(h).terms:
        for left, right, sign in ((term, p, 1.0), (p, term, -1.0)):
            phase, product = _multiply(left, right)
            collected[product.paulis] = collected.get(product.paulis, 0j) + sign * phase
    terms = [
        PauliString(paulis, float(np.real(1j * coeff)))
        for paulis, coeff in collected.items()
        if abs(np.real(1j * coeff)) > 1e-12
    ]
    return PauliSum(tuple(terms))
