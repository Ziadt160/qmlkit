"""The variational quantum eigensolver.

VQE is a loop, not an architecture: prepare a parameterised state, measure an energy,
step downhill, repeat. Everything that makes one VQE differ from another — the
ansatz, the optimiser, the gradient method, the shot budget — is therefore an
argument, and the class itself is thin on purpose.

    from qmlkit.algorithms import VQE, ising_hamiltonian

    H = ising_hamiltonian(4, j=1.0, h=0.5)
    result = VQE(H, n_qubits=4).run(seed=0)
    print(result.energy, result.error_vs_exact)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.algorithms.hamiltonians import exact_ground_energy
from qmlkit.ansatz.library import Ansatz, hardware_efficient
from qmlkit.core.execute import BackendLike, expectation
from qmlkit.core.observables import Observable, observable_support

__all__ = ["VQE", "VQEResult", "OPTIMIZERS"]

#: ``fn(loss, theta0, **kwargs) -> (theta, history)`` — the shape every optimiser here
#: already has, so a custom one is a function rather than an adapter class.
Optimizer = Callable[..., "tuple[npt.NDArray[Any], list[float]]"]


def _rotosolve(loss, theta0, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.optim import minimize_rotosolve

    return minimize_rotosolve(loss, theta0, **kw)


def _spsa(loss, theta0, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.gradients.spsa import minimize_spsa

    return minimize_spsa(loss, theta0, **kw)


def _gradient_descent(loss, theta0, *, grad, n_steps=100, lr=0.1, **kw):  # type: ignore[no-untyped-def]
    theta = np.asarray(theta0, dtype=float).copy()
    history = [float(loss(theta))]
    for _ in range(n_steps):
        theta = theta - lr * grad(theta)
        history.append(float(loss(theta)))
    return theta, history


OPTIMIZERS: dict[str, Optimizer] = {
    "rotosolve": _rotosolve,
    "spsa": _spsa,
    "gradient-descent": _gradient_descent,
}


@dataclass
class VQEResult:
    """What a run produced, and how good it actually is."""

    energy: float
    theta: npt.NDArray[Any]
    history: list[float] = field(default_factory=list)
    exact: float | None = None
    n_evaluations: int = 0

    @property
    def error_vs_exact(self) -> float | None:
        """Absolute error against dense diagonalisation, when that was computed."""
        return None if self.exact is None else abs(self.energy - self.exact)

    def __repr__(self) -> str:
        tail = (
            ""
            if self.exact is None
            else f", exact={self.exact:.8f}, error={self.error_vs_exact:.2e}"
        )
        return f"VQEResult(energy={self.energy:.8f}{tail}, steps={len(self.history) - 1})"


class VQE:
    """Minimise ``<H>`` over a parameterised state.

    Parameters
    ----------
    hamiltonian
        Any observable. :mod:`qmlkit.algorithms.hamiltonians` has constructors.
    ansatz
        The trial state. Defaults to a hardware-efficient circuit wide enough for
        the Hamiltonian's support — replaceable with anything, including one you
        invented, because an ``Ansatz`` is the only contract.
    optimizer
        A name from :data:`OPTIMIZERS` or any ``fn(loss, theta0, **kw)`` returning
        ``(theta, history)``.
    gradient
        Passed through to :func:`qmlkit.grad`. Only consulted by gradient-based
        optimisers; Rotosolve and SPSA never ask for one.
    """

    def __init__(
        self,
        hamiltonian: Observable,
        ansatz: Ansatz | None = None,
        n_qubits: int | None = None,
        optimizer: str | Optimizer = "rotosolve",
        gradient: str = "auto",
        backend: BackendLike = None,
        shots: int | None = None,
    ) -> None:
        support = observable_support(hamiltonian)
        width = n_qubits or (max(support) + 1 if support else 1)
        if ansatz is not None and ansatz.n_qubits < width:
            raise ValueError(
                f"the ansatz has {ansatz.n_qubits} qubits but the Hamiltonian acts on {width}"
            )
        self.hamiltonian = hamiltonian
        self.ansatz = ansatz or hardware_efficient(width, n_layers=2)
        self.n_qubits = self.ansatz.n_qubits
        self.optimizer = optimizer
        self.gradient = gradient
        self.backend = backend
        self.shots = shots
        self._spec = self.ansatz.build()
        self.n_evaluations = 0

    # ------------------------------------------------------------------ energy --
    def energy(self, theta: Sequence[float]) -> float:
        """``<H>`` at these parameters."""
        self.n_evaluations += 1
        value = expectation(
            self._spec,
            self.hamiltonian,
            theta=np.asarray(theta, dtype=float),
            shots=self.shots,
            backend=self.backend,
        )
        return float(value)

    def gradient_of_energy(self, theta: Sequence[float]) -> npt.NDArray[Any]:
        from qmlkit.gradients.dispatch import grad

        return grad(
            self._spec,
            np.asarray(theta, dtype=float),
            self.hamiltonian,
            method=self.gradient,
            backend=self.backend,
            shots=self.shots,
        )

    # --------------------------------------------------------------------- run --
    def run(
        self,
        theta0: Sequence[float] | None = None,
        seed: int | None = None,
        compare_exact: bool | None = None,
        **optimizer_kwargs: Any,
    ) -> VQEResult:
        """Optimise, and report against the exact answer when that is affordable."""
        start = (
            np.asarray(theta0, dtype=float)
            if theta0 is not None
            else self.ansatz.init("small", seed=seed)
        )
        fn = OPTIMIZERS[self.optimizer] if isinstance(self.optimizer, str) else self.optimizer
        if fn is _gradient_descent:
            optimizer_kwargs.setdefault("grad", self.gradient_of_energy)
        if fn is _spsa:
            optimizer_kwargs.setdefault("seed", seed)

        self.n_evaluations = 0
        theta, history = fn(self.energy, start, **optimizer_kwargs)

        # dense diagonalisation is exponential, so only offer it where it is cheap
        if compare_exact is None:
            compare_exact = self.n_qubits <= 12
        exact = exact_ground_energy(self.hamiltonian, self.n_qubits) if compare_exact else None

        return VQEResult(
            energy=float(history[-1]),
            theta=theta,
            history=list(history),
            exact=exact,
            n_evaluations=self.n_evaluations,
        )

    def __repr__(self) -> str:
        return (
            f"VQE(n_qubits={self.n_qubits}, ansatz={self.ansatz.name!r}, "
            f"P={self.ansatz.n_weights}, optimizer={self.optimizer!r})"
        )
