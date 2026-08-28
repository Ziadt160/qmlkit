"""Mixed-state backends, where the noise model is the point.

A noisy backend is a normal backend with one primitive swapped: it has a density
matrix instead of a statevector. Everything above that - basis rotation, qubit-wise
commuting groups, sampling, expectation values - is inherited unchanged, so a noisy
run and an exact one differ in the number they produce and in nothing else.

**Two capability flags do the work, and they are not the same flag.**

``supports_statevector`` is false: there is no pure state, so ``adjoint`` and
``backprop`` refuse rather than quietly differentiating a noiseless circuit and
handing back a machine-precision number to someone who asked about a noisy one.
Parameter-shift keeps working, because a shift rule never inspects a state.

``supports_exact`` stays **true**, which is the part worth being deliberate about.
A density-matrix simulator can give a shot-free expectation - exact *given the noise
model* - and that separates the two error sources that otherwise arrive together.
Studying decoherence with shot noise layered on top means never knowing which one
you are looking at. Ask for ``shots=N`` when you want both.

Nothing here selects a backend for you. Noise is an argument to a *named* backend,
so the choice of simulator is always written down in the code that used it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend
from qmlkit.core.ir import CircuitSpec


class NoisyBackend(Backend):
    """Base for backends that evolve a density matrix rather than a state.

    Subclasses implement :meth:`density_matrix`. Probabilities come from its
    diagonal, and every other semantic is the base class's.
    """

    #: no pure state exists, so state-based differentiation is refused, not approximated
    supports_statevector = False
    #: exact *given the noise model* - the density matrix is evolved, not sampled
    supports_exact = True

    # ------------------------------------------------------------ primitives --
    def density_matrix(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        """Final state as a ``(2**n, 2**n)`` density matrix, qubit 0 most significant."""
        raise NotImplementedError(f"the {self.name!r} backend must implement density_matrix()")

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        """Always refuses. A mixed state is not a vector.

        Returning the noiseless statevector here would be the plausible-wrong-number
        failure this library exists to refuse: it would run, agree with the exact
        backend to machine precision, and answer a question nobody asked.
        """
        raise NotImplementedError(
            f"the {self.name!r} backend evolves a density matrix and has no statevector. "
            "Use density_matrix(spec), or probabilities(spec); for gradients use "
            'method="parameter-shift", which is measurement-only.'
        )

    def probabilities(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        """Outcome probabilities: the real diagonal of the density matrix."""
        rho = np.asarray(self.density_matrix(spec))
        probs = np.real(np.diagonal(rho)).astype(float)
        # a physical rho has a non-negative diagonal summing to 1; simulators drift by
        # ~1e-16 and a negative probability would poison sampling downstream
        probs = np.clip(probs, 0.0, None)
        total = probs.sum()
        if total <= 0:  # pragma: no cover - defensive
            raise ValueError(f"the {self.name!r} backend returned a density matrix with zero trace")
        normalised: npt.NDArray[Any] = probs / total
        return normalised

    # ------------------------------------------------------------- utilities --
    def purity(self, spec: CircuitSpec) -> float:
        """``Tr(rho^2)`` - 1 for a pure state, ``1/2**n`` for the maximally mixed one.

        The cheapest single number that says how much the noise model actually did.
        """
        rho = np.asarray(self.density_matrix(spec))
        return float(np.real(np.trace(rho @ rho)))
