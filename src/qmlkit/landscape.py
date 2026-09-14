r"""Why is it not training? — gradients, curvature, minima, overparametrisation.

A variational model that will not learn gives you one number, the loss, and it stays
flat for four different reasons that a loss curve cannot tell apart:

* the gradient is **exponentially small everywhere** — a barren plateau, and no
  initialisation escapes it;
* the gradient is fine but the optimiser has **settled into a spurious minimum**,
  which a different start would have missed;
* the parameters are **dead or unmeasurable**, so the flat loss is correct and the
  model has fewer knobs than it claims;
* the model is **underparametrised** — the minimum it reached is the best its
  parameter space contains.

Each has a different fix, and three of the four are cured by something that makes the
fourth worse. :func:`landscape` measures all four and names which one you have.

The four mechanisms behind a plateau
------------------------------------
"Barren plateau" is one name for four distinct phenomena with four distinct cures,
and :func:`plateau_mechanism` separates them by measurement rather than by assertion:

============================ ======================================================
Mechanism                    What it is, and what actually helps
============================ ======================================================
**Cost globality**           A global observable flattens at ``O(1)`` depth while a
                             local one survives to ``O(log n)`` (Cerezo et al. 2021).
                             Measure something local.
**Expressibility / depth**   Approaching a 2-design drives the variance to the
                             ``2^-n`` scale (McClean et al. 2018). Reduce depth, or
                             initialise near identity.
**Entanglement**             Volume-law entanglement with the unmeasured wires flattens
                             the visible gradient (Ortiz Marrero et al. 2021).
                             Restrict the entangler.
**Noise**                    Decoherence contracts the whole landscape towards the
                             maximally mixed state (Wang et al. 2021). Nothing in the
                             ansatz fixes it; this one is a hardware budget.
============================ ======================================================

The first three are properties of a circuit you chose and can choose differently. The
fourth is not, which is why it is reported separately and never with an ansatz fix.

Overparametrisation, and what it actually buys
----------------------------------------------
The rank of the quantum Fisher information counts the directions in parameter space
that move the state at all. It grows with depth and then stops, capped by the
dimension of the circuit's dynamical Lie algebra; past that point extra parameters
add only flat directions, and the landscape is reported to lose its spurious minima
(Larocca et al. 2023). :func:`overparametrisation` finds where the rank saturates.

Measured here, on ``hardware_efficient``, the saturated rank is exactly ``2 * 2^n - 2``
— the real dimension of projective Hilbert space — reached at 8 parameters on 2 qubits
and 18 on 3. The threshold predicts *reachability* very sharply: on a 2-qubit
Hamiltonian, the ansatz one layer below saturation converged from every start to
-1.65803 against a true ground state of -1.77652, and the first ansatz at saturated
rank hit the ground state exactly.

It predicts the minima too, and on a frustrated 4-qubit Heisenberg model it predicted
them exactly: the rank saturates at 30 of 32 parameters, first reached at 4 layers,
and 4 layers is the first depth at which 16 random starts all land in one basin — 1, 2
and 3 layers each leave two, spreading 3.78, 0.11 and 0.11 in final loss. The
threshold and the crossover are the same layer.

One run of 16 starts at one width is not a proof, which is why :func:`minima_scan`
reports the spread rather than a verdict: the sharp statement in the literature is
asymptotic, and the model in front of you is not.

This is the trade the whole module exists to show: parameters bought to flatten the
landscape are paid for in trainability, because the same depth that saturates the rank
pushes the circuit towards a 2-design.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.ansatz.library import Ansatz
from qmlkit.core.execute import BackendLike
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z, iter_terms, observable_support
from qmlkit.diagnostics import Finding, Report, _sorted
from qmlkit.gradients.batch import grad_batch
from qmlkit.utils.shots import shots_for_precision

__all__ = [
    "GradientStats",
    "gradient_stats",
    "plateau_mechanism",
    "HessianSpectrum",
    "hessian_spectrum",
    "MinimaScan",
    "minima_scan",
    "Overparametrisation",
    "overparametrisation",
    "LandscapeReport",
    "landscape",
]

# A gradient variance below this is machine zero, not a small number: the parameter
# does not move the observable at all. Reporting it as a plateau is the single most
# common way this kind of tool lies, so it is excluded everywhere rather than clipped.
_DEAD = 1e-24

# Below this, resolving one gradient entry against shot noise costs more than any
# realistic budget. Matches `_FLAT` in diagnostics, deliberately.
_FLAT = 1e-3

# Eigenvalues of the Hessian within this fraction of the largest are "flat": they are
# the directions a minimum is free to move along, and counting them is how a wide
# minimum is told from a sharp one.
_FLAT_CURVATURE = 1e-6

# Two final losses within this *relative* gap came from the same basin. Relative, not
# absolute, because the tolerance has to mean the same thing whether the loss is a Z
# expectation in [-1, 1] or a Hamiltonian around -6: measured on a frustrated 4-qubit
# Heisenberg model, an absolute 1e-3 split one minimum at -6.1154 into four, purely on
# the residual Adam had not yet worked off.
_SAME_BASIN = 1e-3

# A controlled comparison has to beat this ratio before it is called a mechanism.
# Measured on a 1-layer hardware-efficient ansatz, a global Z^n against a local Z(0)
# separates by 2.0x at 3 qubits and 7.7x at 7 -- the effect is unambiguous well before
# it is large, so a threshold of 10 would stay silent at every width you can simulate.
# Two is above the +-14% a 100-sample variance carries, which is what makes it safe.
_SEPARATES = 2.0

# A point counts as stationary when the outstanding Newton step, |g| / lambda_max, is
# under this many radians. Relative because an absolute cut means different things
# under different observables; 1e-3 rad is far below the separation between any two
# minima a scan could tell apart.
_STATIONARY = 1e-3

# Adam at a fixed step does not settle: measured on a 4-qubit re-uploading model, five
# of six runs sat at |g| ~ 1e-2 after 3,000 steps while their losses had been stable
# since step 400 -- oscillation in a basin, not a slope. Running the same budget over
# these decreasing rates took |g| to ~1e-4 and left the losses unchanged.
_LR_DECAY = (1.0, 0.2, 0.04)

# Expressibility KL below this reads as "close to Haar", the 2-design regime.
_NEAR_HAAR = 0.1

# Meyer-Wallach Q above this is near-maximal entanglement.
_NEAR_MAXIMAL = 0.9


# --------------------------------------------------------------------------- #
# gradient statistics
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GradientStats:
    r"""Every parameter's gradient statistics, from one batch of random points.

    :func:`~qmlkit.metrics.gradient_variance` probes one parameter and throws away the
    other ``p - 1`` entries of every gradient it computes. They cost nothing extra —
    one adjoint pass returns the whole vector — and they are what distinguishes a
    barren plateau from a single dead parameter, so this keeps them.

    ``variance`` and ``mean`` are per parameter, over ``n_samples`` draws of
    ``theta ~ U(-pi, pi)``.
    """

    n_qubits: int
    n_params: int
    n_samples: int
    observable: str
    variance: npt.NDArray[Any]
    mean: npt.NDArray[Any]

    @property
    def silent(self) -> npt.NDArray[Any]:
        """Indices whose gradient is identically zero — no gradient, not a small one.

        Silent is not the same as dead, and the difference decides the fix. A *dead*
        parameter cannot change the state at all; a merely *unmeasurable* one changes
        it and cannot change **this observable**, so measuring something else revives
        it. Both are silent here, because both have a gradient of exactly zero;
        :func:`landscape` separates them.
        """
        return np.flatnonzero(self.variance < _DEAD)

    @property
    def live(self) -> npt.NDArray[Any]:
        """Indices that actually move the observable."""
        return np.flatnonzero(self.variance >= _DEAD)

    @property
    def typical_variance(self) -> float:
        """Median variance over live parameters — the number the literature quotes.

        The median rather than the mean, because one well-placed parameter in an
        otherwise flat circuit drags a mean upwards and reads as trainability the
        optimiser cannot reach.
        """
        live = self.variance[self.live]
        return float(np.median(live)) if live.size else 0.0

    @property
    def best_variance(self) -> float:
        """The largest live variance — the most trainable single direction there is.

        A plateau is a statement about *every* direction, so this is the one number
        that can refute it: if it is healthy, some descent direction is resolvable,
        whatever the median says.
        """
        live = self.variance[self.live]
        return float(np.max(live)) if live.size else 0.0

    @property
    def relative_error(self) -> float:
        r"""Relative standard error on each variance, ``sqrt(2 / (N - 1))``.

        A sample variance is itself a random variable, and at the default 100 draws it
        is only good to about 14%. Quoting a variance to three figures off 30 samples
        is the mistake this property exists to make visible.
        """
        return float(np.sqrt(2.0 / max(self.n_samples - 1, 1)))

    @property
    def shots_needed(self) -> int:
        """Shots to resolve a typical gradient entry against sampling noise."""
        typical = self.typical_variance
        return shots_for_precision(float(np.sqrt(typical))) if typical > 0 else 0

    def __str__(self) -> str:
        pct = 100 * self.relative_error
        silent = self.silent.size
        lines = [
            f"gradient statistics on {self.n_qubits} qubits, observable {self.observable}",
            f"  parameters          {self.n_params} ({self.live.size} live, {silent} silent)",
            f"  samples             {self.n_samples}  (variances good to +-{pct:.0f}%)",
            f"  typical variance    {self.typical_variance:.3e}   (median over live)",
            f"  best variance       {self.best_variance:.3e}   (most trainable direction)",
            f"  shots for a typical gradient entry  {self.shots_needed:,}",
        ]
        return "\n".join(lines)


def _observable_label(obs: Observable) -> str:
    text = repr(obs)
    return text if len(text) <= 40 else text[:37] + "..."


def _locality(obs: Observable) -> int:
    """Largest Pauli weight in the observable — 1 is local, ``n`` is global."""
    weights = [len(term.support()) for term in iter_terms(obs)]
    return max(weights) if weights else 0


def _random_thetas(n: int, p: int, seed: int | None) -> npt.NDArray[Any]:
    return np.random.default_rng(seed).uniform(-np.pi, np.pi, (n, p))


def gradient_stats(
    ansatz: Ansatz,
    obs: Observable | None = None,
    *,
    n_samples: int = 100,
    seed: int | None = 0,
    backend: BackendLike = None,
) -> GradientStats:
    """Gradient mean and variance for **every** parameter, in one batched pass.

    Every parameter means every *slot*: on an ansatz that reserves input slots, those
    are drawn from the same uniform range as the weights, so the variance is over
    random data and random weights together. That is the ensemble the barren-plateau
    literature uses and the one
    :func:`~qmlkit.metrics.gradient_variance` already used, but it is not the variance
    at a fixed data point — for that, look at
    :func:`~qmlkit.diagnostics.diagnose`, which probes a model with its encoding held
    in front.

    Examples
    --------
    >>> import qmlkit as qk
    >>> s = qk.gradient_stats(qk.hardware_efficient(3, 2), n_samples=40)
    >>> s.n_params
    12
    >>> s.best_variance > 0
    True
    """
    obs = Z(0) if obs is None else obs
    spec = ansatz.build()
    thetas = _random_thetas(n_samples, ansatz.n_params, seed)
    grads = np.asarray(grad_batch(spec, thetas, obs, backend=backend), dtype=float)
    return GradientStats(
        n_qubits=ansatz.n_qubits,
        n_params=ansatz.n_params,
        n_samples=n_samples,
        observable=_observable_label(obs),
        variance=np.var(grads, axis=0),
        mean=np.mean(grads, axis=0),
    )


# --------------------------------------------------------------------------- #
# which plateau is it
# --------------------------------------------------------------------------- #
def plateau_mechanism(
    ansatz: Ansatz,
    obs: Observable | None = None,
    *,
    n_samples: int = 100,
    seed: int | None = 0,
    backend: BackendLike = None,
) -> list[Finding]:
    """Attribute a flat gradient to a cause, by controlled comparison.

    Two of the four mechanisms can be tested directly, by re-measuring the *same*
    circuit with one thing changed — the observable made local, or the noise model
    removed. Those run always, because the comparison is evidence at any width: a
    model whose gradient quadruples when you measure ``Z(0)`` instead has the
    globality problem at 6 qubits, where you can still simulate it, and will have it
    much worse at 20.

    The other two — expressibility and entanglement — are correlations rather than
    controlled experiments, so they are only reported when there is a genuinely small
    gradient for them to explain. Firing them on every deep circuit would be noise,
    and a diagnostic people learn to ignore is worse than no diagnostic.

    Severity follows the absolute number: ``warning`` when the gradient is already
    below the resolvable scale, ``info`` when the mechanism is present but the model
    is still trainable at this width.
    """
    from qmlkit.core.backends.registry import get_backend
    from qmlkit.metrics import entangling_capability, expressibility

    obs = Z(0) if obs is None else obs
    found: list[Finding] = []
    stats = gradient_stats(ansatz, obs, n_samples=n_samples, seed=seed, backend=backend)
    typical = stats.typical_variance
    if typical == 0.0:
        return found  # every parameter is silent; that is a different report

    n = ansatz.n_qubits
    exact = get_backend(backend).supports_statevector
    flat = typical < _FLAT
    severity = "warning" if flat else "info"

    # -- noise first: it is the only mechanism no ansatz edit can undo, and leaving it
    # -- unseparated makes every other number below a mixture of two effects.
    if not exact:
        clean = gradient_stats(ansatz, obs, n_samples=n_samples, seed=seed, backend=None)
        ratio = clean.typical_variance / typical
        if ratio > _SEPARATES:
            found.append(
                Finding(
                    "PLATEAU_NOISE",
                    severity,
                    f"gradient variance is {typical:.2e} on this backend and "
                    f"{clean.typical_variance:.2e} on an exact one -- {ratio:.1f}x larger "
                    "without the noise model. That much of the flatness is decoherence "
                    "contracting the state towards maximally mixed, not the ansatz.",
                    fix="No ansatz change recovers this. Reduce circuit depth so fewer gates "
                    "accumulate error, or accept it as the hardware's budget.",
                    value=float(ratio),
                )
            )

    # -- cost globality: re-measure the identical circuit against the most local
    # -- observable there is. If the gradient comes back, the observable was the problem.
    k = _locality(obs)
    if k > 1:
        local = gradient_stats(ansatz, Z(0), n_samples=n_samples, seed=seed, backend=backend)
        ratio = local.typical_variance / typical
        if ratio > _SEPARATES:
            found.append(
                Finding(
                    "PLATEAU_COST_GLOBALITY",
                    severity,
                    f"the observable acts on {k} qubits at once, and that is what is flat: "
                    f"variance {typical:.2e} for it against {local.typical_variance:.2e} for "
                    f"Z(0) on the same circuit, {ratio:.1f}x. A global cost flattens at "
                    "constant depth while a local one survives to O(log n) of it, so this gap "
                    "widens exponentially with every qubit you add.",
                    fix="Measure a local observable and sum the terms -- "
                    "sum(Z(i) for i in range(n)) rather than a product over every wire.",
                    value=float(ratio),
                )
            )

    # -- The remaining two are correlational. They only speak when something is flat.
    if flat and exact and n > 1:
        kl = expressibility(ansatz, n_samples=max(100, n_samples), seed=seed, backend=backend)
        if kl < _NEAR_HAAR:
            found.append(
                Finding(
                    "PLATEAU_EXPRESSIBILITY",
                    "info",
                    f"the state distribution is {kl:.3f} in KL from Haar, so this circuit is "
                    "close to a 2-design -- the regime in which gradient variance is driven to "
                    f"the 2^-n scale ({2.0**-n:.1e} at this width, measured {typical:.1e}). "
                    "Expressibility and trainability are the same knob pulled in opposite "
                    "directions.",
                    fix="Reduce the number of layers, or start from ansatz.init('small'), "
                    "which keeps the circuit near identity and away from this regime.",
                    value=float(kl),
                )
            )

        q = entangling_capability(
            ansatz, n_samples=max(50, n_samples // 2), seed=seed, backend=backend
        )
        if q > _NEAR_MAXIMAL:
            found.append(
                Finding(
                    "PLATEAU_ENTANGLEMENT",
                    "info",
                    f"Meyer-Wallach Q is {q:.3f}, near the maximum: every qubit is close to "
                    "maximally entangled with the rest, so the reduced state on the measured "
                    "wire is close to maximally mixed and its gradient is correspondingly "
                    "flat.",
                    fix="Use a sparser entangler -- 'linear' rather than 'full', or fewer "
                    "entangling layers.",
                    value=float(q),
                )
            )

    if flat and not found:
        found.append(
            Finding(
                "PLATEAU_UNATTRIBUTED",
                "info",
                f"gradient variance is {typical:.2e}, which is small, but none of the four "
                "known mechanisms separates on this circuit: the observable is local, the "
                "circuit is not near a 2-design, entanglement is moderate, and the backend is "
                "exact. A small gradient is not yet a barren plateau -- that is a claim about "
                "scaling, and this is one width.",
                fix="Run metrics.barren_plateau_scan across widths. A variance that is merely "
                "small at one width and constant across them is not a plateau.",
                value=float(typical),
            )
        )
    return found


# --------------------------------------------------------------------------- #
# curvature at a point
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HessianSpectrum:
    """What kind of stationary point this is, from the curvature around it."""

    eigenvalues: npt.NDArray[Any]
    gradient_norm: float

    @property
    def scale(self) -> float:
        return float(np.max(np.abs(self.eigenvalues))) if self.eigenvalues.size else 0.0

    @property
    def _tol(self) -> float:
        return max(self.scale * _FLAT_CURVATURE, 1e-9)

    @property
    def index(self) -> int:
        """Number of descent directions — 0 at a minimum, more at a saddle."""
        return int(np.sum(self.eigenvalues < -self._tol))

    @property
    def n_flat(self) -> int:
        """Directions the loss does not curve along at all.

        At a minimum these are the free parameters of a *manifold* of equally good
        solutions, and their count is the overparametrisation signature: a model with
        more parameters than its dynamical Lie algebra has room for spends the surplus
        here.
        """
        return int(np.sum(np.abs(self.eigenvalues) <= self._tol))

    @property
    def is_stationary(self) -> bool:
        r"""Is the remaining descent step negligible?

        Relative to the curvature, not absolute. ``|g| / lambda_max`` is the size of
        the Newton step still outstanding, in radians, and that is the quantity with a
        scale-free meaning: an absolute cut on ``|g|`` calls the same point converged
        under ``Z(0)`` and unconverged under ``sum_i Z(i)``, purely because the second
        observable is four times larger. Measured on a 4-qubit re-uploading model,
        endpoints sat at ``|g| = 3.8e-4`` against a curvature scale of 4.6 — a
        remaining step of 8e-5 radians, and a minimum by any useful reading.
        """
        return self.gradient_norm <= _STATIONARY * max(self.scale, 1.0)

    @property
    def kind(self) -> str:
        """``minimum``, ``saddle``, ``maximum``, or ``not-stationary``."""
        if not self.is_stationary:
            return "not-stationary"
        if self.index == 0:
            return "minimum"
        if self.index == int(np.sum(np.abs(self.eigenvalues) > self._tol)):
            return "maximum"
        return "saddle"

    def __str__(self) -> str:
        return (
            f"{self.kind} (|grad| {self.gradient_norm:.2e}, {self.index} descent "
            f"direction(s), {self.n_flat} flat of {self.eigenvalues.size})"
        )


def hessian_spectrum(
    spec: CircuitSpec,
    theta: ArrayLike,
    obs: Observable | None = None,
    *,
    free: Sequence[int] | None = None,
    backend: BackendLike = None,
    eps: float = 1e-4,
) -> HessianSpectrum:
    """Classify the point ``theta``: minimum, saddle, maximum, or still on a slope.

    A converged run that stopped at a saddle and one that stopped at a genuine minimum
    produce the same flat loss curve, and the eigenvalues are what tell them apart. A
    negative eigenvalue is a direction the optimiser could still have gone down.

    ``free`` restricts the verdict to a subspace of the coordinates — the weights of a
    model whose input slots are held fixed, say. It matters: the gradient along a
    coordinate nobody optimised is not zero at the optimum, so measuring the full
    vector would call every converged point "not stationary".
    """
    from qmlkit.gradients.dispatch import grad, hessian

    obs = Z(0) if obs is None else obs
    arr = np.asarray(theta, dtype=float).ravel()
    h = hessian(spec, arr, obs, backend=backend, eps=eps)
    g = grad(spec, arr, obs, backend=backend)
    if free is not None:
        idx = np.asarray(list(free), dtype=int)
        h, g = h[np.ix_(idx, idx)], g[idx]
    return HessianSpectrum(
        eigenvalues=np.linalg.eigvalsh(h),
        gradient_norm=float(np.linalg.norm(g)),
    )


# --------------------------------------------------------------------------- #
# do random starts agree
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MinimaScan:
    """Where independent random starts ended up, and whether they agreed."""

    losses: npt.NDArray[Any]
    kinds: tuple[str, ...]
    n_steps: int
    best_theta: npt.NDArray[Any] = field(repr=False, default_factory=lambda: np.zeros(0))

    @property
    def best(self) -> float:
        return float(np.min(self.losses))

    @property
    def worst(self) -> float:
        return float(np.max(self.losses))

    @property
    def spread(self) -> float:
        return self.worst - self.best

    @property
    def tolerance(self) -> float:
        """Absolute gap below which two final losses count as the same minimum."""
        scale = float(np.max(np.abs(self.losses))) if self.losses.size else 1.0
        return _SAME_BASIN * max(scale, 1.0)

    @property
    def n_basins(self) -> int:
        """Distinct final losses — 1 means every start found the same value."""
        ordered = np.sort(self.losses)
        return 1 + int(np.sum(np.diff(ordered) > self.tolerance)) if ordered.size else 0

    @property
    def reached_best(self) -> float:
        """Fraction of starts that reached the best value found."""
        if not self.losses.size:
            return 0.0
        return float(np.mean(self.losses <= self.best + self.tolerance))

    @property
    def benign(self) -> bool:
        """True when every start agreed — no spurious minimum was found.

        Absence of evidence: a scan that agrees has not proven the landscape has no
        spurious minima, only that this many starts did not find one.
        """
        return self.n_basins == 1

    @property
    def converged(self) -> bool | None:
        """Did every run finish descending? ``None`` when endpoints were not classified.

        This gates the verdict rather than decorating it. A run still on a slope has
        not reached a minimum, so its final loss is wherever the step budget ran out —
        and counting those as distinct basins would report spurious minima that are
        really an under-optimised scan. That is the false positive this whole module
        is built to avoid, so :func:`landscape` will not assert one without this.
        """
        if not self.kinds or "unclassified" in self.kinds:
            return None
        return "not-stationary" not in self.kinds

    @property
    def _endpoints(self) -> str:
        kinds = ", ".join(sorted(set(self.kinds)))
        if "not-stationary" in self.kinds:
            # Honest, and a common surprise: a run still on a slope has no curvature
            # verdict to give, and that is a budget problem rather than a saddle.
            kinds += "  (still descending -- raise n_steps to classify these)"
        return kinds

    def __str__(self) -> str:
        verdict = (
            "every start agreed -- no spurious minimum found"
            if self.benign
            else f"{self.n_basins} distinct final losses -- spurious minima"
        )
        return (
            f"{self.losses.size} random starts, {self.n_steps} steps each\n"
            f"  best / worst        {self.best:+.6f} / {self.worst:+.6f}"
            f"   (spread {self.spread:.2e})\n"
            f"  reached the best    {100 * self.reached_best:.0f}% of starts\n"
            f"  endpoints           {self._endpoints}\n"
            f"  verdict             {verdict}"
        )


def minima_scan(
    ansatz: Ansatz,
    obs: Observable | None = None,
    *,
    x: ArrayLike | None = None,
    n_starts: int = 8,
    n_steps: int = 300,
    lr: float = 0.1,
    seed: int | None = 0,
    backend: BackendLike = None,
    classify: bool = True,
) -> MinimaScan:
    """Minimise from ``n_starts`` random points and see whether they agree.

    This is the only honest way to ask about local minima on a landscape nobody can
    plot: run it repeatedly and look at the spread of what comes back. Starts are drawn
    uniformly rather than near identity, because a small initialisation is precisely
    the trick that hides the question.

    ``x`` is required when the ansatz reserves input slots, and holds them fixed while
    the weights move. Without it the optimiser would drive the data as if it were a
    free parameter, which silently answers a different question — the best this circuit
    can reach over its whole parameter space, rather than the training problem — so it
    raises instead of guessing. The landscape of a re-uploading model is a property of
    the model *and a point*, and there is no default point.

    ``n_steps`` is split over three decreasing learning rates rather than spent at one.
    Fixed-step Adam does not settle: measured on a 4-qubit re-uploading model, five of
    six runs still had ``|g| ~ 1e-2`` after 3,000 steps while their losses had not moved
    since step 400 — orbiting a basin, not descending a slope. The same budget decayed
    reaches ``|g| ~ 1e-4`` and leaves the losses where they were, which is what makes
    ``converged`` mean anything.

    ``classify`` computes a Hessian per endpoint to separate a genuine minimum from a
    saddle the optimiser stalled on. It costs ``p`` extra gradients per start; turn it
    off on wide circuits.
    """
    from qmlkit.core.execute import expectation
    from qmlkit.gradients.dispatch import grad
    from qmlkit.optim import minimize_adam

    obs = Z(0) if obs is None else obs
    spec = ansatz.build()

    fixed = np.zeros(0)
    if ansatz.n_inputs:
        if x is None:
            raise ValueError(
                f"{ansatz.name!r} reserves {ansatz.n_inputs} input slots, so its landscape "
                "depends on where in the data you stand and there is no default point. Pass "
                "x=<one sample> to hold the encoding fixed while the weights move. "
                "(Optimising the slots too would report the best this circuit can reach over "
                "every input, which is not the problem you are training.)"
            )
        fixed = np.ravel(ansatz.angles(x))
    free_indices = range(ansatz.n_inputs, ansatz.n_params)
    starts = _random_thetas(n_starts, ansatz.n_weights, seed)

    def full(weights: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.concatenate([fixed, weights]) if fixed.size else weights

    def loss(w: npt.NDArray[Any]) -> float:
        return float(expectation(spec.bind(full(w)), obs, backend=backend))

    def gradient(w: npt.NDArray[Any]) -> npt.NDArray[Any]:
        g = grad(spec, full(w), obs, backend=backend)
        return np.asarray(g[ansatz.n_inputs :], dtype=float)

    # The stages sum to exactly n_steps -- the last absorbs the remainder -- so the
    # number MinimaScan reports is the number that ran. Splitting with a floor instead
    # quietly ran 99 steps for a requested 100.
    base, remainder = divmod(n_steps, len(_LR_DECAY))
    budgets = [base] * len(_LR_DECAY)
    budgets[-1] += remainder

    losses, kinds, weights = [], [], []
    for theta0 in starts:
        theta, value = theta0, float(loss(theta0))
        # The budget is split over decreasing rates rather than spent at one: a run
        # that oscillates in a basin has the same loss as one that settled in it, and
        # only the settled one can be told from a saddle.
        for budget, factor in zip(budgets, _LR_DECAY, strict=True):
            theta, history = minimize_adam(loss, theta, gradient, n_steps=budget, lr=lr * factor)
            value = history[-1]
        losses.append(value)
        weights.append(theta)
        kinds.append(
            hessian_spectrum(spec, full(theta), obs, free=free_indices, backend=backend).kind
            if classify
            else "unclassified"
        )

    values = np.asarray(losses, dtype=float)
    return MinimaScan(
        losses=values,
        kinds=tuple(kinds),
        n_steps=n_steps,
        best_theta=np.asarray(full(weights[int(np.argmin(values))]), dtype=float),
    )


# --------------------------------------------------------------------------- #
# overparametrisation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Overparametrisation:
    """Where the QFIM rank stops growing, and what that means for the minima."""

    n_params: tuple[int, ...]
    ranks: tuple[int, ...]
    layers: tuple[int, ...]

    @property
    def saturated_rank(self) -> int:
        """The largest rank reached — an estimate of the reachable dimension."""
        return max(self.ranks) if self.ranks else 0

    @property
    def threshold(self) -> int | None:
        """Parameter count at which the rank first saturates, or None if it never did.

        None is the honest answer when the rank was still climbing at the widest point
        scanned: the threshold exists, and this scan did not reach it.
        """
        top = self.saturated_rank
        for p, r in zip(self.n_params, self.ranks, strict=True):
            if r >= top:
                return p
        return None

    @property
    def saturated(self) -> bool:
        """Did the rank actually stop growing inside the scanned range?"""
        return len(self.ranks) >= 2 and self.ranks[-1] == self.ranks[-2]

    def verdict(self, n_params: int) -> str:
        """``over``, ``under``, or ``unknown`` for a model of this parameter count."""
        if not self.saturated or self.threshold is None:
            return "unknown"
        return "over" if n_params >= self.threshold else "under"

    def __str__(self) -> str:
        rows = "\n".join(
            f"    {l:>3} layers  {p:>4} params   rank {r:>4}"
            for l, p, r in zip(self.layers, self.n_params, self.ranks, strict=True)
        )
        if self.saturated and self.threshold is not None:
            tail = (
                f"  rank saturates at {self.saturated_rank}, first reached with "
                f"{self.threshold} parameters.\n"
                "  At or above that count the surplus directions are flat rather than "
                "uphill,\n  which is the regime where spurious minima disappear."
            )
        else:
            tail = (
                f"  rank was still growing at {self.n_params[-1]} parameters "
                f"(reached {self.saturated_rank}).\n"
                "  The threshold is above this scan -- widen layer_range to find it."
            )
        return f"QFIM rank against parameter count\n{rows}\n{tail}"


def overparametrisation(
    ansatz_factory: Callable[[int], Ansatz],
    layer_range: Sequence[int],
    *,
    probes: int = 3,
    seed: int | None = 0,
    backend: BackendLike = None,
) -> Overparametrisation:
    """Find the overparametrisation threshold by watching the QFIM rank saturate.

    The rank of the quantum Fisher information counts the directions in parameter
    space that actually move the state. It grows with depth and then stops, capped by
    the dimension of the circuit's dynamical Lie algebra; past that point extra
    parameters can only add flat directions, and the landscape loses its spurious
    minima (Larocca et al. 2023).

    ``probes`` random points per width, taking the largest rank, because rank is
    generically maximal and drops only on a measure-zero set — one unlucky point
    would otherwise report a threshold that is not there.

    Examples
    --------
    >>> import qmlkit as qk
    >>> scan = qk.overparametrisation(lambda L: qk.hardware_efficient(2, L), [1, 2, 3])
    >>> scan.saturated_rank > 0
    True
    """
    from qmlkit.optim import quantum_fisher_information

    layers, params, ranks = [], [], []
    for n_layers in layer_range:
        ansatz = ansatz_factory(n_layers)
        spec = ansatz.build()
        thetas = _random_thetas(probes, ansatz.n_params, seed)
        best = 0
        for theta in thetas:
            qfim = quantum_fisher_information(spec, theta, backend=backend)
            # A relative tolerance, not the default absolute one: QFIM eigenvalues run
            # over several orders of magnitude and an absolute cut counts noise as rank.
            best = max(best, int(np.linalg.matrix_rank(qfim, tol=1e-8 * max(1.0, np.trace(qfim)))))
        layers.append(int(n_layers))
        params.append(int(ansatz.n_params))
        ranks.append(best)
    return Overparametrisation(n_params=tuple(params), ranks=tuple(ranks), layers=tuple(layers))


# --------------------------------------------------------------------------- #
# one call
# --------------------------------------------------------------------------- #
def _silent_findings(
    ansatz: Ansatz,
    stats: GradientStats,
    obs: Observable,
    *,
    seed: int | None,
    backend: BackendLike,
) -> list[Finding]:
    """Split the zero-gradient parameters into the two cases that need different fixes.

    A parameter with no gradient is either dead — it cannot change the state — or
    unmeasurable, meaning it changes the state and this particular observable cannot
    see it. The second is fixed by measuring something else and the first is not, so
    reporting them under one heading would send half the readers to the wrong edit.
    """
    from qmlkit.diagnostics import _dead_parameters, _structural_backend

    if not stats.silent.size:
        return []

    structural, _ = _structural_backend(backend)
    dead = set(int(i) for i in _dead_parameters(ansatz, probes=3, seed=seed, backend=structural))
    silent = [int(i) for i in stats.silent]
    unmeasurable = [i for i in silent if i not in dead]
    truly_dead = [i for i in silent if i in dead]

    def listing(idx: list[int]) -> str:
        return f"{idx[:8]}{'...' if len(idx) > 8 else ''}"

    found: list[Finding] = []
    if truly_dead:
        found.append(
            Finding(
                "DEAD_PARAMETERS",
                "warning",
                f"{len(truly_dead)} of {stats.n_params} parameters cannot change the state at "
                f"all, at any probe (indices {listing(truly_dead)}). The optimiser carries "
                "them every step and nothing downstream can depend on them.",
                fix="Usually a rotation whose generator already fixes the state it acts on -- "
                "an Rz on |0> is the standard case.",
                value=float(len(truly_dead)),
            )
        )
    if unmeasurable:
        # A narrow readout is the dominant cause and has its own fix, so say which one
        # this is. Measured against Z(0), 58-69% of parameters come back unmeasurable on
        # every stock ansatz -- that is the single-qubit readout, not the circuit, and
        # reporting it as an ansatz defect would be a finding that fires on everything.
        watched = len(observable_support(obs))
        narrow = watched < ansatz.n_qubits
        if narrow:
            cause = (
                f"The readout is the reason: {_observable_label(obs)} watches {watched} of "
                f"{ansatz.n_qubits} wires, and a rotation outside its light cone cannot move "
                "it whatever it does to the state."
            )
            fix = (
                f"Measure every wire -- sum(Z(i) for i in range({ansatz.n_qubits})) -- which "
                "is what a model with a trained readout effectively does. Measured on "
                "hardware_efficient(4, 2), that takes the count from 11 of 16 to 4 of 16."
            )
        else:
            cause = (
                "The observable already watches every wire, so this is the circuit rather "
                "than the readout."
            )
            fix = (
                "Usually a rotation past the last gate that can reach a measured wire -- a "
                "trailing Rz before a Z readout is the standard case. Drop the final "
                "rotation layer."
            )
        found.append(
            Finding(
                "UNMEASURABLE_PARAMETERS",
                "warning" if not narrow else "info",
                f"{len(unmeasurable)} of {stats.n_params} parameters change the state but have "
                f"a gradient of exactly zero against {_observable_label(obs)} (indices "
                f"{listing(unmeasurable)}). This is not a plateau -- a plateau is a small "
                "gradient and these are no gradient at all -- and it is not a dead parameter "
                f"either. {cause}",
                fix=fix,
                value=float(len(unmeasurable)),
            )
        )
    return found


@dataclass(frozen=True)
class LandscapeReport:
    """Gradients, mechanism, minima and curvature for one ansatz."""

    subject: str
    gradients: GradientStats
    findings: Report
    minima: MinimaScan | None = None

    def __bool__(self) -> bool:
        return bool(self.findings)

    @property
    def codes(self) -> tuple[str, ...]:
        return self.findings.codes

    def __str__(self) -> str:
        parts = [f"{self.subject}", str(self.gradients)]
        if self.minima is not None:
            parts.append(str(self.minima))
        parts.append(str(self.findings))
        return "\n\n".join(parts)


def landscape(
    ansatz: Ansatz,
    obs: Observable | None = None,
    *,
    x: ArrayLike | None = None,
    n_samples: int = 100,
    n_starts: int = 6,
    n_steps: int = 300,
    seed: int | None = 0,
    backend: BackendLike = None,
    minima: bool = True,
) -> LandscapeReport:
    """Everything this module measures, for one ansatz, in one call.

    Gradient statistics over every parameter, the mechanism behind a flat one if it is
    flat, and — unless ``minima=False`` — a multi-start scan for spurious minima. The
    minima scan is the expensive half; turn it off for a quick trainability read.

    ``x`` is one sample, and it is only needed by the minima scan, only when the ansatz
    reserves input slots. Without it the scan is skipped and the report says so rather
    than optimising the data as though it were a weight.

    Examples
    --------
    >>> import qmlkit as qk
    >>> report = qk.landscape(qk.hardware_efficient(3, 2), n_samples=40, minima=False)
    >>> report.gradients.n_params
    12
    """
    obs = Z(0) if obs is None else obs
    stats = gradient_stats(ansatz, obs, n_samples=n_samples, seed=seed, backend=backend)
    found = plateau_mechanism(ansatz, obs, n_samples=n_samples, seed=seed, backend=backend)

    found += _silent_findings(ansatz, stats, obs, seed=seed, backend=backend)

    scan = None
    if minima and ansatz.n_inputs and x is None:
        found.append(
            Finding(
                "MINIMA_SCAN_SKIPPED",
                "info",
                f"{ansatz.name!r} reserves {ansatz.n_inputs} input slots, so its landscape is "
                "a property of the model and a data point together. No point was given, so "
                "the multi-start scan was skipped rather than run over the data as if it "
                "were trainable.",
                fix="Pass x=<one sample> to scan the landscape the weights actually see.",
                value=float(ansatz.n_inputs),
            )
        )
    elif minima:
        scan = minima_scan(
            ansatz, obs, x=x, n_starts=n_starts, n_steps=n_steps, seed=seed, backend=backend
        )
    if scan is not None and not scan.benign:
        if scan.converged is False:
            # The runs disagree, but some had not finished descending, so the spread is
            # at least partly the step budget. Saying "spurious minima" here would be
            # the exact false positive this module exists to avoid.
            found.append(
                Finding(
                    "MINIMA_SCAN_INCONCLUSIVE",
                    "info",
                    f"{n_starts} random starts reached {scan.n_basins} different final losses "
                    f"(spread {scan.spread:.2e}), but not all of them had stopped descending "
                    f"after {n_steps} steps. Part of that spread is unspent step budget "
                    "rather than distinct minima, and the two are not separable from here.",
                    fix=f"Raise n_steps past {n_steps} until every endpoint classifies as a "
                    "minimum, then read the verdict. Until then this is not evidence either "
                    "way.",
                    value=float(scan.n_basins),
                )
            )
        else:
            found.append(
                Finding(
                    "SPURIOUS_MINIMA",
                    "warning",
                    f"{n_starts} random starts converged to {scan.n_basins} different final "
                    f"losses (best {scan.best:+.4f}, worst {scan.worst:+.4f}); only "
                    f"{100 * scan.reached_best:.0f}% found the best. Every endpoint is a "
                    "genuine minimum, so a single run of this model will land on the worse "
                    "value some of the time, and its loss curve will look converged when it "
                    "does.",
                    fix="Restart from several initialisations and keep the best, or add "
                    "parameters: past the overparametrisation threshold these minima flatten "
                    "out. landscape.overparametrisation finds that threshold.",
                    value=float(scan.n_basins),
                )
            )

    return LandscapeReport(
        subject=f"{ansatz.name} on {ansatz.n_qubits} qubits, observable {_observable_label(obs)}",
        gradients=stats,
        findings=Report(f"{ansatz.name} landscape", _sorted(found)),
        minima=scan,
    )
