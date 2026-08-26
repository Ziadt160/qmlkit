r"""Generative models — learning a *distribution* rather than a mapping.

Two families, and the split matters:

* **Born machines** (QCBM, qGAN) are **implicit**. Measuring the circuit samples
  ``p(x) = |<x|psi>|^2`` directly, so sampling is free and scoring is not: you
  cannot ask such a model for ``p(x)`` of an arbitrary ``x`` without estimating it.
  Training therefore uses a *sample-based* loss — MMD, or a discriminator.
* **Energy models** (QBM, quantum Hopfield) are **explicit**. They define
  ``p(x) ∝ exp(-E(x))``, so scoring is easy and sampling is hard, because the
  partition function sums over ``2^n`` states.

That is the whole taxonomy, and it decides which loss you can even write down.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from qmlkit.ansatz.library import Ansatz, hardware_efficient
from qmlkit.core.execute import BackendLike, probabilities, run_counts
from qmlkit.core.ir import CircuitSpec

__all__ = [
    "gaussian_kernel",
    "mmd_squared",
    "kl_divergence",
    "total_variation",
    "QCBM",
    "QGAN",
    "boltzmann",
    "partition_function",
    "ising_energy",
    "QuantumBoltzmannMachine",
    "QuantumHopfield",
]


# --------------------------------------------------------------------------- #
# sample-based losses — the only kind an implicit model admits
# --------------------------------------------------------------------------- #
def gaussian_kernel(a: np.ndarray, b: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """``exp(-gamma |a - b|^2)`` over all pairs."""
    x = np.atleast_2d(np.asarray(a, dtype=float))
    y = np.atleast_2d(np.asarray(b, dtype=float))
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"sample widths differ: {x.shape[1]} vs {y.shape[1]}")
    sq = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * sq)


def mmd_squared(x: np.ndarray, y: np.ndarray, gamma: float | Sequence[float] = 1.0) -> float:
    """Maximum mean discrepancy between two sample sets.

    Zero exactly when the distributions match. A *sample* statistic — it never needs
    ``p(x)``, which is why an implicit model can be trained on it at all. Passing
    several ``gamma`` values averages kernels of different widths, which stops the
    loss going blind at one scale.
    """
    gammas = [gamma] if isinstance(gamma, (int, float)) else list(gamma)
    total = 0.0
    for g in gammas:
        total += float(
            gaussian_kernel(x, x, g).mean()
            + gaussian_kernel(y, y, g).mean()
            - 2 * gaussian_kernel(x, y, g).mean()
        )
    return total / len(gammas)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """``KL(p || q)`` over two discrete distributions."""
    a = np.clip(np.asarray(p, dtype=float), eps, None)
    b = np.clip(np.asarray(q, dtype=float), eps, None)
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum(a * np.log(a / b)))


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    """``0.5 * sum |p - q|`` — bounded in ``[0, 1]``, unlike KL."""
    a = np.asarray(p, dtype=float)
    b = np.asarray(q, dtype=float)
    return float(0.5 * np.abs(a / a.sum() - b / b.sum()).sum())


# --------------------------------------------------------------------------- #
# Born machines — implicit
# --------------------------------------------------------------------------- #
class QCBM:
    """Quantum circuit Born machine.

    The circuit *is* the distribution: measuring it samples ``|<x|psi>|^2``. There is
    no likelihood to maximise, so training minimises MMD between its samples and the
    data — a distance you can compute from samples alone.
    """

    def __init__(
        self,
        n_qubits: int,
        ansatz: Ansatz | None = None,
        n_layers: int = 3,
        backend: BackendLike = None,
        shots: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.n_qubits = n_qubits
        self.ansatz = ansatz or hardware_efficient(
            n_qubits, n_layers, entangler="cx", pattern="ring"
        )
        self.backend = backend
        self.shots = shots
        self.params_ = self.ansatz.init("uniform", seed)
        self.history_: list[float] = []

    # ------------------------------------------------------------------------
    def circuit(self, params: Sequence[float] | None = None) -> CircuitSpec:
        return self.ansatz.build(self.params_ if params is None else params)

    def probabilities(self, params: Sequence[float] | None = None) -> np.ndarray:
        return probabilities(self.circuit(params), backend=self.backend)

    def sample(
        self, n_samples: int = 512, params: Sequence[float] | None = None, seed: int | None = None
    ) -> np.ndarray:
        """Draw bitstrings as a ``(n_samples, n_qubits)`` array of 0/1."""
        counts = run_counts(self.circuit(params), shots=n_samples, backend=self.backend, seed=seed)
        rows = [[int(b) for b in bits] for bits, n in counts.items() for _ in range(n)]
        return np.array(rows, dtype=int)

    def fit(
        self,
        data: np.ndarray,
        n_iterations: int = 100,
        gamma: float | Sequence[float] = (0.25, 1.0, 4.0),
        n_samples: int = 512,
        seed: int | None = None,
        callback: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> QCBM:
        """Train by minimising MMD against ``data``, using SPSA.

        SPSA because the loss is a sample statistic: two circuit evaluations per step
        whatever the parameter count, and it tolerates the sampling noise.
        """
        from qmlkit.gradients.spsa import minimize_spsa

        target = np.atleast_2d(np.asarray(data, dtype=float))
        rng = np.random.default_rng(seed)

        def loss(p: np.ndarray) -> float:
            drawn = self.sample(n_samples, p, seed=int(rng.integers(1 << 31)))
            return mmd_squared(drawn, target, gamma)

        best, history = minimize_spsa(
            loss, self.params_, n_iterations=n_iterations, seed=seed, callback=callback
        )
        self.params_ = best
        self.history_ = history
        return self

    def score(
        self,
        data: np.ndarray,
        gamma: float | Sequence[float] = (0.25, 1.0, 4.0),
        n_samples: int = 1024,
        seed: int | None = None,
    ) -> float:
        """MMD² against the data — zero means the distributions match.

        This is a *sampled* estimate, so pass ``seed`` if you need it reproducible;
        without one it draws on the backend's own RNG. For a deterministic comparison
        on a simulator use :meth:`exact_distance`, which needs no samples at all.
        """
        return mmd_squared(self.sample(n_samples, seed=seed), np.atleast_2d(data), gamma)

    def exact_distance(
        self, data: np.ndarray, metric: str = "tv", params: Sequence[float] | None = None
    ) -> float:
        """Distance to the target distribution with **no sampling**.

        On a simulator the model's distribution is available exactly, so progress can
        be measured without shot noise — which is what makes a before/after comparison
        trustworthy rather than a coin flip.
        """
        rows = np.atleast_2d(np.asarray(data, dtype=int))
        target = np.zeros(2**self.n_qubits)
        for row in rows:
            target[int("".join(map(str, row)), 2)] += 1
        target /= target.sum()

        model = self.probabilities(params)
        if metric == "tv":
            return total_variation(model, target)
        if metric == "kl":
            return kl_divergence(target, model)
        raise ValueError(f"unknown metric {metric!r}; expected 'tv' or 'kl'")

    def __repr__(self) -> str:
        return f"QCBM(n_qubits={self.n_qubits}, n_params={self.ansatz.n_params})"


class QGAN:
    """Quantum generator, classical discriminator.

    The generator is a Born machine; the discriminator is any callable scoring a
    batch as real. They are trained against each other, and at equilibrium the
    discriminator should be at chance — which is what :meth:`equilibrium_gap` reports.
    """

    def __init__(
        self,
        generator: QCBM,
        discriminator: Callable[[np.ndarray], np.ndarray],
        seed: int | None = None,
    ) -> None:
        self.generator = generator
        self.discriminator = discriminator
        self.rng = np.random.default_rng(seed)
        self.history_: list[float] = []

    def generator_loss(self, params: np.ndarray, n_samples: int = 256) -> float:
        """Generator wants the discriminator to call its samples real."""
        fake = self.generator.sample(n_samples, params, seed=int(self.rng.integers(1 << 31)))
        scores = np.asarray(self.discriminator(fake), dtype=float)
        return float(-np.log(np.clip(scores, 1e-9, 1.0)).mean())

    def fit_generator(
        self, n_iterations: int = 50, n_samples: int = 256, seed: int | None = None
    ) -> QGAN:
        """Train the generator against a fixed discriminator."""
        from qmlkit.gradients.spsa import minimize_spsa

        best, history = minimize_spsa(
            lambda p: self.generator_loss(p, n_samples),
            self.generator.params_,
            n_iterations=n_iterations,
            seed=seed,
        )
        self.generator.params_ = best
        self.history_ = history
        return self

    def equilibrium_gap(self, real: np.ndarray, n_samples: int = 256) -> float:
        """``|accuracy - 0.5|`` — zero when the discriminator is guessing."""
        fake = self.generator.sample(n_samples)
        real_scores = np.asarray(self.discriminator(np.atleast_2d(real)), dtype=float)
        fake_scores = np.asarray(self.discriminator(fake), dtype=float)
        accuracy = 0.5 * ((real_scores > 0.5).mean() + (fake_scores <= 0.5).mean())
        return float(abs(accuracy - 0.5))


# --------------------------------------------------------------------------- #
# energy-based models — explicit
# --------------------------------------------------------------------------- #
def boltzmann(energies: np.ndarray, beta: float = 1.0) -> tuple[np.ndarray, float]:
    """``(p, Z)`` for ``p(x) = exp(-beta E(x)) / Z``."""
    e = np.asarray(energies, dtype=float)
    weights = np.exp(-beta * (e - e.min()))  # shift for numerical stability
    z = float(weights.sum())
    return weights / z, z


def partition_function(energies: np.ndarray, beta: float = 1.0) -> float:
    """``Z = sum exp(-beta E)`` — the sum over ``2^n`` states that makes sampling hard."""
    return float(np.exp(-beta * np.asarray(energies, dtype=float)).sum())


def ising_energy(spins: Sequence[int], fields: np.ndarray, couplings: dict) -> float:
    """``-sum b_i s_i - sum w_ij s_i s_j`` for spins in ``{+1, -1}``."""
    s = np.asarray(spins, dtype=float)
    energy = -float(np.dot(np.asarray(fields, dtype=float), s))
    for (i, j), w in couplings.items():
        energy -= float(w) * float(s[i]) * float(s[j])
    return energy


class QuantumBoltzmannMachine:
    """A transverse-field Ising model as a generative model.

    The classical part is diagonal (``Z`` fields and ``ZZ`` couplings); the transverse
    field ``Gamma * X`` is **off-diagonal**, which is what makes it quantum — and what
    makes the log-likelihood gradient intractable, since the model term no longer has
    a spins-in-number-out form. Training therefore optimises a *lower bound*, and
    ``grad`` here is the bound's ``clamped - model`` difference.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int = 0,
        gamma: float = 0.7,
        beta: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_spins = n_visible + n_hidden
        self.gamma = gamma
        self.beta = beta
        rng = np.random.default_rng(seed)
        self.fields = rng.normal(0, 0.1, self.n_spins)
        self.couplings = {(i, i + 1): float(rng.normal(0, 0.1)) for i in range(self.n_spins - 1)}

    def energies(self) -> np.ndarray:
        """Diagonal energy of every spin configuration."""
        configs = list(itertools.product([1, -1], repeat=self.n_spins))
        return np.array([ising_energy(c, self.fields, self.couplings) for c in configs])

    def probabilities(self) -> np.ndarray:
        """Boltzmann distribution over configurations (diagonal part only)."""
        return boltzmann(self.energies(), self.beta)[0]

    def visible_marginal(self) -> np.ndarray:
        """Marginalise the hidden spins away."""
        p = self.probabilities()
        if self.n_hidden == 0:
            return p
        return p.reshape(2**self.n_visible, 2**self.n_hidden).sum(axis=1)

    @staticmethod
    def grad(clamped: np.ndarray, model: np.ndarray) -> np.ndarray:
        """``data - model`` — the sculptor move behind every Boltzmann update."""
        return np.asarray(clamped, dtype=float) - np.asarray(model, dtype=float)

    def free_energy(self, entropy: float, temperature: float = 1.0) -> float:
        """``<H> - T S`` — what a thermal state actually minimises."""
        return float(np.dot(self.probabilities(), self.energies()) - temperature * entropy)

    def __repr__(self) -> str:
        return (
            f"QuantumBoltzmannMachine(visible={self.n_visible}, hidden={self.n_hidden}, "
            f"gamma={self.gamma})"
        )


class QuantumHopfield:
    """Associative memory: store patterns, recall the nearest by state overlap.

    Recall is a fidelity comparison against each stored pattern — the same overlap a
    swap test estimates, which is why this belongs beside the kernel methods.
    """

    def __init__(self) -> None:
        self.patterns_: dict[Any, np.ndarray] = {}

    def store(self, patterns: dict[Any, Sequence[float]]) -> QuantumHopfield:
        """Store named patterns, normalised to unit vectors."""
        for name, p in patterns.items():
            vec = np.asarray(p, dtype=float).ravel()
            norm = np.linalg.norm(vec)
            if norm < 1e-12:
                raise ValueError(f"pattern {name!r} is the zero vector")
            self.patterns_[name] = vec / norm
        return self

    def overlaps(self, cue: Sequence[float]) -> dict[Any, float]:
        """``|<pattern|cue>|^2`` for every stored pattern."""
        if not self.patterns_:
            raise ValueError("no patterns stored")
        c = np.asarray(cue, dtype=float).ravel()
        norm = np.linalg.norm(c)
        if norm < 1e-12:
            raise ValueError("cue is the zero vector")
        c = c / norm
        return {k: float(abs(np.dot(v, c)) ** 2) for k, v in self.patterns_.items()}

    def recall(self, cue: Sequence[float]) -> Any:
        """The stored pattern the cue overlaps most."""
        scores = self.overlaps(cue)
        return max(scores, key=lambda k: scores[k])

    @staticmethod
    def swap_probability(overlap: float) -> float:
        """``P(anc=0) = (1 + overlap) / 2`` — the swap-test readout."""
        return 0.5 + 0.5 * float(overlap)

    @staticmethod
    def overlap_from_probability(p0: float) -> float:
        """Invert the swap-test readout."""
        return 2.0 * float(p0) - 1.0
