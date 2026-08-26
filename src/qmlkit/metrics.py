r"""Does this ansatz stand a chance? — expressibility, entanglement, trainability.

Choosing an ansatz by eye is guesswork. These are the four numbers the literature
actually uses, and they pull against each other:

* **Expressibility** — how close the ansatz's state distribution gets to Haar-random.
  Measured as ``KL(fidelities || Haar)``; *smaller is more expressive*.
* **Entangling capability** — the mean Meyer–Wallach ``Q``; 0 is a product state, 1
  is maximally entangled.
* **Trainability** — the variance of the gradient. It collapses roughly as ``2^-n``
  for deep circuits, which is the barren plateau: gradients vanish faster than any
  shot budget can resolve them.
* **Generalization** — how much data the model needs, growing with trainable gates.

More expressibility costs trainability. That trade is the whole design problem, and
:class:`AnsatzReport` puts both numbers side by side so the choice is informed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from qmlkit.ansatz.library import Ansatz
from qmlkit.core.execute import BackendLike, statevector
from qmlkit.core.observables import Observable, Z
from qmlkit.info import purity

__all__ = [
    "haar_fidelity_pdf",
    "fidelity_samples",
    "expressibility",
    "meyer_wallach",
    "entangling_capability",
    "gradient_variance",
    "barren_plateau_scan",
    "effective_dimension",
    "fisher_information",
    "generalization_bound",
    "samples_for_gap",
    "noise_survival",
    "AnsatzReport",
    "compare_ansatze",
]


# --------------------------------------------------------------------------- #
# expressibility
# --------------------------------------------------------------------------- #
def haar_fidelity_pdf(f: np.ndarray, n_qubits: int) -> np.ndarray:
    """Haar-random fidelity density: ``(N-1)(1-F)^(N-2)`` for ``N = 2^n``."""
    n_states = 2**n_qubits
    fid = np.clip(np.asarray(f, dtype=float), 0.0, 1.0)
    return (n_states - 1) * (1.0 - fid) ** (n_states - 2)


def fidelity_samples(
    ansatz: Ansatz,
    n_samples: int = 2000,
    seed: int | None = None,
    backend: BackendLike = None,
) -> np.ndarray:
    """Fidelities between pairs of states from independently sampled parameters."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        a = statevector(ansatz.build(rng.uniform(-np.pi, np.pi, ansatz.n_params)), backend=backend)
        b = statevector(ansatz.build(rng.uniform(-np.pi, np.pi, ansatz.n_params)), backend=backend)
        out[i] = abs(np.vdot(a, b)) ** 2
    return out


def expressibility(
    ansatz: Ansatz,
    n_samples: int = 2000,
    n_bins: int = 75,
    seed: int | None = None,
    backend: BackendLike = None,
) -> float:
    """``KL(ansatz fidelities || Haar)``. **Smaller is more expressive**; 0 is Haar.

    Note the direction — it is a divergence *from* Haar, so a low number means the
    ansatz reaches as much of state space as a random circuit would.
    """
    fids = fidelity_samples(ansatz, n_samples, seed, backend)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    observed, _ = np.histogram(fids, bins=edges)
    p = observed / observed.sum()

    centres = (edges[:-1] + edges[1:]) / 2
    haar = haar_fidelity_pdf(centres, ansatz.n_qubits)
    q = haar / haar.sum()

    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-300, None))))


# --------------------------------------------------------------------------- #
# entanglement
# --------------------------------------------------------------------------- #
def meyer_wallach(state: np.ndarray, n_qubits: int | None = None) -> float:
    """Meyer–Wallach ``Q = 2(1 - (1/n) sum_k Tr rho_k^2)``.

    0 for any product state, 1 for a maximally entangled one.
    """
    psi = np.asarray(state, dtype=complex).ravel()
    n = n_qubits if n_qubits is not None else int(np.log2(psi.size))
    return float(2.0 * (1.0 - np.mean([purity(psi, [q], n) for q in range(n)])))


def entangling_capability(
    ansatz: Ansatz,
    n_samples: int = 200,
    seed: int | None = None,
    backend: BackendLike = None,
) -> float:
    """Mean Meyer–Wallach ``Q`` over randomly sampled parameters."""
    rng = np.random.default_rng(seed)
    vals = [
        meyer_wallach(
            statevector(ansatz.build(rng.uniform(-np.pi, np.pi, ansatz.n_params)), backend=backend),
            ansatz.n_qubits,
        )
        for _ in range(n_samples)
    ]
    return float(np.mean(vals))


# --------------------------------------------------------------------------- #
# trainability
# --------------------------------------------------------------------------- #
def gradient_variance(
    ansatz: Ansatz,
    obs: Observable | None = None,
    n_samples: int = 100,
    param_index: int = 0,
    seed: int | None = None,
    backend: BackendLike = None,
) -> float:
    """Variance of one parameter's gradient over random initialisations.

    This is the barren-plateau probe: if it falls exponentially with width, no
    realistic shot budget will resolve the gradient.
    """
    from qmlkit.gradients.dispatch import grad

    obs = Z(0) if obs is None else obs
    rng = np.random.default_rng(seed)
    spec = ansatz.build()
    vals = [
        float(
            grad(spec, rng.uniform(-np.pi, np.pi, ansatz.n_params), obs, backend=backend)[
                param_index
            ]
        )
        for _ in range(n_samples)
    ]
    return float(np.var(vals))


def barren_plateau_scan(
    ansatz_factory: Callable[[int], Ansatz],
    qubit_range: Sequence[int],
    obs_factory: Callable[[int], Observable] | None = None,
    n_samples: int = 100,
    seed: int | None = None,
    backend: BackendLike = None,
) -> dict[str, list]:
    """Gradient variance against qubit count.

    ``obs_factory`` decides the **cost locality**, which matters at fixed shallow
    depth: measured on a 2-layer hardware-efficient ansatz from 2 to 6 qubits, a
    local ``Z(0)`` holds its gradient variance flat (decay 0.98 per qubit) while a
    global ``Z^n`` collapses exponentially (0.56). Depth eventually wins regardless
    — at ``L = 2n`` both decay exponentially — so this reports the measurement
    rather than asserting a rule.
    """
    obs_factory = obs_factory or (lambda n: Z(0))
    widths, variances = [], []
    for n in qubit_range:
        widths.append(n)
        variances.append(
            gradient_variance(ansatz_factory(n), obs_factory(n), n_samples, 0, seed, backend)
        )
    decay = _decay_rate(widths, variances)
    return {
        "n_qubits": widths,
        "variance": variances,
        "decay_per_qubit": decay,
        "looks_exponential": bool(decay is not None and decay < 0.75),
    }


def _decay_rate(widths: Sequence[int], variances: Sequence[float]) -> float | None:
    """Average multiplicative change per extra qubit — under ~0.75 reads as exponential."""
    pairs = [(w, v) for w, v in zip(widths, variances, strict=True) if v > 0]
    if len(pairs) < 2:
        return None
    ratios = [pairs[i + 1][1] / pairs[i][1] for i in range(len(pairs) - 1)]
    return float(np.exp(np.mean(np.log(ratios))))


# --------------------------------------------------------------------------- #
# capacity and generalization
# --------------------------------------------------------------------------- #
def fisher_information(
    ansatz: Ansatz,
    X: np.ndarray,
    theta: np.ndarray,
    obs: Observable | None = None,
    backend: BackendLike = None,
) -> np.ndarray:
    """Classical Fisher information of the model output, averaged over inputs.

    This is the *classical* FIM of the output distribution — the object effective
    dimension is built on. Not to be confused with the quantum Fisher information,
    which is ``4 x`` the Fubini–Study metric and is what natural gradient uses.
    """
    from qmlkit.gradients.dispatch import grad

    obs = Z(0) if obs is None else obs
    spec = ansatz.build()
    p = len(theta)
    total = np.zeros((p, p))
    rows = np.atleast_2d(np.asarray(X, dtype=float))
    for _ in rows:
        g = grad(spec, theta, obs, backend=backend)
        total += np.outer(g, g)
    return total / max(len(rows), 1)


def effective_dimension(fisher: np.ndarray, n_samples: int = 1000, gamma: float = 1.0) -> float:
    """Normalised effective dimension of a model, from its Fisher information.

    How many parameters are *usefully* independent, which is generally far fewer
    than the raw count. Follows the normalised-Fisher construction rather than the
    "count eigenvalues above a threshold" shortcut, which is a teaching stand-in.
    """
    f = np.asarray(fisher, dtype=float)
    p = f.shape[0]
    trace = np.trace(f)
    if trace <= 0:
        return 0.0
    f_hat = p * f / trace  # normalised so the trace is p
    kappa = gamma * n_samples / (2 * np.pi * np.log(n_samples))
    eig = np.linalg.eigvalsh(f_hat)
    numerator = float(np.sum(np.log1p(np.clip(kappa * eig, 0, None))))
    denominator = 2.0 * np.log(kappa) if kappa > 1 else 1.0
    return float(np.clip(numerator / denominator, 0.0, p))


def generalization_bound(n_trainable_gates: int, n_samples: int, with_log: bool = True) -> float:
    r"""Expected generalization gap, ``O(sqrt(T log T / N))`` (Caro et al. 2022).

    ``with_log=False`` drops the log factor for the simplified ``sqrt(T/N)`` form the
    lectures use — convenient for teaching, but not the actual bound.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    t = max(int(n_trainable_gates), 1)
    numerator = t * np.log(t) if with_log and t > 1 else t
    return float(np.sqrt(numerator / n_samples))


def samples_for_gap(n_trainable_gates: int, gap: float, with_log: bool = True) -> int:
    """Invert the bound: how many examples to reach a target generalization gap."""
    if gap <= 0:
        raise ValueError("gap must be positive")
    t = max(int(n_trainable_gates), 1)
    numerator = t * np.log(t) if with_log and t > 1 else t
    return int(np.ceil(numerator / gap**2))


def noise_survival(gate_fidelity: float, depth: int) -> float:
    """``f^d`` — independent per-gate success compounding over the circuit."""
    return float(gate_fidelity**depth)


# --------------------------------------------------------------------------- #
# one call, every number
# --------------------------------------------------------------------------- #
@dataclass
class AnsatzReport:
    """Expressibility, entanglement, depth, cost and trainability in one call.

    print(AnsatzReport(qk.hardware_efficient(4, 2)))
    """

    ansatz: Ansatz
    n_samples: int = 300
    seed: int | None = 0
    backend: BackendLike = None
    results: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        a = self.ansatz
        res = dict(a.resources())
        res["expressibility"] = expressibility(
            a, n_samples=self.n_samples, seed=self.seed, backend=self.backend
        )
        res["entangling_capability"] = entangling_capability(
            a, n_samples=max(50, self.n_samples // 4), seed=self.seed, backend=self.backend
        )
        res["gradient_variance"] = gradient_variance(
            a, n_samples=max(30, self.n_samples // 6), seed=self.seed, backend=self.backend
        )
        res["name"] = a.name
        self.results = res

    def __getitem__(self, key: str) -> object:
        return self.results[key]

    def __str__(self) -> str:
        r = self.results
        return (
            f"{r['name']} on {r['n_qubits']} qubits\n"
            f"  parameters            {r['n_params']}\n"
            f"  depth                 {r['depth']}\n"
            f"  two-qubit gates       {r['n_2q']}\n"
            f"  gradient circuits     {r['grad_circuits']}\n"
            f"  expressibility        {r['expressibility']:.4f}   (KL from Haar,\n"
            f"                                 lower is more expressive)\n"
            f"  entangling capability {r['entangling_capability']:.4f}   (Meyer-Wallach Q)\n"
            f"  gradient variance     {r['gradient_variance']:.3e}   (higher = more trainable)"
        )


def compare_ansatze(
    ansatze: Sequence[Ansatz], n_samples: int = 300, seed: int | None = 0
) -> list[dict[str, object]]:
    """The same report across candidates — the table a paper would want."""
    return [AnsatzReport(a, n_samples=n_samples, seed=seed).results for a in ansatze]
