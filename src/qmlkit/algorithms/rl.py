"""Quantum policy gradient — a variational circuit as an RL policy.

REINFORCE, with the policy replaced by a circuit: encode the observation, measure a
few observables, softmax them into action probabilities, and push up the log-
probability of whatever earned reward.

    policy = QuantumPolicy(n_observations=2, n_actions=2)
    result = train_reinforce(policy, ContextualBandit(seed=0), n_episodes=200, seed=0)

The environment is an argument with a three-method protocol (``reset``, ``step``,
``n_observations``/``n_actions``), so a Gym environment wraps in a few lines and
nothing here needs Gym installed. The ansatz and feature map are arguments too, as
everywhere else.

The gradient is the ordinary policy-gradient one — ``grad log pi(a|s) * return`` —
computed through :func:`qmlkit.grad`, so it is exact rather than finite-differenced.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from qmlkit.ansatz.library import Ansatz, hardware_efficient
from qmlkit.core.execute import BackendLike, expval
from qmlkit.core.observables import Observable, Z
from qmlkit.encoding.feature_maps import AngleFeatureMap, FeatureMap
from qmlkit.gradients.dispatch import grad

__all__ = ["QuantumPolicy", "ContextualBandit", "train_reinforce", "ReinforceResult", "Environment"]


class Environment(Protocol):
    """The three things an RL loop needs. Deliberately not Gym."""

    n_observations: int
    n_actions: int

    def reset(self) -> npt.NDArray[Any]: ...

    def step(self, action: int) -> tuple[npt.NDArray[Any], float, bool]: ...


class ContextualBandit:
    """A tiny environment with a known optimal policy, so training is checkable.

    The observation is a random vector; the correct action is the sign of its first
    coordinate. One step per episode, reward 1 for right and 0 for wrong — so the
    optimal return is exactly 1.0 and "did it learn" has an unambiguous answer.
    """

    n_observations = 2
    n_actions = 2

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._state = np.zeros(self.n_observations)

    def reset(self) -> npt.NDArray[Any]:
        self._state = self._rng.uniform(-1.0, 1.0, self.n_observations)
        return self._state

    def step(self, action: int) -> tuple[npt.NDArray[Any], float, bool]:
        correct = int(self._state[0] > 0)
        return self._state, float(action == correct), True

    def optimal_return(self) -> float:
        return 1.0


class QuantumPolicy:
    """A circuit policy: observation in, action probabilities out."""

    def __init__(
        self,
        n_observations: int,
        n_actions: int,
        feature_map: FeatureMap | None = None,
        ansatz: Ansatz | None = None,
        observables: Sequence[Observable] | None = None,
        n_layers: int = 2,
        beta: float = 2.0,
        backend: BackendLike = None,
        seed: int | None = None,
    ) -> None:
        n_qubits = max(n_observations, n_actions)
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.n_qubits = n_qubits
        self.feature_map = feature_map or AngleFeatureMap(n_qubits, entangle=n_qubits > 1)
        self.ansatz = ansatz or hardware_efficient(n_qubits, n_layers)
        self.observables = list(observables) if observables else [Z(a) for a in range(n_actions)]
        self.beta = beta  # softmax inverse temperature
        self.backend = backend
        # Seeded from the argument, not from a constant: with seed baked in, every
        # QuantumPolicy started from the same point and "average over seeds" was
        # impossible to express.
        self.theta = self.ansatz.init("small", seed=seed)
        self._spec = self.feature_map.build_parametric().compose(self.ansatz.build())
        self._n_inputs = self.feature_map.n_angles

    def _full(self, observation: npt.NDArray[Any], theta: npt.NDArray[Any]) -> npt.NDArray[Any]:
        padded = np.zeros(self.n_qubits)
        padded[: self.n_observations] = np.asarray(observation, dtype=float).ravel()
        return np.concatenate([self.feature_map.angles(padded), theta])

    def logits(
        self, observation: npt.NDArray[Any], theta: npt.NDArray[Any] | None = None
    ) -> npt.NDArray[Any]:
        full = self._full(observation, self.theta if theta is None else theta)
        return np.array(
            [expval(self._spec, o, theta=full, backend=self.backend) for o in self.observables]
        )

    def probabilities(
        self, observation: npt.NDArray[Any], theta: npt.NDArray[Any] | None = None
    ) -> npt.NDArray[Any]:
        z = self.beta * self.logits(observation, theta)
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, observation: npt.NDArray[Any], rng: np.random.Generator) -> int:
        return int(rng.choice(self.n_actions, p=self.probabilities(observation)))

    def grad_log_prob(self, observation: npt.NDArray[Any], action: int) -> npt.NDArray[Any]:
        r"""``d/dtheta log pi(a|s)``.

        For a softmax over measured observables this is
        ``beta * (dO_a/dtheta - sum_b pi_b dO_b/dtheta)`` — one exact circuit gradient
        per action, no finite differences anywhere.
        """
        full = self._full(observation, self.theta)
        probs = self.probabilities(observation)
        jac = np.stack(
            [
                grad(self._spec, full, o, backend=self.backend)[self._n_inputs :]
                for o in self.observables
            ]
        )
        return self.beta * (jac[action] - probs @ jac)

    def __repr__(self) -> str:
        return (
            f"QuantumPolicy(obs={self.n_observations}, actions={self.n_actions}, "
            f"P={self.theta.size}, ansatz={self.ansatz.name!r})"
        )


@dataclass
class ReinforceResult:
    theta: npt.NDArray[Any]
    returns: list[float] = field(default_factory=list)

    def mean_return(self, last: int = 50) -> float:
        return float(np.mean(self.returns[-last:])) if self.returns else 0.0

    def __repr__(self) -> str:
        return (
            f"ReinforceResult(episodes={len(self.returns)}, "
            f"final_mean_return={self.mean_return():.4f})"
        )


def train_reinforce(
    policy: QuantumPolicy,
    env: Environment,
    n_episodes: int = 200,
    lr: float = 0.2,
    baseline: bool = True,
    seed: int | None = None,
) -> ReinforceResult:
    """REINFORCE with an optional moving-average baseline.

    The baseline subtracts a running mean return before scaling the gradient. It does
    not change what is being optimised, only the variance of the estimate — which for
    a policy sampled one episode at a time is the thing that decides whether it
    learns at all.
    """
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    running = 0.0

    for episode in range(n_episodes):
        observation = env.reset()
        action = policy.sample(observation, rng)
        _, reward, _ = env.step(action)
        returns.append(float(reward))

        advantage = reward - running if baseline else reward
        policy.theta = policy.theta + lr * advantage * policy.grad_log_prob(observation, action)
        running += (reward - running) / (episode + 1)

    return ReinforceResult(theta=policy.theta, returns=returns)
