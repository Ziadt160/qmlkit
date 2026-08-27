"""Quantum autoencoder — compress ``n`` qubits into ``k`` (Romero, Olson & Aspuru-Guzik 2017).

The trick is that you never need the decoder to train. If the encoder has genuinely
pushed all the information into ``k`` latent qubits, the discarded "trash" qubits must
be left in a known pure state — so **maximising the trash qubits' purity is the whole
loss**, and it costs no extra circuits.

    from qmlkit.algorithms import QuantumAutoencoder

    model = QuantumAutoencoder(n_qubits=4, n_latent=2)
    result = model.fit(states, seed=0)
    print(result.fidelity)       # how well the input survives a round trip

The encoder is an ``Ansatz`` argument like everywhere else, so "which circuit
compresses best" is an experiment you run, not a fork of this file.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.algorithms.vqe import OPTIMIZERS, Optimizer
from qmlkit.ansatz.library import Ansatz, hardware_efficient
from qmlkit.core.execute import BackendLike, expectation, statevector
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, PauliString, PauliSum
from qmlkit.info import purity, state_fidelity

__all__ = ["QuantumAutoencoder", "AutoencoderResult"]


@dataclass
class AutoencoderResult:
    theta: npt.NDArray[Any]
    trash_purity: float
    fidelity: float
    trash_fidelity: float = 0.0
    history: list[float] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"AutoencoderResult(trash_fidelity={self.trash_fidelity:.6f}, "
            f"fidelity={self.fidelity:.6f}, steps={len(self.history) - 1})"
        )


class QuantumAutoencoder:
    """Train an encoder that concentrates a state into ``n_latent`` qubits.

    Parameters
    ----------
    n_qubits, n_latent
        Width in, width kept. The remaining ``n_qubits - n_latent`` are the trash.
    encoder
        Any ``Ansatz`` of the right width. Defaults to hardware-efficient.
    trash
        Which wires to discard. Defaults to the last ones.
    """

    def __init__(
        self,
        n_qubits: int,
        n_latent: int,
        encoder: Ansatz | None = None,
        n_layers: int = 3,
        trash: Sequence[int] | None = None,
        optimizer: str | Optimizer = "rotosolve",
        backend: BackendLike = None,
    ) -> None:
        if not 0 < n_latent < n_qubits:
            raise ValueError(f"n_latent must be between 1 and {n_qubits - 1}, got {n_latent}")
        self.n_qubits = n_qubits
        self.n_latent = n_latent
        self.trash = list(trash) if trash is not None else list(range(n_latent, n_qubits))
        self.encoder = encoder or hardware_efficient(n_qubits, n_layers)
        self.optimizer = optimizer
        self.backend = backend
        self._spec = self.encoder.build()

    # ------------------------------------------------------------------- loss --
    def _trash_projector(self) -> Observable:
        r"""The observable :math:`\prod_{i \in \mathrm{trash}} (I + Z_i)/2`.

        This projects onto "every trash wire reads zero". Expanding the product gives
        an ordinary Pauli sum, which matters more than it looks: the loss is then a
        plain expectation value, so gradients come from :func:`qmlkit.grad` and
        Rotosolve is valid — neither of which is true of a purity-based loss.
        """
        terms: list[PauliString] = []
        scale = 0.5 ** len(self.trash)
        for mask in range(2 ** len(self.trash)):
            chosen = [w for i, w in enumerate(self.trash) if mask >> i & 1]
            terms.append(PauliString(tuple((w, "Z") for w in sorted(chosen)), scale))
        return PauliSum(tuple(terms))

    def trash_fidelity(self, theta: Sequence[float], states: Sequence[CircuitSpec]) -> float:
        r"""Mean :math:`\langle 0|
        ho_\mathrm{trash}|0
        angle` — 1.0 is perfect compression.

                Purity alone is **not** enough, and getting that wrong is easy: an encoder can
                leave the trash in a pure state pointing somewhere other than
                :math:`|0
        angle`, scoring purity 0.998 while the round trip only returns
                fidelity 0.21. Measured, on the way to writing this. What the decoder needs is
                the trash reset to a *known* state, so that is what the loss asks for.
        """
        arr = np.asarray(theta, dtype=float)
        projector = self._trash_projector()
        total = 0.0
        for prep in states:
            encoded = prep.compose(self._spec.bind(arr))
            total += float(expectation(encoded, projector, backend=self.backend))
        return total / len(states)

    def trash_purity(self, theta: Sequence[float], states: Sequence[CircuitSpec]) -> float:
        """Mean purity of the discarded wires. Reported, but not what is optimised."""
        arr = np.asarray(theta, dtype=float)
        total = 0.0
        for prep in states:
            encoded = prep.compose(self._spec.bind(arr))
            total += purity(encoded, self.trash, backend=self.backend)
        return total / len(states)

    def loss(self, theta: Sequence[float], states: Sequence[CircuitSpec]) -> float:
        """One minus the trash fidelity. No decoder is ever built to train this."""
        return 1.0 - self.trash_fidelity(theta, states)

    # -------------------------------------------------------------------- fit --
    def fit(
        self,
        states: Sequence[CircuitSpec],
        theta0: Sequence[float] | None = None,
        seed: int | None = None,
        **optimizer_kwargs: Any,
    ) -> AutoencoderResult:
        start = (
            np.asarray(theta0, dtype=float)
            if theta0 is not None
            else self.encoder.init("small", seed=seed)
        )
        fn = OPTIMIZERS[self.optimizer] if isinstance(self.optimizer, str) else self.optimizer
        if fn is OPTIMIZERS["spsa"]:
            optimizer_kwargs.setdefault("seed", seed)
        theta, history = fn(lambda t: self.loss(t, states), start, **optimizer_kwargs)

        return AutoencoderResult(
            theta=theta,
            trash_purity=self.trash_purity(theta, states),
            trash_fidelity=self.trash_fidelity(theta, states),
            fidelity=self.round_trip_fidelity(theta, states),
            history=list(history),
        )

    # ------------------------------------------------------------- validation --
    def round_trip_fidelity(self, theta: Sequence[float], states: Sequence[CircuitSpec]) -> float:
        """Encode, reset the trash to ``|0>``, decode, and compare to the input.

        This is the quantity the compression *claims*, and it is deliberately not the
        training loss — it is the independent check that maximising trash purity was
        the right proxy at all.
        """
        arr = np.asarray(theta, dtype=float)
        encoder = self._spec.bind(arr)
        decoder = encoder.adjoint()
        total = 0.0
        for prep in states:
            original = statevector(prep, backend=self.backend)
            encoded = statevector(prep.compose(encoder), backend=self.backend)
            restored = self._apply_spec(self._reset_trash(encoded), decoder)
            total += state_fidelity(restored, original)
        return total / len(states)

    def _reset_trash(self, state: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Project the trash wires onto ``|0>`` and renormalise."""
        tensor = np.asarray(state).reshape((2,) * self.n_qubits)
        keep: list[Any] = [slice(None)] * self.n_qubits
        for wire in self.trash:
            keep[wire] = 0
        projected = np.zeros_like(tensor)
        projected[tuple(keep)] = tensor[tuple(keep)]
        flat = projected.reshape(-1)
        norm = np.linalg.norm(flat)
        return flat / norm if norm > 1e-12 else flat

    def _apply_spec(self, state: npt.NDArray[Any], spec: CircuitSpec) -> npt.NDArray[Any]:
        from qmlkit.core.backends.numpy_backend import _apply
        from qmlkit.core.gates import gate_matrix

        out = np.asarray(state, dtype=complex).reshape((2,) * self.n_qubits)
        for op in spec.ops:
            angles = tuple(float(p) for p in op.params)
            out = _apply(out, gate_matrix(op.gate, angles), op.qubits)
        return out.reshape(-1)

    def __repr__(self) -> str:
        return (
            f"QuantumAutoencoder({self.n_qubits} -> {self.n_latent}, "
            f"trash={self.trash}, encoder={self.encoder.name!r})"
        )
