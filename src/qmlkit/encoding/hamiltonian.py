r"""Hamiltonian (IQP-style) encoding and data re-uploading.

**Hamiltonian encoding** evolves the register under a data-dependent Hamiltonian
:math:`H(x) = \sum_i x_i Z_i + \sum_{(i,j)} x_i x_j Z_i Z_j` for a time ``t``,
Trotterised into ``steps`` slices. Because every term commutes here, the Trotter
split is *exact* at any number of steps — ``steps`` changes the circuit depth and
nothing else. That is worth knowing before anyone tunes it hoping for accuracy.

**Data re-uploading** interleaves the encoding with trainable blocks. Each repeat
widens the reachable Fourier spectrum: ``L`` uploads reach frequencies ``0..L``,
which is the knob that decides *which functions the model can represent at all*,
separately from the ansatz that picks the coefficients.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from qmlkit.core.builder import QCircuit, entangler_pairs
from qmlkit.core.ir import CircuitSpec, ParamRef

__all__ = [
    "hamiltonian_encode",
    "trotter_rz_angle",
    "trotter_zz_angle",
    "DataReuploadEncoder",
    "n_reachable_frequencies",
]


def trotter_rz_angle(xi: float, t: float, steps: int) -> float:
    """Single-qubit ``Rz`` angle per Trotter step: ``2 * x_i * t / steps``."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return 2.0 * float(xi) * float(t) / steps


def trotter_zz_angle(xi: float, xj: float, t: float, steps: int) -> float:
    """Two-qubit coupling angle per Trotter step: ``2 * x_i * x_j * t / steps``."""
    if steps < 1:
        raise ValueError("steps must be at least 1")
    return 2.0 * float(xi) * float(xj) * float(t) / steps


def hamiltonian_encode(
    x: Sequence[float],
    t: float = 1.0,
    steps: int = 3,
    entanglement: str = "chain",
    initial_hadamard: bool = True,
) -> CircuitSpec:
    """Evolve under a data-dependent Ising Hamiltonian.

    ``initial_hadamard=True`` starts in the uniform superposition, which is what
    makes the Z-diagonal evolution do anything observable — without it the register
    stays in a computational basis state and only picks up a global phase.
    """
    values = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
    n = values.size
    if n == 0:
        raise ValueError("hamiltonian_encode needs at least one feature")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    qc = QCircuit(n)
    if initial_hadamard:
        for i in range(n):
            qc.h(i)

    pairs = entangler_pairs(n, entanglement)
    for _ in range(steps):
        for i, xi in enumerate(values):
            qc.rz(i, trotter_rz_angle(xi, t, steps))
        for a, b in pairs:
            qc.cx(a, b)
            qc.rz(b, trotter_zz_angle(values[a], values[b], t, steps))
            qc.cx(a, b)
    return qc.to_spec()


def n_reachable_frequencies(n_uploads: int) -> int:
    """``L`` uploads reach frequencies ``0..L`` -- so ``L + 1`` of them.

    This holds only when the trainable block does **not** commute with the encoding
    rotation. If it does, the uploads merge into one rotation and the model reaches
    a single frequency instead. :class:`DataReuploadEncoder` warns when you build
    such a pairing.
    """
    if n_uploads < 0:
        raise ValueError("n_uploads cannot be negative")
    return n_uploads + 1


class DataReuploadEncoder:
    """One convenient re-uploading shape: angle encoding, rotations, entangler.

    .. note::
       Re-uploading is a **pattern, not a structure** — any feature map, any
       trainable block, any interleaving. This class fixes one convenient choice.
       For anything else use :func:`qmlkit.reupload`, or compose
       :class:`~qmlkit.ansatz.blocks.EncodingLayer` directly with the block
       vocabulary. This remains for the plain angle-encoding case.

    The circuit alternates ``S(x)`` — an angle encoding — with ``W(theta)``, a
    trainable rotation block, ``n_uploads`` times. Data enters as *literals* by
    default; pass ``trainable_input=True`` to make the features circuit parameters
    too, which is what yields ``df/dx`` for a classical pre-net.

    The parameter vector is laid out as ``(n_uploads, n_qubits, len(rotations))``,
    flattened, with the input parameters (if trainable) appended after it.
    """

    def __init__(
        self,
        n_features: int,
        n_uploads: int = 3,
        rotations: Sequence[str] = ("rz", "ry", "rz"),
        encoding_rotation: str = "ry",
        entanglement: str | None = "chain",
        trainable_input: bool = False,
    ) -> None:
        if n_features < 1:
            raise ValueError("n_features must be at least 1")
        if n_uploads < 1:
            raise ValueError("n_uploads must be at least 1")
        self.n_features = n_features
        self.n_qubits = n_features
        self.n_uploads = n_uploads
        self.rotations = tuple(rotations)
        self.encoding_rotation = encoding_rotation
        self.entanglement = entanglement
        self.trainable_input = trainable_input

        # A trainable block that COMMUTES with the encoding is a silent trap:
        # Ry(x) Ry(t1) Ry(x) Ry(t2) = Ry(2x + t1 + t2), so the model collapses to a
        # single frequency and every weight becomes a phase shift. Measured: with
        # W = Ry only, L uploads give exactly one frequency (amplitude 1.0); with
        # W = Rz Ry Rz they give the full 0..L spectrum.
        if set(self.rotations) <= {self.encoding_rotation}:
            warnings.warn(
                f"rotations={self.rotations} commutes with encoding_rotation="
                f"{self.encoding_rotation!r}, so the uploads collapse into a single "
                f"rotation: the model reaches only frequency {n_uploads}, not 0..{n_uploads}, "
                "and its weights have no effect beyond a phase. Use a non-commuting "
                'block such as ("rz", "ry", "rz").',
                UserWarning,
                stacklevel=2,
            )

    @property
    def n_weights(self) -> int:
        """Trainable angles in the variational blocks."""
        return self.n_uploads * self.n_qubits * len(self.rotations)

    @property
    def weight_shape(self) -> tuple[int, int, int]:
        return (self.n_uploads, self.n_qubits, len(self.rotations))

    @property
    def n_params(self) -> int:
        """Total circuit parameters — weights, plus inputs when they are trainable."""
        return self.n_weights + (self.n_features if self.trainable_input else 0)

    @property
    def n_frequencies(self) -> int:
        return n_reachable_frequencies(self.n_uploads)

    def build(self, x: Sequence[float] | None = None) -> CircuitSpec:
        """Build the circuit.

        With ``trainable_input=False`` (default) ``x`` is required and baked in.
        With ``trainable_input=True`` ``x`` is ignored: the features become
        parameters, supplied later alongside the weights.
        """
        if not self.trainable_input:
            if x is None:
                raise ValueError("x is required unless trainable_input=True")
            values = np.atleast_1d(np.asarray(x, dtype=float)).ravel()
            if values.size != self.n_features:
                raise ValueError(f"expected {self.n_features} features, got {values.size}")

        qc = QCircuit(self.n_qubits)
        w = 0
        for _ in range(self.n_uploads):
            for i in range(self.n_qubits):  # S(x): inject the data
                angle = ParamRef(self.n_weights + i) if self.trainable_input else float(values[i])
                qc.apply(self.encoding_rotation, i, angle)
            for i in range(self.n_qubits):  # W(theta): the trainable block
                for gate in self.rotations:
                    qc.apply(gate, i, ParamRef(w))
                    w += 1
            if self.entanglement and self.n_qubits > 1:
                qc.entangle(self.entanglement)
        return qc.to_spec()

    def __call__(self, x: Sequence[float] | None = None) -> CircuitSpec:
        return self.build(x)

    def __repr__(self) -> str:
        return (
            f"DataReuploadEncoder(n_features={self.n_features}, n_uploads={self.n_uploads}, "
            f"n_weights={self.n_weights}, frequencies=0..{self.n_uploads})"
        )
