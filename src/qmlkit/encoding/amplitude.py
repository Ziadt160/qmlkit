r"""Amplitude encoding — :math:`n` qubits hold :math:`2^n` numbers.

Built from **uniformly-controlled rotations**, not from a backend state-preparation
primitive. That matters: the resulting circuit is made of ordinary registered gates,
so it runs identically on every backend, can be drawn and transpiled, and its
resource cost is visible rather than hidden inside an SDK call.

The construction is the standard one. Magnitudes come from a binary tree of partial
norms, each level applying a uniformly-controlled ``Ry``; phases, when the data is
complex, come from a second cascade of uniformly-controlled ``Rz``.

A uniformly-controlled rotation decomposes recursively:

.. code-block:: text

    UCR(theta, [c0, ...], t) = UCR(alpha, [...], t) . CX(c0, t) . UCR(beta, [...], t) . CX(c0, t)
    alpha_j = (theta_j + theta_{j + h}) / 2      beta_j = (theta_j - theta_{j + h}) / 2

which costs ``2**m`` rotations and ``2**m`` CX gates for ``m`` controls — the
exponential price of loading exponentially many numbers.

**Global phase.** The phase cascade reproduces every *relative* phase exactly and
drops one overall factor, which is unobservable. If you embed an amplitude-encoded
block inside a larger controlled circuit, that factor stops being global; use
``check=True`` to assert the prepared state matches your target up to phase.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.builder import QCircuit
from qmlkit.core.ir import CircuitSpec

__all__ = [
    "amplitude_encode",
    "pad_to_power_of_two",
    "uniformly_controlled_rotation",
    "state_preparation_angles",
]


def pad_to_power_of_two(vec: Sequence[float] | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Zero-pad a vector up to the next power of two."""
    arr = np.atleast_1d(np.asarray(vec)).ravel()
    if arr.size == 0:
        raise ValueError("cannot encode an empty vector")
    # at least one qubit, so the amplitude vector always matches the circuit width
    n = max(1, int(np.ceil(np.log2(arr.size))))
    size = 2**n
    if arr.size == size:
        return arr.astype(complex)
    out = np.zeros(size, dtype=complex)
    out[: arr.size] = arr
    return out


def uniformly_controlled_rotation(
    qc: QCircuit, rotation: str, angles: npt.NDArray[Any], controls: Sequence[int], target: int
) -> None:
    """Apply ``R(angles[k])`` to ``target`` for each control basis state ``k``.

    ``controls[0]`` is the most significant bit of ``k``. Emits only ``Ry``/``Rz``
    and ``CX``, so it works on any backend.
    """
    angles = np.asarray(angles, dtype=float).ravel()
    m = len(controls)
    if angles.size != 2**m:
        raise ValueError(f"expected {2**m} angles for {m} controls, got {angles.size}")

    if m == 0:
        if abs(angles[0]) > 1e-15:
            qc.apply(rotation, target, float(angles[0]))
        return

    half = angles.size // 2
    alpha = (angles[:half] + angles[half:]) / 2.0
    beta = (angles[:half] - angles[half:]) / 2.0
    rest = list(controls[1:])

    uniformly_controlled_rotation(qc, rotation, alpha, rest, target)
    qc.cx(controls[0], target)
    uniformly_controlled_rotation(qc, rotation, beta, rest, target)
    qc.cx(controls[0], target)


def state_preparation_angles(
    amplitudes: npt.NDArray[Any],
) -> tuple[list[npt.NDArray[Any]], list[npt.NDArray[Any]]]:
    """Ry angles per level (magnitudes) and Rz angles per level (phases)."""
    amps = np.asarray(amplitudes, dtype=complex).ravel()
    n = int(np.log2(amps.size))

    # --- magnitudes: a binary tree of partial norms -------------------------
    norms: list[npt.NDArray[Any]] = [np.abs(amps)]
    for _ in range(n):
        prev = norms[0]
        norms.insert(0, np.sqrt(prev[0::2] ** 2 + prev[1::2] ** 2))

    ry_angles: list[npt.NDArray[Any]] = []
    for level in range(n):
        parent = norms[level]
        child = norms[level + 1]
        # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, so the branch ratio
        # fixes theta = 2 * atan2(lower, upper). A zero-norm parent contributes
        # nothing observable, so its angle is free; zero keeps the circuit small.
        with np.errstate(invalid="ignore", divide="ignore"):
            theta = 2.0 * np.arctan2(child[1::2], child[0::2])
        theta = np.where(parent > 1e-15, theta, 0.0)
        ry_angles.append(theta)

    # --- phases: pair up, emit the difference, recurse on the mean ----------
    rz_angles: list[npt.NDArray[Any]] = []
    phases = np.angle(amps)
    if np.allclose(phases, 0.0, atol=1e-15):
        return ry_angles, rz_angles

    current = phases
    for _ in range(n):
        # Rz(a) = diag(e^{-ia/2}, e^{+ia/2}), so a = phi_upper_pair_difference
        rz_angles.insert(0, current[1::2] - current[0::2])
        current = (current[0::2] + current[1::2]) / 2.0  # leftover -> higher level
    return ry_angles, rz_angles


def amplitude_encode(
    vec: Sequence[float] | npt.NDArray[Any],
    normalize: bool = True,
    pad: bool = True,
    check: bool = False,
) -> CircuitSpec:
    """Encode a vector into the amplitudes of ``ceil(log2 len(vec))`` qubits.

    Only the *direction* of the vector survives — amplitudes must be normalised, so
    the magnitude is lost. ``normalize=False`` refuses a vector that is not already
    a unit vector rather than silently rescaling it.

    ``check=True`` re-simulates the circuit and asserts it prepares the intended
    state (up to global phase). Cheap insurance while you are getting a pipeline
    working; leave it off in a training loop.
    """
    arr = np.atleast_1d(np.asarray(vec)).ravel()
    if not pad and arr.size & (arr.size - 1):
        raise ValueError(f"length {arr.size} is not a power of two; pass pad=True to zero-fill")
    amps = pad_to_power_of_two(arr)

    norm = np.linalg.norm(amps)
    if norm < 1e-15:
        raise ValueError("cannot encode the zero vector: it has no direction")
    if normalize:
        amps = amps / norm
    elif not np.isclose(norm, 1.0, atol=1e-9):
        raise ValueError(f"vector has norm {norm:.6g}, not 1; pass normalize=True to rescale it")

    n_qubits = int(np.log2(amps.size))
    ry_angles, rz_angles = state_preparation_angles(amps)

    qc = QCircuit(n_qubits)
    for level, angles in enumerate(ry_angles):
        uniformly_controlled_rotation(qc, "ry", angles, list(range(level)), level)
    for level, angles in enumerate(rz_angles):
        uniformly_controlled_rotation(qc, "rz", angles, list(range(level)), level)
    spec = qc.to_spec()

    if check:
        from qmlkit.core.execute import statevector

        got = statevector(spec)
        overlap = abs(np.vdot(amps, got))
        if not np.isclose(overlap, 1.0, atol=1e-8):
            raise AssertionError(
                f"amplitude encoding produced a state with overlap {overlap:.12f}, expected 1"
            )
    return spec
