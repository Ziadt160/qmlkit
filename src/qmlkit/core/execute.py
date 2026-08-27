"""Running circuits and reading answers out.

One entry point per question, each taking an optional ``theta`` so a parameterised
circuit can be run without binding it by hand first. ``shots=None`` means exact —
the default, because 0.x is simulator-only and paying for sampling noise you did
not ask for is not a feature. Pass ``shots=N`` to model a real device.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.core.backends.base import Backend
from qmlkit.core.backends.registry import get_backend
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z
from qmlkit.utils.shots import standard_error

BackendLike = str | Backend | None

__all__ = [
    "statevector",
    "expval",
    "run_counts",
    "probabilities",
    "expectation",
    "expectation_batch",
]


def _prepare(spec: CircuitSpec, theta: ArrayLike | None) -> CircuitSpec:
    if spec.is_bound:
        if theta is not None and spec.n_params == 0 and len(np.atleast_1d(theta)) > 0:
            raise ValueError("circuit has no free parameters, but theta was given")
        return spec
    return spec.bind(theta)


def statevector(
    spec: CircuitSpec,
    theta: ArrayLike | None = None,
    backend: BackendLike = None,
) -> npt.NDArray[Any]:
    """Final state as a flat ``2**n`` complex vector."""
    return get_backend(backend).statevector(_prepare(spec, theta))


def run_counts(
    spec: CircuitSpec,
    shots: int = 8192,
    theta: ArrayLike | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> dict[str, int]:
    """Sample the computational basis. Keys are ``n_qubits``-wide bitstrings."""
    return get_backend(backend).counts(_prepare(spec, theta), shots, seed)


def probabilities(
    spec: CircuitSpec,
    theta: ArrayLike | None = None,
    backend: BackendLike = None,
) -> npt.NDArray[Any]:
    """Exact outcome probabilities over the ``2**n`` basis states."""
    return get_backend(backend).probabilities(_prepare(spec, theta))


def expectation(
    spec: CircuitSpec,
    obs: Observable | None = None,
    theta: ArrayLike | None = None,
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
    return_std: bool = False,
) -> float | tuple[float, float]:
    """``<O>`` for a circuit.

    ``shots=None`` (default) returns the exact value. With ``shots=N`` the value is
    sampled; ``return_std=True`` then also gives the standard error, which is the
    honest thing to report alongside any sampled number.
    """
    obs = Z(0) if obs is None else obs
    value = get_backend(backend).expectation(_prepare(spec, theta), obs, shots, seed)
    if not return_std:
        return value
    if shots is None:
        return value, 0.0
    return value, standard_error(value, shots)


def expectation_batch(
    specs: Sequence[CircuitSpec],
    obs: Observable | None = None,
    thetas: Sequence[ArrayLike] | None = None,
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """``<O>`` for several circuits, resolving the backend once."""
    be = get_backend(backend)
    obs = Z(0) if obs is None else obs
    if thetas is None:
        thetas = [None] * len(specs)  # type: ignore[list-item]
    if len(thetas) != len(specs):
        raise ValueError(f"got {len(specs)} circuits but {len(thetas)} parameter vectors")
    return np.array(
        [
            be.expectation(_prepare(s, t), obs, shots, seed)
            for s, t in zip(specs, thetas, strict=False)
        ],
        dtype=float,
    )


def expectation_over(
    spec: CircuitSpec,
    thetas: ArrayLike,
    obs: Observable | None = None,
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """``<O>`` for **one** circuit at many parameter vectors — the batched path.

    ``expectation_batch`` takes many circuits; this takes one circuit and a
    ``(batch, n_params)`` array, which is the shape a training loop actually has: the
    same ansatz, one parameter vector per sample, because the encoding differs per
    sample and the weights do not.

    Knowing the structure is shared is what lets a backend do better than a loop. The
    NumPy backend carries the batch as a leading axis and applies each gate to the
    whole stack at once, which is 4-30x faster than one-at-a-time up to 10 qubits.
    Backends that cannot do better inherit a loop, so this is always correct and never
    slower than calling :func:`expectation` yourself.

        >>> import numpy as np, qmlkit as qk
        >>> a = qk.hardware_efficient(3, 2)
        >>> thetas = np.zeros((4, a.n_params))
        >>> qk.expectation_over(a.build(), thetas, qk.Z(0)).shape
        (4,)
    """
    return get_backend(backend).expectation_over(
        spec, np.atleast_2d(np.asarray(thetas, dtype=float)), Z(0) if obs is None else obs,
        shots, seed,
    )


def expval(
    spec: CircuitSpec,
    obs: Observable | None = None,
    theta: ArrayLike | None = None,
    shots: int | None = None,
    backend: BackendLike = None,
    seed: int | None = None,
) -> float:
    """``<O>`` as a plain float -- :func:`expectation` without the optional error bar."""
    value = expectation(spec, obs, theta, shots, backend, seed, return_std=False)
    assert not isinstance(value, tuple)
    return float(value)
