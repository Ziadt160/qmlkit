r"""What function does this model actually represent?

A variational model with data re-uploading is a **truncated Fourier series** in its
inputs. The encoding fixes which frequencies are reachable; the ansatz only sets
their coefficients. That is the central claim of the re-uploading literature, and
this module turns it from a claim into a measurement:

    coeffs = fourier_coefficients(f, degree=4)
    spectrum(f)                 # which frequencies actually carry weight

Useful for two things. Checking that ``L`` uploads really did buy frequencies
``0..L`` — and diagnosing a model that will not fit its target, because if the
target's frequency is not in the spectrum, no amount of training will reach it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "fourier_coefficients",
    "spectrum",
    "reconstruct",
    "reachable_frequencies",
    "dominant_frequency",
]

ScalarFn = Callable[[float], float]


def fourier_coefficients(
    f: ScalarFn, degree: int = 5, n_samples: int | None = None
) -> npt.NDArray[Any]:
    """Complex Fourier coefficients ``c_-d .. c_d`` of a ``2*pi``-periodic function.

    Sampled on a uniform grid and transformed exactly — no fitting, no optimiser.
    The grid must be at least ``2*degree + 1`` points to avoid aliasing, which is
    the default.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    n = n_samples or (2 * degree + 1)
    if n < 2 * degree + 1:
        raise ValueError(
            f"{n} samples cannot resolve degree {degree}; need at least {2 * degree + 1}"
        )
    grid = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    values = np.array([float(f(x)) for x in grid], dtype=complex)
    full = np.fft.fft(values) / n
    return np.concatenate([full[-degree:], full[: degree + 1]]) if degree else full[:1]


def spectrum(
    f: ScalarFn, degree: int = 5, n_samples: int | None = None, tol: float = 1e-8
) -> dict[int, float]:
    """Frequency -> amplitude, keeping only what is actually present.

    Amplitudes are ``|c_k| + |c_-k|`` for ``k > 0``, so a real-valued model reports
    one number per frequency rather than a conjugate pair.
    """
    coeffs = fourier_coefficients(f, degree, n_samples)
    out: dict[int, float] = {}
    zero = degree
    if abs(coeffs[zero]) > tol:
        out[0] = float(abs(coeffs[zero]))
    for k in range(1, degree + 1):
        amp = float(abs(coeffs[zero + k]) + abs(coeffs[zero - k]))
        if amp > tol:
            out[k] = amp
    return out


def reconstruct(coeffs: npt.NDArray[Any], x: float | npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Evaluate the series these coefficients describe."""
    c = np.asarray(coeffs, dtype=complex)
    degree = (c.size - 1) // 2
    freqs = np.arange(-degree, degree + 1)
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    return np.real(np.exp(1j * np.outer(xs, freqs)) @ c)


def reachable_frequencies(n_uploads: int) -> list[int]:
    """``L`` uploads of a Pauli-rotation encoding reach frequencies ``0..L``."""
    if n_uploads < 0:
        raise ValueError("n_uploads cannot be negative")
    return list(range(n_uploads + 1))


def dominant_frequency(f: ScalarFn, degree: int = 8) -> int:
    """The non-zero frequency carrying the most weight — 0 if the model is constant."""
    spec = {k: v for k, v in spectrum(f, degree).items() if k != 0}
    return max(spec, key=lambda k: spec[k]) if spec else 0


def model_spectrum(
    encoder: object,
    theta: Sequence[float],
    obs: object | None = None,
    degree: int | None = None,
    backend: object = None,
) -> dict[int, float]:
    """Spectrum of a one-feature re-uploading model, as a function of its input.

    The direct check that an encoder buys the frequencies it claims to.
    """
    from qmlkit.core.execute import expval
    from qmlkit.core.observables import Z

    obs = Z(0) if obs is None else obs
    n_uploads = getattr(encoder, "n_uploads", 1)
    deg = degree if degree is not None else n_uploads + 2

    def f(x: float) -> float:
        return expval(encoder.build([x]), obs, theta=theta, backend=backend)  # type: ignore[attr-defined]

    return spectrum(f, deg)
