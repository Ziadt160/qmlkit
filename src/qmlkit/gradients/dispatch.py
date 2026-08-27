"""One gradient function, several methods, and a registry for adding more.

    qk.grad(spec, theta, obs)                      # picks the right method for you
    qk.grad(spec, theta, obs, method="parameter-shift")

``method="auto"`` (the default) uses adjoint when every gate has a closed-form
derivative and the backend can hand back a statevector — exact, and independent of
the parameter count. It falls back to parameter-shift otherwise, which is exact
too, just ``2P`` circuits instead of one pass.

A researcher with a different estimator registers it and it becomes a keyword
everywhere the library takes ``method=``:

    @register_gradient("my_estimator")
    def my_estimator(spec, theta, obs, *, backend=None, shots=None, **kw):
        ...
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.core.backends.base import Backend
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z
from qmlkit.utils.errors import unknown

__all__ = [
    "grad",
    "register_gradient",
    "list_gradient_methods",
    "choose_method",
    "hessian",
    "gradient_cost",
]

GradFn = Callable[..., npt.NDArray[Any]]
_METHODS: dict[str, GradFn] = {}


def register_gradient(name: str, fn: GradFn | None = None) -> Callable[[GradFn], GradFn] | GradFn:
    """Register a gradient estimator under ``name``. Usable as a decorator."""

    def _register(f: GradFn) -> GradFn:
        if name in _METHODS:
            raise ValueError(f"gradient method {name!r} is already registered")
        _METHODS[name] = f
        return f

    return _register if fn is None else _register(fn)


def list_gradient_methods() -> tuple[str, ...]:
    return tuple(sorted(_METHODS))


def choose_method(
    spec: CircuitSpec, backend: Backend | str | None = None, shots: int | None = None
) -> str:
    """What ``method="auto"`` resolves to, and why.

    Sampling means the statevector is off the table by definition, so shots force
    parameter-shift.
    """
    if shots is not None:
        return "parameter-shift"
    from qmlkit.gradients.adjoint import supports_adjoint

    return "adjoint" if supports_adjoint(spec, backend) else "parameter-shift"


def grad(
    spec: CircuitSpec,
    theta: ArrayLike,
    obs: Observable | None = None,
    method: str = "auto",
    backend: Backend | str | None = None,
    shots: int | None = None,
    **kwargs: object,
) -> npt.NDArray[Any]:
    """Gradient of ``<obs>`` with respect to ``theta``.

    Parameters
    ----------
    method
        ``auto`` (default), ``adjoint``, ``parameter-shift``, ``spsa``, or
        ``finite-diff``. Registered methods are accepted by name too.
    shots
        Sampling budget. Anything other than ``None`` rules out adjoint.
    """
    obs = Z(0) if obs is None else obs
    resolved = choose_method(spec, backend, shots) if method == "auto" else method
    try:
        fn = _METHODS[resolved]
    except KeyError:
        raise unknown(
            "gradient method",
            resolved,
            list_gradient_methods(),
            hint='"auto" chooses for you; add your own with register_gradient(name, fn).',
            error=KeyError,
        ) from None
    return fn(spec, np.asarray(theta, dtype=float), obs, backend=backend, shots=shots, **kwargs)


# --------------------------------------------------------------------------- #
# the built-in methods
# --------------------------------------------------------------------------- #
@register_gradient("adjoint")
def _adjoint(spec, theta, obs, *, backend=None, shots=None, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.gradients.adjoint import adjoint_grad

    if shots is not None:
        raise ValueError(
            "adjoint differentiation reads the statevector, so it cannot honour a shot "
            'budget; use method="parameter-shift" with shots=N'
        )
    return adjoint_grad(spec, theta, obs, backend)


@register_gradient("parameter-shift")
def _param_shift(spec, theta, obs, *, backend=None, shots=None, seed=None, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.gradients.parameter_shift import param_shift_grad_circuit

    return param_shift_grad_circuit(spec, theta, obs, shots=shots, backend=backend, seed=seed)


@register_gradient("finite-diff")
def _finite_diff(spec, theta, obs, *, backend=None, shots=None, eps=1e-6, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.core.execute import expval
    from qmlkit.gradients.parameter_shift import finite_diff_grad

    def f(t: npt.NDArray[Any]) -> float:
        return expval(spec, obs, theta=t, shots=shots, backend=backend)

    return finite_diff_grad(f, theta, eps=eps)


@register_gradient("spsa")
def _spsa(spec, theta, obs, *, backend=None, shots=None, c=0.1, n_avg=1, seed=None, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.core.execute import expval
    from qmlkit.gradients.spsa import spsa_grad

    def f(t: npt.NDArray[Any]) -> float:
        return expval(spec, obs, theta=t, shots=shots, backend=backend, seed=seed)

    return spsa_grad(f, theta, c=c, n_avg=n_avg, seed=seed)


@register_gradient("hadamard")
def _hadamard(spec, theta, obs, *, backend=None, shots=None, seed=None, **kw):  # type: ignore[no-untyped-def]
    from qmlkit.gradients.hadamard import hadamard_grad

    return hadamard_grad(spec, theta, obs, backend=backend, shots=shots, seed=seed)


@register_gradient("backprop")
def _backprop(spec, theta, obs, *, backend=None, shots=None, **kw):  # type: ignore[no-untyped-def]
    """Differentiate the torch statevector simulator directly."""
    from qmlkit.core.backends.base import BackendNotAvailable

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        # `backprop` is registered unconditionally, so it shows up in
        # list_gradient_methods() on a bare install too. Anyone iterating those
        # methods deserves the install command rather than a raw ModuleNotFoundError.
        raise BackendNotAvailable(
            "the backprop gradient needs PyTorch:\n    pip install 'qmlkit[torch]'"
        ) from exc

    from qmlkit.core.backends.torch_backend import torch_expectation

    if shots is not None:
        raise ValueError(
            "backprop differentiates the statevector, so it cannot honour a shot "
            'budget; use method="parameter-shift" with shots=N'
        )
    t = torch.tensor(np.asarray(theta, dtype=float), dtype=torch.float64, requires_grad=True)
    torch_expectation(spec, t, obs).backward()
    assert t.grad is not None
    return t.grad.detach().cpu().numpy()


def hessian(
    spec: CircuitSpec,
    theta: Sequence[float],
    obs: Observable | None = None,
    backend: Backend | str | None = None,
    eps: float = 1e-4,
) -> npt.NDArray[Any]:
    """Second derivatives, by differencing the exact gradient.

    The gradient itself is exact (adjoint), so only the outer derivative is
    approximated — far more accurate than differencing the expectation twice.
    """
    obs = Z(0) if obs is None else obs
    arr = np.asarray(theta, dtype=float).ravel()
    p = arr.size
    out = np.zeros((p, p))
    for k in range(p):
        plus, minus = arr.copy(), arr.copy()
        plus[k] += eps
        minus[k] -= eps
        out[k] = (
            grad(spec, plus, obs, backend=backend) - grad(spec, minus, obs, backend=backend)
        ) / (2 * eps)
    return 0.5 * (out + out.T)  # symmetrise away the differencing asymmetry


def gradient_cost(spec: CircuitSpec, method: str = "parameter-shift") -> int | str:
    """Circuit evaluations one gradient needs under a given method."""
    from qmlkit.gradients.hadamard import hadamard_grad_cost
    from qmlkit.gradients.parameter_shift import grad_circuit_cost

    costs = {
        "adjoint": 1,
        "backprop": 1,
        "hadamard": hadamard_grad_cost(spec),
        "parameter-shift": grad_circuit_cost(spec),
        "finite-diff": 2 * spec.n_params,
        "spsa": 2,
    }
    return costs.get(method, "unknown")
