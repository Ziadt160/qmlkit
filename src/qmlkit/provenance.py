"""Two questions a result has to answer: is it right, and can it be reproduced.

A quantum machine learning number is produced by a stack — library version, SDK
version, backend, seed, shot count — where any layer can change the answer and none
of them is usually recorded. Six months later the same script gives a different
number and there is no way to tell which layer moved.

:func:`fingerprint` records the stack. :func:`selfcheck` asks whether the number was
right in the first place, by computing it more than one way.

The second is the more unusual. This library ships four *independent* exact routes
to a gradient — adjoint, backprop, Hadamard-test and parameter-shift — and any two
of them agreeing to machine precision is strong evidence that both are correct,
because they share almost no code. Disagreement localises a bug that no single
implementation could have caught::

    >>> import numpy as np, qmlkit as qk
    >>> a = qk.hardware_efficient(3, 2)
    >>> spec = a.build()
    >>> report = qk.selfcheck(spec, np.full(a.n_params, 0.3), qk.Z(0))
    >>> bool(report)          # falsy when every route agrees
    False

That is the parity idea from ``tests/test_pennylane_parity.py`` turned into
something a user can point at their own circuit.
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable

__all__ = ["Fingerprint", "fingerprint", "selfcheck"]

#: Four exact routes to the same gradient. Two agreeing is evidence; four is proof
#: enough for a working scientist.
_EXACT_METHODS = ("adjoint", "backprop", "hadamard", "parameter-shift")

#: Exact methods agree to machine precision. This is loose enough to survive the
#: accumulation over a few hundred gates and tight enough that a wrong shift rule,
#: a transposed matrix or a bit-order slip cannot hide under it.
_AGREEMENT = 1e-9

#: SpinQit's simulator carries a precision floor near 1e-10 rather than machine
#: precision, so a cross-backend comparison involving it needs more room.
_BACKEND_AGREEMENT = 1e-8


def _version(module_name: str) -> str | None:
    try:
        return str(getattr(import_module(module_name), "__version__", "installed"))
    except Exception:  # noqa: BLE001 - absence and breakage are the same answer here
        return None


@dataclass(frozen=True)
class Fingerprint:
    """Everything that could change a number, recorded in one object.

    Paste :meth:`as_dict` into a results file, or :func:`str` into a paper
    appendix. The point is that it is cheap enough to attach to every run.
    """

    qmlkit: str
    python: str
    platform: str
    numpy: str
    default_backend: str
    backends: dict[str, str | None] = field(default_factory=dict)
    optional: dict[str, str | None] = field(default_factory=dict)
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """A plain, JSON-serialisable mapping."""
        return {
            "qmlkit": self.qmlkit,
            "python": self.python,
            "platform": self.platform,
            "numpy": self.numpy,
            "default_backend": self.default_backend,
            "backends": dict(self.backends),
            "optional": dict(self.optional),
            "seed": self.seed,
            **self.extra,
        }

    def __str__(self) -> str:
        installed = {k: v for k, v in self.backends.items() if v is not None}
        lines = [
            f"qmlkit {self.qmlkit}  |  python {self.python}  |  numpy {self.numpy}",
            f"  platform         {self.platform}",
            f"  default backend  {self.default_backend}",
            "  backends         "
            + (", ".join(f"{k} {v}" for k, v in sorted(installed.items())) or "numpy only"),
        ]
        present = {k: v for k, v in self.optional.items() if v is not None}
        if present:
            lines.append(
                "  optional         " + ", ".join(f"{k} {v}" for k, v in sorted(present.items()))
            )
        if self.seed is not None:
            lines.append(f"  seed             {self.seed}")
        lines.extend(f"  {k:<16} {v}" for k, v in self.extra.items())
        return "\n".join(lines)


def fingerprint(seed: int | None = None, **extra: Any) -> Fingerprint:
    """The versions and settings that decide what a number comes out as.

    ``seed`` and any keyword extras are carried verbatim, so the run's own
    parameters — shot count, ansatz name, dataset — sit alongside the environment
    that produced them.
    """
    from qmlkit import __version__
    from qmlkit.core.backends.registry import default_backend

    try:
        current = default_backend().name
    except Exception:  # pragma: no cover - a broken default should not break the record
        current = "unavailable"

    return Fingerprint(
        qmlkit=__version__,
        python=sys.version.split()[0],
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        numpy=np.__version__,
        default_backend=current,
        backends={name: _version(name) for name in ("qiskit", "cirq", "spinqit")},
        optional={name: _version(name) for name in ("torch", "sklearn", "matplotlib")},
        seed=seed,
        extra=dict(extra),
    )


def _gradient_routes(
    spec: CircuitSpec, theta: ArrayLike, obs: Observable, backend: Any
) -> dict[str, npt.NDArray[Any]]:
    """Every exact gradient method that can run here, each computed independently."""
    from qmlkit.gradients.dispatch import grad

    out: dict[str, npt.NDArray[Any]] = {}
    for method in _EXACT_METHODS:
        try:
            out[method] = np.asarray(grad(spec, theta, obs, method=method, backend=backend), float)
        except Exception:  # noqa: BLE001 - an unavailable route is not a failure
            continue
    return out


def selfcheck(
    spec: CircuitSpec,
    theta: ArrayLike,
    obs: Observable,
    backend: Any = None,
    cross_backend: bool = True,
) -> Any:
    """Compute this circuit's value and gradient every available way, and compare.

    Returns a :class:`~qmlkit.diagnostics.Report`, falsy when everything agrees.

    Two independent checks run:

    * **Gradient routes.** Adjoint, backprop, Hadamard-test and parameter-shift are
      four separate derivations of the same quantity. They share the circuit IR and
      almost nothing else, so agreement is evidence and disagreement localises the
      wrong one — the method that stands alone against the others.
    * **Backends.** When more than one SDK is installed, the same circuit is run
      through each. This catches the translation-layer mistakes that no amount of
      testing against a single simulator can: endianness, controlled-gate qubit
      order, dropped idle qubits.

    ``cross_backend=False`` skips the second, which is the slower one.

    This is what to run when a number looks wrong and nothing raised.
    """
    from qmlkit.core.backends.registry import available_backends, get_backend
    from qmlkit.core.execute import expectation
    from qmlkit.diagnostics import Finding, Report

    values = np.asarray(theta, dtype=float)
    findings: list[Finding] = []

    routes = _gradient_routes(spec, values, obs, backend)
    if spec.n_params == 0:
        # every route returns an empty gradient, which agrees trivially and says
        # nothing. Report that rather than comparing zero-length arrays.
        findings.append(
            Finding(
                "selfcheck.one-route",
                "info",
                "this circuit has no parameters, so there is no gradient to cross-check",
                "selfcheck compares gradients; for a fixed circuit the backend "
                "comparison below is the whole check",
                0.0,
            )
        )
    elif len(routes) < 2:
        findings.append(
            Finding(
                "selfcheck.one-route",
                "info",
                f"only {len(routes)} exact gradient route could run here "
                f"({', '.join(routes) or 'none'}), so nothing was cross-checked",
                "pip install 'qmlkit[torch]' adds backprop as a second opinion",
                float(len(routes)),
            )
        )
    else:
        names = list(routes)
        reference = names[0]
        for name in names[1:]:
            delta = float(np.max(np.abs(routes[name] - routes[reference])))
            if delta > _AGREEMENT:
                findings.append(
                    Finding(
                        "selfcheck.gradient-disagreement",
                        "error",
                        f"{name} and {reference} disagree by {delta:.3e}, which is far above "
                        f"the {_AGREEMENT:.0e} these exact methods agree to. One of them is "
                        "computing something else",
                        "compare against a third method to see which one stands alone; a "
                        "custom gate with wrong `frequencies` is the usual cause",
                        delta,
                    )
                )

    if cross_backend:
        installed = [n for n in available_backends() if n != "numpy"]
        if installed:
            # expectation() widens to a tuple only with return_std=True, which is off
            reference_value = float(
                cast(float, expectation(spec, obs, values, backend="numpy"))
            )
            for name in installed:
                try:
                    other = float(
                        cast(float, expectation(spec, obs, values, backend=get_backend(name)))
                    )
                except Exception as exc:  # noqa: BLE001 - report, do not raise
                    findings.append(
                        Finding(
                            "selfcheck.backend-failed",
                            "warning",
                            f"the {name!r} backend could not run this circuit: {exc}",
                            "qk.backend_report() lists what is installed and working",
                        )
                    )
                    continue
                delta = abs(other - reference_value)
                if delta > _BACKEND_AGREEMENT:
                    findings.append(
                        Finding(
                            "selfcheck.backend-disagreement",
                            "error",
                            f"{name} gives {other:.12g} where the NumPy reference gives "
                            f"{reference_value:.12g} (difference {delta:.3e})",
                            f"qk.get_backend({name!r}).to_{name}(spec) shows the translated "
                            "circuit; bit order and controlled-gate qubit order are where "
                            "backends differ",
                            delta,
                        )
                    )

    subject = f"circuit ({spec.n_qubits} qubits, {spec.n_params} parameters)"
    return Report(subject, tuple(findings))
