"""Backend lookup, availability detection, and the process-wide default.

Every SDK is optional. Backends are constructed lazily and their imports deferred,
so ``import qmlkit`` never requires Qiskit, Cirq or SpinQit to be installed - and
asking for one that is missing produces an explanation and an install command
rather than an ``ImportError`` traceback.

``QMLKIT_BACKEND`` in the environment sets the default, which is the least
intrusive way to run an existing script against a different SDK.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from collections.abc import Callable

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.backends.numpy_backend import NumpyBackend
from qmlkit.utils.errors import unknown

__all__ = [
    "register_backend",
    "get_backend",
    "list_backends",
    "available_backends",
    "is_available",
    "default_backend",
    "set_default_backend",
    "backend_report",
]

#: backend name -> (factory, module it needs, pip extra that provides it)
_REGISTRY: dict[str, tuple[Callable[..., Backend], str | None, str | None]] = {}
_default: Backend | None = None


def register_backend(
    name: str,
    factory: Callable[..., Backend],
    requires: str | None = None,
    extra: str | None = None,
) -> None:
    """Register a backend factory.

    ``requires`` is the importable module the backend needs; ``extra`` is the pip
    extra that installs it. Both are used to report availability without importing.
    """
    _REGISTRY[name] = (factory, requires, extra)


def _lazy(module: str, cls_name: str) -> Callable[..., Backend]:
    """Build a factory that imports the backend module only when called."""

    def factory(**kwargs: object) -> Backend:
        mod = importlib.import_module(f"qmlkit.core.backends.{module}")
        cls = getattr(mod, cls_name)
        backend: Backend = cls(**kwargs)
        return backend

    return factory


register_backend("numpy", NumpyBackend)
register_backend("spinqit", _lazy("spinqit_backend", "SpinQitBackend"), "spinqit", "spinqit")
register_backend("qiskit", _lazy("qiskit_backend", "QiskitBackend"), "qiskit", "qiskit")
register_backend("cirq", _lazy("cirq_backend", "CirqBackend"), "cirq", "cirq")
register_backend("torch", _lazy("torch_backend", "TorchBackend"), "torch", "torch")


# --------------------------------------------------------------- availability
def is_available(name: str) -> bool:
    """True if this backend's SDK can be imported in the current interpreter."""
    if name not in _REGISTRY:
        return False
    _, requires, _ = _REGISTRY[name]
    if requires is None:
        return True
    try:
        return importlib.util.find_spec(requires) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs
        return False


def list_backends() -> tuple[str, ...]:
    """Every registered backend name, installed or not."""
    return tuple(sorted(_REGISTRY))


def available_backends() -> tuple[str, ...]:
    """Only the backends whose SDK is actually importable right now."""
    return tuple(name for name in list_backends() if is_available(name))


def backend_report() -> str:
    """A human-readable summary of which backends this interpreter can run."""
    lines = ["qmlkit backends:"]
    for name in list_backends():
        _, requires, extra = _REGISTRY[name]
        if is_available(name):
            lines.append(f"  [ok]      {name}")
        else:
            hint = f"pip install 'qmlkit[{extra}]'" if extra else f"pip install {requires}"
            lines.append(f"  [missing] {name:8s} -> {hint}")
    return "\n".join(lines)


# ------------------------------------------------------------------- lookup
def get_backend(backend: str | Backend | None = None, **kwargs: object) -> Backend:
    """Resolve a backend name, instance, or ``None`` (the default) to an instance."""
    if isinstance(backend, Backend):
        return backend
    if backend is None:
        return default_backend()
    try:
        factory, requires, extra = _REGISTRY[backend]
    except KeyError:
        raise unknown(
            "backend",
            backend,
            list_backends(),
            hint=f"Importable in this interpreter right now: {', '.join(available_backends())}.",
            error=KeyError,
        ) from None
    if requires is not None and not is_available(backend):
        hint = f"pip install 'qmlkit[{extra}]'" if extra else f"pip install {requires}"
        raise BackendNotAvailable(
            f"the {backend!r} backend needs {requires!r}, which is not installed here.\n"
            f"    {hint}\n"
            f"Available now: {', '.join(available_backends())}"
        )
    return factory(**kwargs)


def default_backend() -> Backend:
    """The process-wide default.

    NumPy unless ``QMLKIT_BACKEND`` says otherwise - exact, always present, and the
    reference every other backend is tested against.
    """
    global _default
    if _default is None:
        requested = os.environ.get("QMLKIT_BACKEND")
        _default = get_backend(requested) if requested else NumpyBackend()
    return _default


def set_default_backend(backend: str | Backend, **kwargs: object) -> Backend:
    """Set the process-wide default backend and return it."""
    global _default
    _default = get_backend(backend, **kwargs)
    return _default
