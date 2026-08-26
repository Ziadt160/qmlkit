"""Backends: one protocol, interchangeable implementations."""

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.backends.numpy_backend import NumpyBackend
from qmlkit.core.backends.registry import (
    default_backend,
    get_backend,
    list_backends,
    register_backend,
    set_default_backend,
)

__all__ = [
    "Backend",
    "BackendNotAvailable",
    "NumpyBackend",
    "get_backend",
    "default_backend",
    "set_default_backend",
    "register_backend",
    "list_backends",
]
