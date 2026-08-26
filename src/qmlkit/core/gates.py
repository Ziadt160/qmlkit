"""Gate registry: matrices, adjoints, and — critically — generator frequencies.

The ``frequencies`` field is what keeps the parameter-shift rule correct. A gate
of the form ``exp(-i θ G / 2)`` has a derivative determined entirely by the set of
unique positive differences between the eigenvalues of its generator. Declare that
set and :mod:`qmlkit.gradients` derives the right shift rule automatically; omit it
and differentiation of that gate is refused rather than silently wrong.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

Matrix = npt.NDArray[Any]


@dataclass(frozen=True)
class GateDef:
    """Everything the library needs to know about one gate."""

    name: str
    n_qubits: int
    n_params: int
    matrix: Callable[..., Matrix]
    #: unique positive eigenvalue gaps of the generator, in units of 1/theta
    frequencies: tuple[float, ...] = ()
    #: exact d(matrix)/d(theta), which the adjoint gradient needs
    dmatrix: Callable[..., Matrix] | None = None
    #: name of the inverse for non-parameterised gates (None => self-inverse)
    adjoint_name: str | None = None
    aliases: tuple[str, ...] = field(default=())

    @property
    def is_parametric(self) -> bool:
        return self.n_params > 0

    @property
    def is_differentiable(self) -> bool:
        return bool(self.frequencies)

    @property
    def has_derivative(self) -> bool:
        """True if the exact derivative matrix is known (adjoint differentiation)."""
        return self.dmatrix is not None


# --------------------------------------------------------------------------- #
# constant matrices
# --------------------------------------------------------------------------- #
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_TDG = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=complex)


def _rx(theta: float) -> Matrix:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta: float) -> Matrix:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> Matrix:
    e = np.exp(-1j * theta / 2)
    return np.array([[e, 0], [0, np.conj(e)]], dtype=complex)


def _phase(theta: float) -> Matrix:
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def _controlled(sub: Matrix) -> Matrix:
    """Lift a 1-qubit matrix to a 2-qubit controlled gate (control = first wire)."""
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = sub
    return out


_CX = _controlled(_X)
_CY = _controlled(_Y)
_CZ = _controlled(_Z)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


# --------------------------------------------------------------------------- #
# exact derivatives
#
# For U(theta) = exp(-i theta P / 2) the derivative is -i/2 * P * U -- no finite
# differencing anywhere in the adjoint path, which is what keeps it *exact* rather
# than merely fast.
# --------------------------------------------------------------------------- #
def _d_rx(theta: float) -> Matrix:
    return -0.5j * _X @ _rx(theta)


def _d_ry(theta: float) -> Matrix:
    return -0.5j * _Y @ _ry(theta)


def _d_rz(theta: float) -> Matrix:
    return -0.5j * _Z @ _rz(theta)


def _d_phase(theta: float) -> Matrix:
    return np.array([[0, 0], [0, 1j * np.exp(1j * theta)]], dtype=complex)


def _d_controlled(sub_derivative: Matrix) -> Matrix:
    """Only the control-1 block varies, so the control-0 block differentiates to 0."""
    out = np.zeros((4, 4), dtype=complex)
    out[2:, 2:] = sub_derivative
    return out


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
_REGISTRY: dict[str, GateDef] = {}


def register_gate(gate: GateDef) -> GateDef:
    """Add a gate to the registry. Re-registering the same name is an error."""
    for key in (gate.name, *gate.aliases):
        if key in _REGISTRY:
            raise ValueError(f"gate {key!r} is already registered")
    for key in (gate.name, *gate.aliases):
        _REGISTRY[key] = gate
    return gate


def get_gate(name: str) -> GateDef:
    try:
        return _REGISTRY[name.lower()]
    except KeyError:
        raise KeyError(
            f"unknown gate {name!r}; known gates: {', '.join(sorted(set(_REGISTRY)))}"
        ) from None


def list_gates() -> tuple[str, ...]:
    return tuple(sorted({g.name for g in _REGISTRY.values()}))


def _const(name: str, n_qubits: int, m: Matrix, **kw: object) -> None:
    register_gate(GateDef(name, n_qubits, 0, lambda mm=m: mm, **kw))  # type: ignore[arg-type]


# Pauli rotations: generator eigenvalues ±1/2, so the unique positive gap is 1.
# One frequency => the familiar two-term ±π/2 rule.
register_gate(GateDef("rx", 1, 1, _rx, frequencies=(1.0,), dmatrix=_d_rx))
register_gate(GateDef("ry", 1, 1, _ry, frequencies=(1.0,), dmatrix=_d_ry))
register_gate(GateDef("rz", 1, 1, _rz, frequencies=(1.0,), dmatrix=_d_rz))
# phase/P: generator eigenvalues {0, 1}, gap 1 -- also one frequency.
register_gate(GateDef("phase", 1, 1, _phase, frequencies=(1.0,), dmatrix=_d_phase, aliases=("p",)))

# Controlled rotations: generator eigenvalues {0, 0, +1/2, -1/2}. Unique positive
# gaps are 1/2 and 1 -- TWO frequencies, so these need the four-term rule. This is
# exactly the mixture that a single global shift rule would get silently wrong.
register_gate(
    GateDef(
        "crx",
        2,
        1,
        lambda t: _controlled(_rx(t)),
        frequencies=(0.5, 1.0),
        dmatrix=lambda t: _d_controlled(_d_rx(t)),
    )
)
register_gate(
    GateDef(
        "cry",
        2,
        1,
        lambda t: _controlled(_ry(t)),
        frequencies=(0.5, 1.0),
        dmatrix=lambda t: _d_controlled(_d_ry(t)),
    )
)
register_gate(
    GateDef(
        "crz",
        2,
        1,
        lambda t: _controlled(_rz(t)),
        frequencies=(0.5, 1.0),
        dmatrix=lambda t: _d_controlled(_d_rz(t)),
    )
)

_const("i", 1, _I, aliases=("id",))
_const("x", 1, _X)
_const("y", 1, _Y)
_const("z", 1, _Z)
_const("h", 1, _H)
_const("s", 1, _S, adjoint_name="sdg")
_const("sdg", 1, _SDG, adjoint_name="s")
_const("t", 1, _T, adjoint_name="tdg")
_const("tdg", 1, _TDG, adjoint_name="t")
_const("cx", 2, _CX, aliases=("cnot",))
_const("cy", 2, _CY)
_const("cz", 2, _CZ)
_const("swap", 2, _SWAP)


def gate_derivative(name: str, params: Sequence[float] = ()) -> Matrix:
    """Exact d(matrix)/d(theta) for a one-parameter gate."""
    g = get_gate(name)
    if g.dmatrix is None:
        raise ValueError(
            f"gate {name!r} has no derivative matrix, so it cannot be differentiated by "
            "the adjoint method. Register it with dmatrix=..., or use "
            'grad_method="parameter-shift".'
        )
    if len(params) != g.n_params:
        raise ValueError(f"gate {name!r} takes {g.n_params} parameter(s), got {len(params)}")
    return g.dmatrix(*params)


def gate_matrix(name: str, params: Sequence[float] = ()) -> Matrix:
    """Return the unitary for ``name`` bound to ``params``."""
    g = get_gate(name)
    if len(params) != g.n_params:
        raise ValueError(f"gate {name!r} takes {g.n_params} parameter(s), got {len(params)}")
    return g.matrix(*params)
