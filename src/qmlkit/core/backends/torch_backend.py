"""Differentiable statevector simulator in PyTorch.

Every gate is built from torch tensors, so the whole circuit sits inside autograd's
graph and ``backward()`` differentiates it directly. That is **backpropagation** —
the fastest gradient available on a simulator, and the one PennyLane reaches for by
default on ``default.qubit``.

It is also the least physical. Backprop reads intermediate states that no device
will ever expose, and its memory grows with circuit depth because every intermediate
is retained. Adjoint gets the same exact answer at constant memory; parameter-shift
gets it from measurements alone. This backend is here because it is fast and because
it makes ``method="backprop"`` a real option, not because it could ever run anywhere
but a simulator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.ir import CircuitSpec, ParamRef
from qmlkit.core.observables import Observable, Z, as_sum

__all__ = ["TorchBackend", "torch_expectation"]


def _gate_tensor(torch: Any, gate: str, angles: list[Any]) -> Any:
    """Build a gate as a torch tensor, keeping any angle inside the graph."""
    c64 = torch.complex128

    def const(rows: list[list[complex]]) -> Any:
        return torch.tensor(rows, dtype=c64)

    if gate in ("i", "id"):
        return const([[1, 0], [0, 1]])
    if gate == "x":
        return const([[0, 1], [1, 0]])
    if gate == "y":
        return const([[0, -1j], [1j, 0]])
    if gate == "z":
        return const([[1, 0], [0, -1]])
    if gate == "h":
        return const([[1, 1], [1, -1]]) / np.sqrt(2)
    if gate == "s":
        return const([[1, 0], [0, 1j]])
    if gate == "sdg":
        return const([[1, 0], [0, -1j]])
    if gate == "t":
        return const([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    if gate == "tdg":
        return const([[1, 0], [0, np.exp(-1j * np.pi / 4)]])
    if gate == "cx":
        return const([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    if gate == "cy":
        return const([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]])
    if gate == "cz":
        return const([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
    if gate == "swap":
        return const([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

    theta = angles[0]
    if not torch.is_tensor(theta):
        theta = torch.tensor(float(theta), dtype=torch.float64)
    theta = theta.to(torch.float64)
    half = theta / 2
    cos, sin = torch.cos(half).to(c64), torch.sin(half).to(c64)
    i = torch.tensor(1j, dtype=c64)

    if gate == "rx":
        return torch.stack([torch.stack([cos, -i * sin]), torch.stack([-i * sin, cos])])
    if gate == "ry":
        return torch.stack([torch.stack([cos, -sin]), torch.stack([sin, cos])])
    if gate == "rz":
        e = torch.exp(-i * half.to(c64))
        zero = torch.zeros((), dtype=c64)
        return torch.stack([torch.stack([e, zero]), torch.stack([zero, torch.conj(e)])])
    if gate in ("phase", "p"):
        e = torch.exp(i * theta.to(c64))
        one = torch.ones((), dtype=c64)
        zero = torch.zeros((), dtype=c64)
        return torch.stack([torch.stack([one, zero]), torch.stack([zero, e])])
    if gate in ("crx", "cry", "crz"):
        sub = _gate_tensor(torch, gate[1:], [theta])
        top = torch.zeros((2, 2), dtype=c64)
        return torch.cat(
            [
                torch.cat([torch.eye(2, dtype=c64), top], dim=1),
                torch.cat([top, sub], dim=1),
            ],
            dim=0,
        )
    raise NotImplementedError(f"gate {gate!r} has no torch tensor form")


def _apply_torch(torch: Any, state: Any, matrix: Any, qubits: tuple[int, ...]) -> Any:
    k = len(qubits)
    op = matrix.reshape((2,) * (2 * k))
    state = torch.tensordot(op, state, dims=([*range(k, 2 * k)], list(qubits)))
    perm = list(range(k, state.dim()))
    for slot, q in enumerate(qubits):
        perm.insert(q, slot)
    return state.permute(perm)


def torch_expectation(spec: CircuitSpec, theta: Any, obs: Observable | None = None) -> Any:
    """``<obs>`` as a torch scalar, differentiable through ``theta``."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise BackendNotAvailable(
            "the torch backend needs PyTorch:\n    pip install 'qmlkit[torch]'"
        ) from exc

    obs = Z(0) if obs is None else obs
    n = spec.n_qubits
    if not torch.is_tensor(theta):
        theta = torch.tensor(np.asarray(theta, dtype=float), dtype=torch.float64)

    slot_angles = []
    for slot in spec.slots():
        ref = slot.ref
        slot_angles.append(ref.scale * theta[ref.index] + ref.offset)

    state = torch.zeros((2,) * n, dtype=torch.complex128)
    state[(0,) * n] = 1.0

    cursor = 0
    for op in spec.ops:
        angles: list[Any] = []
        for p in op.params:
            if isinstance(p, ParamRef):
                angles.append(slot_angles[cursor])
                cursor += 1
            else:
                angles.append(float(p))
        state = _apply_torch(torch, state, _gate_tensor(torch, op.gate, angles), op.qubits)

    flat = state.reshape(-1)
    total = torch.zeros((), dtype=torch.complex128)
    for term in as_sum(obs).terms:
        out = state
        for qubit, pauli in term.paulis:
            if pauli == "I":
                continue
            out = _apply_torch(torch, out, _gate_tensor(torch, pauli.lower(), []), (qubit,))
        total = total + term.coeff * torch.sum(torch.conj(flat) * out.reshape(-1))
    return torch.real(total)


class TorchBackend(Backend):
    """Statevector simulation in torch — differentiable end to end."""

    name = "torch"
    supports_statevector = True
    supports_exact = True

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(seed)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise BackendNotAvailable(
                "the torch backend needs PyTorch:\n    pip install 'qmlkit[torch]'"
            ) from exc
        self._torch = torch

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        self._check_bound(spec)
        torch = self._torch
        n = spec.n_qubits
        state = torch.zeros((2,) * n, dtype=torch.complex128)
        state[(0,) * n] = 1.0
        for op in spec.ops:
            angles = [float(p) for p in op.params if not isinstance(p, ParamRef)]
            state = _apply_torch(torch, state, _gate_tensor(torch, op.gate, angles), op.qubits)
        return state.reshape(-1).detach().cpu().numpy()

    def expectation_tensor(
        self, spec: CircuitSpec, theta: Any, obs: Observable | None = None
    ) -> Any:
        """The differentiable path — returns a torch scalar, not a float."""
        return torch_expectation(spec, theta, obs)
