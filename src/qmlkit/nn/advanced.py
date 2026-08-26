"""QCNN, QLSTM and MPS layers — architectures with structure, not just depth.

Each is an ordinary ``nn.Module`` built from a :class:`QuantumLayer`, so they train
the same way and compose with anything else in torch. What distinguishes them is
*where* the structure lives:

* **QCNN** — a convolution filter shared across every pair, then pooling that halves
  the register. Log-depth in the qubit count, and few parameters because the filter
  is tied. Provably free of the exponential barren plateau.
* **QLSTM** — four small circuits standing in for the gates of an LSTM cell. The
  recurrence and nonlinearities stay classical; only the gates are quantum.
* **MPS** — a staircase of two-qubit blocks, matching a bond-dimension-2 matrix
  product state. Linear depth, and classically simulable at small bond dimension,
  which is worth knowing before claiming an advantage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from qmlkit.ansatz.library import Ansatz, mps_ansatz, qcnn_ansatz
from qmlkit.core.observables import Observable, Z
from qmlkit.encoding.feature_maps import AngleFeatureMap, FeatureMap
from qmlkit.nn.layer import QuantumLayer

__all__ = ["QCNNLayer", "MPSLayer", "QLSTMCell", "QLSTM", "DressedQuantumNet"]


def _default_map(n_qubits: int) -> FeatureMap:
    return AngleFeatureMap(n_qubits, entangle=n_qubits > 1)


class QCNNLayer(nn.Module):
    """Quantum convolutional layer: shared filter, then pooling.

    The filter is tied across every pair it slides over, so an 8-qubit QCNN carries
    6 parameters where an untied version needs 22 — at the same gradient cost.
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        tie_weights: bool = True,
        observables: Sequence[Observable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumLayer(
            feature_map or _default_map(n_qubits),
            qcnn_ansatz(n_qubits, tie_weights=tie_weights),
            observables or [Z(n_qubits - 1)],  # the surviving wire after pooling
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum(x)

    @property
    def n_weights(self) -> int:
        return int(self.quantum.theta.numel())

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}, n_weights={self.n_weights}"


class MPSLayer(nn.Module):
    """Matrix-product-state layer — a staircase of two-qubit blocks."""

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap | None = None,
        observables: Sequence[Observable] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.quantum = QuantumLayer(
            feature_map or _default_map(n_qubits),
            mps_ansatz(n_qubits),
            observables or [Z(n_qubits - 1)],  # readout on the last wire of the chain
            **kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quantum(x)

    def extra_repr(self) -> str:
        return f"n_qubits={self.n_qubits}"


class QLSTMCell(nn.Module):
    """One LSTM cell with its four gates replaced by small circuits.

    ``forget``, ``input``, ``candidate`` and ``output`` each become a
    :class:`QuantumLayer`; the recurrence, the sigmoids and the tanh stay classical.
    A classical projection maps ``[x, h]`` down to the qubit count first, which is
    what keeps the circuits small enough to be worth running.
    """

    GATES = ("forget", "input", "candidate", "output")

    def __init__(
        self,
        n_inputs: int,
        hidden_size: int,
        n_qubits: int = 4,
        ansatz: Ansatz | None = None,
        n_layers: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from qmlkit.ansatz.library import hardware_efficient

        self.n_inputs = n_inputs
        self.hidden_size = hidden_size
        self.n_qubits = n_qubits
        self.project = nn.Linear(n_inputs + hidden_size, n_qubits)
        self.gates = nn.ModuleDict(
            {
                name: QuantumLayer(
                    _default_map(n_qubits),
                    ansatz or hardware_efficient(n_qubits, n_layers),
                    [Z(i) for i in range(n_qubits)],
                    init_seed=i,
                    **kwargs,
                )
                for i, name in enumerate(self.GATES)
            }
        )
        self.readout = nn.ModuleDict(
            {name: nn.Linear(n_qubits, hidden_size) for name in self.GATES}
        )

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        if state is None:
            h = x.new_zeros((batch, self.hidden_size))
            c = x.new_zeros((batch, self.hidden_size))
        else:
            h, c = state

        v = torch.tanh(self.project(torch.cat([x, h], dim=-1)))
        g = {name: self.readout[name](self.gates[name](v)) for name in self.GATES}

        f = torch.sigmoid(g["forget"])
        i = torch.sigmoid(g["input"])
        o = torch.sigmoid(g["output"])
        c = f * c + i * torch.tanh(g["candidate"])
        h = o * torch.tanh(c)
        return h, c

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, hidden_size={self.hidden_size}, "
            f"n_qubits={self.n_qubits}, gates=4"
        )


class QLSTM(nn.Module):
    """A QLSTM over a sequence. Returns ``(outputs, (h, c))``."""

    def __init__(self, n_inputs: int, hidden_size: int, n_qubits: int = 4, **kwargs: Any) -> None:
        super().__init__()
        self.cell = QLSTMCell(n_inputs, hidden_size, n_qubits, **kwargs)
        self.hidden_size = hidden_size

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if x.dim() != 3:
            raise ValueError(f"expected (batch, time, features), got shape {tuple(x.shape)}")
        outputs = []
        h_c = state
        for t in range(x.shape[1]):
            h, c = self.cell(x[:, t, :], h_c)
            h_c = (h, c)
            outputs.append(h)
        assert h_c is not None
        return torch.stack(outputs, dim=1), h_c


class DressedQuantumNet(nn.Module):
    """The dressed circuit: a frozen backbone, then ``Linear -> quantum -> Linear``.

    Transfer learning with a quantum head. The backbone is frozen, so only the
    dressed block trains — and because the layer returns input gradients, the
    ``Linear`` that feeds the circuit trains too. The lecture's version returns
    ``None`` there, which silently freezes exactly that layer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        in_features: int,
        n_qubits: int,
        n_outputs: int,
        n_layers: int = 2,
        feature_map: FeatureMap | None = None,
        ansatz: Ansatz | None = None,
        freeze_backbone: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        from qmlkit.ansatz.library import hardware_efficient

        self.backbone = backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)
        self.pre = nn.Linear(in_features, n_qubits)
        self.quantum = QuantumLayer(
            feature_map or _default_map(n_qubits),
            ansatz or hardware_efficient(n_qubits, n_layers),
            [Z(i) for i in range(n_qubits)],
            **kwargs,
        )
        self.post = nn.Linear(n_qubits, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.post(self.quantum(torch.tanh(self.pre(self.backbone(x)))))

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def frozen_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
