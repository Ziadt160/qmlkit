"""The PyTorch bridge: circuits as ``nn.Module`` layers, and ready-made models."""

from qmlkit.nn.advanced import QLSTM, DressedQuantumNet, MPSLayer, QCNNLayer, QLSTMCell
from qmlkit.nn.layer import QuantumFunction, QuantumLayer
from qmlkit.nn.models import VQC, HybridModel, VQRegressor

__all__ = [
    "QuantumLayer",
    "QuantumFunction",
    "HybridModel",
    "VQC",
    "VQRegressor",
    "QCNNLayer",
    "MPSLayer",
    "QLSTMCell",
    "QLSTM",
    "DressedQuantumNet",
]
