"""Ready-made models — the two-line path.

    model = qk.VQC(n_features=4, n_classes=2)
    model.fit(X, y)
    model.score(X, y)

Every default here is a choice you can override, and each override is one keyword.
Pass your own ``feature_map`` or ``ansatz``, and the rest still works — nothing in
these models knows anything about the specific ones they default to.

If you want the layer without the training loop, use
:class:`~qmlkit.nn.layer.QuantumLayer` directly and treat it as any other
``nn.Module``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from qmlkit.ansatz.library import Ansatz, hardware_efficient
from qmlkit.core.observables import Observable, Z
from qmlkit.encoding.feature_maps import AngleFeatureMap, FeatureMap
from qmlkit.encoding.scaling import AngleScaler
from qmlkit.nn.layer import QuantumLayer

__all__ = ["HybridModel", "VQC", "VQRegressor"]


class HybridModel(nn.Module):
    """Shared machinery: a training loop, and sensible construction defaults."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        n_qubits: int | None = None,
        n_layers: int = 2,
        feature_map: FeatureMap | None = None,
        ansatz: Ansatz | None = None,
        observables: Sequence[Observable] | None = None,
        shots: int | None = None,
        backend: Any = None,
        grad_method: str = "auto",
        scale_inputs: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        n_qubits = n_qubits or (feature_map.n_qubits if feature_map else n_features)

        self.feature_map = feature_map or AngleFeatureMap(n_qubits, entangle=n_qubits > 1)
        self.ansatz = ansatz or hardware_efficient(n_qubits, n_layers)
        obs = list(observables) if observables is not None else [Z(i) for i in range(n_qubits)]

        # a classical projection only when the widths genuinely differ
        self.pre: nn.Module = (
            nn.Sequential(nn.Linear(n_features, n_qubits), nn.Tanh())
            if n_features != n_qubits
            else nn.Identity()
        )
        self.quantum = QuantumLayer(
            self.feature_map,
            self.ansatz,
            obs,
            shots=shots,
            backend=backend,
            grad_method=grad_method,
            init_seed=seed,
        )
        self.head = nn.Linear(len(obs), n_outputs)
        self.scaler = AngleScaler() if scale_inputs else None
        self.history_: list[float] = []

    # ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.quantum(self.pre(x)))

    def _prepare(self, X: npt.NDArray[Any], fit_scaler: bool = False) -> torch.Tensor:
        arr = np.atleast_2d(np.asarray(X, dtype=float))
        if self.scaler is not None and self.pre.__class__ is nn.Identity:
            # only meaningful when features feed rotations directly
            arr = self.scaler.fit_transform(arr) if fit_scaler else self.scaler.transform(arr)
        return torch.as_tensor(arr, dtype=torch.get_default_dtype())

    def _loss_fn(self, y: npt.NDArray[Any]) -> nn.Module:  # pragma: no cover - overridden
        """The loss for these targets.

        Takes ``y`` because a class-weighted loss cannot be built until the label
        distribution is known, and that is a property of the training set rather
        than of the model.
        """
        raise NotImplementedError

    def _targets(self, y: npt.NDArray[Any]) -> torch.Tensor:  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(
        self,
        X: npt.NDArray[Any],
        y: npt.NDArray[Any],
        epochs: int = 30,
        lr: float = 0.05,
        batch_size: int | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        verbose: bool = False,
    ) -> HybridModel:
        """Train. Returns ``self``, so it chains."""
        xt = self._prepare(X, fit_scaler=True)
        yt = self._targets(np.asarray(y))
        opt = optimizer or torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = self._loss_fn(np.asarray(y))
        n = xt.shape[0]
        bs = batch_size or n
        self.history_ = []

        for epoch in range(epochs):
            perm = torch.randperm(n)
            total = 0.0
            for start in range(0, n, bs):
                idx = perm[start : start + bs]
                opt.zero_grad()
                loss = loss_fn(self(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(idx)
            self.history_.append(total / n)
            if verbose:
                print(f"epoch {epoch + 1:3d}/{epochs}  loss {self.history_[-1]:.5f}")
        return self

    def resources(self) -> dict[str, object]:
        """What one training step costs, so nobody discovers it an hour in."""
        out = dict(self.quantum.resources())
        out["trainable_parameters"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return out

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, n_outputs={self.n_outputs}"


class VQC(HybridModel):
    """Variational quantum classifier.

    model = VQC(n_features=4, n_classes=3).fit(X, y)
    model.score(X, y)

    ``class_weight="balanced"`` reweights the loss by class frequency, which is
    what stops a skewed training set training the circuit to a constant. The
    weights are computed from the ``y`` passed to :meth:`fit`, so they describe the
    data actually trained on rather than an assumption made at construction.
    ``focal_gamma`` additionally down-weights examples the model already gets
    right; ``0.0`` disables it, ``2.0`` is the published default.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        class_weight: str | None = None,
        focal_gamma: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_features, n_classes, **kwargs)
        self.n_classes = n_classes
        self.class_weight = class_weight
        self.focal_gamma = focal_gamma

    def _loss_fn(self, y: npt.NDArray[Any]) -> nn.Module:
        from qmlkit.nn.losses import FocalLoss, class_weight_tensor

        weight = (
            class_weight_tensor(y, self.class_weight, self.n_classes)
            if self.class_weight is not None
            else None
        )
        if self.focal_gamma:
            return FocalLoss(gamma=self.focal_gamma, weight=weight)
        return nn.CrossEntropyLoss(weight=weight)

    def _targets(self, y: npt.NDArray[Any]) -> torch.Tensor:
        return torch.as_tensor(np.asarray(y).ravel(), dtype=torch.long)

    @torch.no_grad()
    def predict_proba(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return torch.softmax(self(self._prepare(X)), dim=-1).cpu().numpy()

    @torch.no_grad()
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return self(self._prepare(X)).argmax(dim=-1).cpu().numpy()

    def score(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
        """Mean accuracy."""
        return float((self.predict(X) == np.asarray(y).ravel()).mean())


class VQRegressor(HybridModel):
    """Variational quantum regressor.

    model = VQRegressor(n_features=3).fit(X, y)
    model.predict(X)
    """

    def __init__(self, n_features: int, n_outputs: int = 1, **kwargs: Any) -> None:
        super().__init__(n_features, n_outputs, **kwargs)

    def _loss_fn(self, y: npt.NDArray[Any]) -> nn.Module:
        return nn.MSELoss()

    def _targets(self, y: npt.NDArray[Any]) -> torch.Tensor:
        arr = np.asarray(y, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.as_tensor(arr, dtype=torch.get_default_dtype())

    @torch.no_grad()
    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:
        out = self(self._prepare(X)).cpu().numpy()
        return out[:, 0] if out.shape[1] == 1 else out

    def score(self, X: npt.NDArray[Any], y: npt.NDArray[Any]) -> float:
        """R² — 1.0 is perfect, 0.0 is no better than predicting the mean."""
        pred = np.asarray(self.predict(X)).ravel()
        truth = np.asarray(y, dtype=float).ravel()
        ss_res = float(((truth - pred) ** 2).sum())
        ss_tot = float(((truth - truth.mean()) ** 2).sum())
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
