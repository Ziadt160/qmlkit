"""Losses that survive a skewed training set.

:mod:`qmlkit.imbalance` computes the weights; this module is where torch consumes
them. Both losses here are drop-in ``nn.Module`` replacements, so they work in any
training loop, not only the one in :class:`~qmlkit.nn.models.HybridModel`::

    import qmlkit as qk
    from qmlkit.nn.losses import FocalLoss, weighted_cross_entropy

    loss_fn = weighted_cross_entropy(y_train)        # class-weighted
    loss_fn = FocalLoss(gamma=2.0, weight=...)       # down-weights easy examples

**Which to reach for.** Class weighting is the first thing to try and usually
enough: it rescales the loss so the minority class contributes as much total
gradient as the majority one. Focal loss goes further and down-weights *easy*
examples of any class, which helps when the majority class is not merely abundant
but trivially separable — the regime where a weighted loss still spends most of
its gradient re-learning what it already knows.

On a variational circuit the distinction matters more than it does classically.
Every gradient entry costs circuits, so a loss that spends its signal on examples
the model already gets right is spending a budget measured in wall-clock hours.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from qmlkit.imbalance import class_weights

__all__ = ["class_weight_tensor", "weighted_cross_entropy", "FocalLoss"]


def class_weight_tensor(
    y: Any, scheme: str = "balanced", n_classes: int | None = None
) -> torch.Tensor:
    """Class weights as a tensor indexed by class, ready for ``CrossEntropyLoss``.

    Labels must be the integer class indices the model outputs, which is what
    :class:`~qmlkit.nn.models.VQC` trains against. A class absent from ``y`` gets
    weight 1.0 rather than being dropped, so the tensor always has ``n_classes``
    entries and the loss never indexes past its end.
    """
    table = class_weights(y, scheme)
    size = n_classes if n_classes is not None else int(max(table)) + 1
    weights = np.ones(size, dtype=float)
    for label, weight in table.items():
        index = int(label)
        if not 0 <= index < size:
            raise ValueError(
                f"label {label!r} is outside the {size} classes this model has; "
                "class weights index the output layer, so labels must be 0..n_classes-1"
            )
        weights[index] = weight
    return torch.as_tensor(weights, dtype=torch.get_default_dtype())


def weighted_cross_entropy(
    y: Any, scheme: str = "balanced", n_classes: int | None = None
) -> nn.CrossEntropyLoss:
    """``CrossEntropyLoss`` already carrying the class weights for ``y``."""
    return nn.CrossEntropyLoss(weight=class_weight_tensor(y, scheme, n_classes))


class FocalLoss(nn.Module):
    r"""``-(1 - p_t)^gamma * log p_t``, averaged over the batch.

    Lin et al. (2017). ``gamma=0`` is exactly weighted cross-entropy, so the
    parameter interpolates rather than switching behaviour; ``gamma=2`` is the
    published default and down-weights an example the model already assigns
    ``p=0.9`` by a factor of 100.

    Parameters
    ----------
    gamma:
        How hard to discount easy examples. Must be non-negative.
    weight:
        Optional per-class weights, as :func:`class_weight_tensor` returns. Focal
        loss and class weighting compose — the first addresses easy examples, the
        second abundant ones, and a badly skewed problem usually has both.
    reduction:
        ``"mean"``, ``"sum"`` or ``"none"``, matching torch's convention.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be non-negative, got {gamma}")
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean', 'sum' or 'none', got {reduction!r}")
        self.gamma = float(gamma)
        self.reduction = reduction
        # a buffer, so .to(device) and state_dict() both carry the weights.
        # register_buffer types as Tensor | Module, so keep a narrowed alias for use.
        self.register_buffer("weight", weight)
        self.class_weight: torch.Tensor | None = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # cross_entropy in 'none' mode already applies the class weight, so p_t is
        # recovered from the *unweighted* log-probability to keep the focal factor
        # a pure function of confidence.
        log_p = torch.log_softmax(logits, dim=-1)
        picked = log_p.gather(-1, target.reshape(-1, 1)).squeeze(-1)
        focal = (1.0 - picked.exp()).pow(self.gamma)
        loss = -focal * picked
        if self.class_weight is not None:
            loss = loss * self.class_weight.to(loss.dtype)[target]
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum() if self.reduction == "sum" else loss

    def extra_repr(self) -> str:
        shape = None if self.class_weight is None else tuple(self.class_weight.shape)
        weighted = "None" if shape is None else f"tensor({shape})"
        return f"gamma={self.gamma}, weight={weighted}, reduction={self.reduction!r}"
