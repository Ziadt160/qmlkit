"""The PyTorch bridge.

Above this boundary everything is ordinary torch — ``Adam``, ``.backward()``,
``DataLoader``, ``nn.Sequential``. Below it, circuits and shots. Nobody has to
write an ``autograd.Function`` themselves.

**Inputs get gradients.** ``backward`` returns ``df/dx`` as well as ``df/dtheta``,
so a classical layer placed *before* the quantum one actually trains. The lecture's
version returns ``None`` there, which silently freezes any pre-net — including the
``Linear(512, 4)`` in its own transfer-learning example.

Getting ``df/dx`` through a *nonlinear* feature map takes two steps: the circuit is
differentiated with respect to its encoding angles, then the chain rule down to the
features is finished classically by the map's ``angle_jacobian``. No circuits are
spent on the classical half.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from qmlkit.ansatz.library import Ansatz
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z
from qmlkit.encoding.feature_maps import FeatureMap
from qmlkit.gradients.dispatch import choose_method, grad

__all__ = ["QuantumFunction", "QuantumLayer"]


class QuantumFunction(torch.autograd.Function):
    """Autograd boundary: forward runs circuits, backward differentiates them."""

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, theta: torch.Tensor, runner: Any) -> torch.Tensor:
        ctx.runner = runner
        ctx.save_for_backward(x, theta)
        values = runner.forward_batch(x.detach().cpu().numpy(), theta.detach().cpu().numpy())
        return torch.as_tensor(values, dtype=x.dtype, device=x.device)

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor):  # type: ignore[no-untyped-def]
        x, theta = ctx.saved_tensors
        runner = ctx.runner
        gx, gtheta = runner.backward_batch(
            x.detach().cpu().numpy(),
            theta.detach().cpu().numpy(),
            grad_out.detach().cpu().numpy(),
        )
        return (
            torch.as_tensor(gx, dtype=x.dtype, device=x.device)
            if ctx.needs_input_grad[0]
            else None,
            torch.as_tensor(gtheta, dtype=theta.dtype, device=theta.device)
            if ctx.needs_input_grad[1]
            else None,
            None,  # runner is not a tensor
        )


def _is_combined(obj: object) -> bool:
    """True for a model that interleaves its own encoding (a re-uploading ansatz)."""
    return all(hasattr(obj, a) for a in ("n_inputs", "n_weights", "angles", "angle_jacobian"))


class _Runner:
    """Numpy-side execution and differentiation for one layer configuration."""

    def __init__(
        self,
        feature_map: FeatureMap,
        ansatz: Ansatz | None,
        observables: Sequence[Observable],
        backend: Any,
        shots: int | None,
        grad_method: str,
        seed: int | None,
    ) -> None:
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.observables = list(observables)
        self.backend = backend
        self.shots = shots
        self.grad_method = grad_method
        self.seed = seed

        if _is_combined(feature_map):
            # a re-uploading model already carries its own encoding, interleaved with
            # the trainable block -- there is nothing to compose
            self.n_angles = int(feature_map.n_inputs)
            self.n_weights = int(feature_map.n_weights)
            self.spec: CircuitSpec = feature_map.build()
        else:
            self.n_angles = feature_map.n_angles
            self.n_weights = ansatz.n_params if ansatz is not None else 0
            enc = feature_map.build_parametric(offset=0)
            self.spec = (
                enc.compose(ansatz.build(), param_offset=self.n_angles)
                if ansatz is not None
                else enc
            )

    def _full(self, x_row: npt.NDArray[Any], theta: npt.NDArray[Any]) -> npt.NDArray[Any]:
        return np.concatenate([self.feature_map.angles(x_row), theta])

    def forward_batch(self, x: npt.NDArray[Any], theta: npt.NDArray[Any]) -> npt.NDArray[Any]:
        from qmlkit.core.execute import expectation

        rows = np.atleast_2d(x)
        out = np.empty((rows.shape[0], len(self.observables)), dtype=float)
        for b, row in enumerate(rows):
            params = self._full(row, theta)
            for j, obs in enumerate(self.observables):
                out[b, j] = float(
                    expectation(
                        self.spec,
                        obs,
                        theta=params,
                        shots=self.shots,
                        backend=self.backend,
                        seed=self.seed,
                    )
                )
        return out if x.ndim == 2 else out[0]

    def backward_batch(
        self, x: npt.NDArray[Any], theta: npt.NDArray[Any], grad_out: npt.NDArray[Any]
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
        rows = np.atleast_2d(x)
        upstream = np.atleast_2d(grad_out)
        gx = np.zeros_like(rows)
        gtheta = np.zeros_like(theta)

        for b, row in enumerate(rows):
            params = self._full(row, theta)
            jac = self.feature_map.angle_jacobian(row)  # (n_angles, n_features), classical
            for j, obs in enumerate(self.observables):
                g = grad(
                    self.spec,
                    params,
                    obs,
                    method=self.grad_method,
                    backend=self.backend,
                    shots=self.shots,
                )
                w = float(upstream[b, j])
                # split the circuit gradient into its encoding and weight halves
                gx[b] += w * (jac.T @ g[: self.n_angles])  # chain rule to the features
                if self.n_weights:
                    gtheta += w * g[self.n_angles :]
        return (gx if x.ndim == 2 else gx[0]), gtheta


class QuantumLayer(nn.Module):
    """A circuit as an ``nn.Module``.

    Maps ``(batch, n_features)`` to ``(batch, n_observables)``, each output an
    expectation value in ``[-1, 1]``.

    Defaults are chosen for a simulator: exact expectations and adjoint gradients.
    Pass ``shots=N`` to model a device, and ``grad_method="parameter-shift"`` to
    compute the way hardware would have to.
    """

    def __init__(
        self,
        feature_map: FeatureMap,
        ansatz: Ansatz | None = None,
        observables: Sequence[Observable] | None = None,
        shots: int | None = None,
        backend: Any = None,
        grad_method: str = "auto",
        seed: int | None = None,
        init: str = "small",
        init_seed: int | None = None,
    ) -> None:
        super().__init__()
        combined = _is_combined(feature_map)
        if combined and ansatz is not None:
            raise ValueError(
                "a re-uploading model already contains its trainable block; pass it "
                "alone, without a separate ansatz"
            )
        self.feature_map = feature_map
        self.ansatz = feature_map if combined else ansatz
        n = feature_map.n_qubits
        self.observables = (
            list(observables) if observables is not None else [Z(i) for i in range(n)]
        )

        if ansatz is not None and ansatz.n_qubits != n:
            raise ValueError(f"feature map uses {n} qubits but the ansatz uses {ansatz.n_qubits}")

        weights = self.ansatz
        self.theta = nn.Parameter(
            torch.as_tensor(weights.init(init, init_seed), dtype=torch.get_default_dtype())
            if weights is not None
            else torch.zeros(0)
        )
        self._runner = _Runner(
            feature_map, ansatz, self.observables, backend, shots, grad_method, seed
        )
        if grad_method == "auto":
            self._runner.grad_method = choose_method(self._runner.spec, backend, shots)

    # ------------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        fm = getattr(self.feature_map, "feature_map", self.feature_map)
        return int(fm.n_features)

    @property
    def n_outputs(self) -> int:
        return len(self.observables)

    def configure(self, shots: int | None = ..., grad_method: str | None = None) -> QuantumLayer:
        """Switch to device-realism mode (or back) without rebuilding the layer."""
        if shots is not ...:
            self._runner.shots = shots
        if grad_method is not None:
            self._runner.grad_method = (
                choose_method(self._runner.spec, self._runner.backend, self._runner.shots)
                if grad_method == "auto"
                else grad_method
            )
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.n_features:
            raise ValueError(f"QuantumLayer expects {self.n_features} features, got {x.shape[-1]}")
        return QuantumFunction.apply(x, self.theta, self._runner)

    def resources(self) -> dict[str, object]:
        """Circuit cost, and what a batch costs under each gradient method."""
        from qmlkit.gradients.parameter_shift import grad_circuit_cost

        spec = self._runner.spec
        out = dict(spec.resources())
        out["grad_method"] = self._runner.grad_method
        out["shots"] = self._runner.shots
        out["n_outputs"] = self.n_outputs
        out["circuits_per_sample_parameter_shift"] = 1 + grad_circuit_cost(spec)
        out["passes_per_sample_adjoint"] = 1
        return out

    def extra_repr(self) -> str:
        return (
            f"n_features={self.n_features}, n_qubits={self.feature_map.n_qubits}, "
            f"n_weights={self.theta.numel()}, n_outputs={self.n_outputs}, "
            f"grad_method={self._runner.grad_method!r}, shots={self._runner.shots}"
        )
