r"""Optimisers that only make sense for quantum circuits.

Adam and SGD come from torch. These three do not exist there, because they exploit
structure a general optimiser cannot see:

* **Rotosolve** — a circuit expectation is a *sinusoid* in any single Pauli-rotation
  angle. Three evaluations pin that sinusoid down exactly, so you can jump straight
  to its minimum instead of stepping toward it. No learning rate, no tuning.
* **Quantum natural gradient** — parameter space is curved. Following the
  Fubini–Study geometry rather than the Euclidean one usually converges in far fewer
  steps.
* **SPSA** — lives in :mod:`qmlkit.gradients.spsa`; two evaluations per step at any
  parameter count.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.execute import BackendLike, expval
from qmlkit.core.ir import CircuitSpec
from qmlkit.core.observables import Observable, Z

__all__ = [
    "supports_rotosolve",
    "rotosolve_step",
    "minimize_rotosolve",
    "metric_tensor",
    "quantum_fisher_information",
    "qng_step",
    "minimize_qng",
    "shots_for_precision",
]

LossFn = Callable[[npt.NDArray[Any]], float]


# --------------------------------------------------------------------------- #
# Rotosolve
# --------------------------------------------------------------------------- #
def _optimal_angle(f: LossFn, theta: npt.NDArray[Any], k: int) -> float:
    r"""Closed-form minimiser of ``f`` in coordinate ``k``.

    Along one Pauli-rotation angle the loss is
    :math:`A\sin(\theta + B) + C`, which three samples determine completely. The
    minimum then follows from an arctangent — exactly, in one jump.
    """
    probe = theta.copy()
    probe[k] = 0.0
    m0 = f(probe)
    probe[k] = np.pi / 2
    mp = f(probe)
    probe[k] = -np.pi / 2
    mm = f(probe)
    return float(-np.pi / 2 - np.arctan2(2.0 * m0 - mp - mm, mp - mm))


def supports_rotosolve(spec: CircuitSpec) -> bool:
    r"""Whether Rotosolve's closed form is actually valid for this circuit.

    Rotosolve works because a circuit expectation is a *single* sinusoid
    :math:`A\sin(	heta + B) + C` in any one Pauli-rotation angle — three samples then
    determine it exactly. That holds when a parameter drives one rotation, or several
    that compose into one (same qubit, same generator).

    It does **not** hold when one parameter is shared across gates that do not compose
    — QAOA's cost angle drives one ``rz`` per graph edge, so ``E(gamma)`` carries one
    frequency per edge. Rotosolve then solves for the wrong minimum, converges
    immediately, and reports a number that looks like a result. Measured on a 5-edge
    MaxCut: frequencies 1 through 4 are all present, and Rotosolve sticks at the
    uniform-state energy no matter how many sweeps it is given.
    """
    from collections import defaultdict

    by_param: dict[int, list[tuple[str, tuple[int, ...]]]] = defaultdict(list)
    for slot in spec.slots():
        op = spec.ops[slot.op_index]
        by_param[slot.ref.index].append((slot.gate, op.qubits))
    # several occurrences are fine only if they are the same rotation on the same wire
    return all(len(set(sites)) <= 1 for sites in by_param.values())


def rotosolve_step(
    f: LossFn, theta: Sequence[float], indices: Sequence[int] | None = None
) -> npt.NDArray[Any]:
    """One sweep: set every coordinate to its exact optimum, in turn.

    Three evaluations per parameter, and each one lands on that coordinate's minimum
    rather than moving toward it.
    """
    arr = np.asarray(theta, dtype=float).ravel().copy()
    for k in indices if indices is not None else range(arr.size):
        arr[k] = _optimal_angle(f, arr, k)
    return arr


def minimize_rotosolve(
    f: LossFn,
    theta0: Sequence[float],
    n_sweeps: int = 20,
    tol: float = 1e-9,
    callback: Callable[[int, npt.NDArray[Any], float], None] | None = None,
) -> tuple[npt.NDArray[Any], list[float]]:
    """Minimise by repeated Rotosolve sweeps. No learning rate to choose.

    **Precondition.** ``f`` must be a single sinusoid in each angle — true for a plain
    expectation value ``<O>`` of a circuit where each parameter drives one Pauli
    rotation. It is *not* true when an angle is shared across gates that do not
    compose (QAOA's cost angle drives one ``rz`` per edge), nor when the loss is
    non-linear in the state (purity is ``Tr(rho^2)``, so it carries double
    frequencies). In those cases Rotosolve converges immediately on the wrong point
    and reports it as a result. :func:`supports_rotosolve` checks the first case;
    the second is a property of your loss, not of the circuit.
    """
    theta = np.asarray(theta0, dtype=float).ravel().copy()
    history = [float(f(theta))]
    for sweep in range(n_sweeps):
        theta = rotosolve_step(f, theta)
        value = float(f(theta))
        history.append(value)
        if callback is not None:
            callback(sweep, theta, value)
        if abs(history[-2] - value) < tol:
            break
    return theta, history


# --------------------------------------------------------------------------- #
# quantum natural gradient
# --------------------------------------------------------------------------- #
def _exact_derivative_states(
    spec: CircuitSpec, theta: npt.NDArray[Any], backend: BackendLike = None
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    r""":math:`|\psi\rangle` and every :math:`|\partial_k\psi\rangle`, exactly.

    One forward sweep. Each already-open derivative state is carried through the next
    gate, and a parameterised gate opens a new one by applying its closed-form
    :math:`dU/d\theta` to the current state. That costs the same memory as
    differencing the circuit ``2P`` times and needs no step size at all.
    """
    from qmlkit.core.backends.numpy_backend import _apply
    from qmlkit.core.gates import gate_derivative, gate_matrix
    from qmlkit.core.ir import ParamRef

    n = spec.n_qubits
    shape = (2,) * n
    slots = spec.slots()
    slot_angles = spec.bind_slots(theta)
    slot_of_op = {s.op_index: i for i, s in enumerate(slots)}

    psi = np.zeros(shape, dtype=complex)
    psi[(0,) * n] = 1.0
    derivatives = np.zeros((spec.n_params, *shape), dtype=complex)
    opened: list[int] = []  # logical parameters whose derivative state is live

    for op_index, op in enumerate(spec.ops):
        slot_i = slot_of_op.get(op_index)
        if slot_i is None:
            angles = tuple(float(p) for p in op.params if not isinstance(p, ParamRef))
        else:
            angles = (float(slot_angles[slot_i]),)
        u = gate_matrix(op.gate, angles)

        fresh = None
        if slot_i is not None:
            ref = slots[slot_i].ref
            du = gate_derivative(op.gate, angles)
            fresh = (ref.index, _apply(psi, du, op.qubits) * ref.scale)

        for k in opened:
            derivatives[k] = _apply(derivatives[k], u, op.qubits)
        if fresh is not None:
            index, state = fresh
            # `+=` so a weight-tied parameter accumulates over its occurrences
            derivatives[index] += state
            if index not in opened:
                opened.append(index)
        psi = _apply(psi, u, op.qubits)

    return psi.reshape(-1), derivatives.reshape(spec.n_params, -1)


def metric_tensor(
    spec: CircuitSpec,
    theta: Sequence[float],
    approx: str = "block-diag",
    backend: BackendLike = None,
    eps: float = 1e-4,
) -> npt.NDArray[Any]:
    r"""Fubini–Study metric — the curvature of parameter space.

    ``approx="diag"`` keeps only the diagonal (cheapest). ``"block-diag"`` and
    ``None`` compute the full tensor from state overlaps; on a simulator that is
    affordable and exact, so they currently coincide. Note that PennyLane's
    ``approx="block-diag"`` means something narrower — it blocks by circuit layer and
    zeroes every cross-layer entry — so the same keyword does not port between the
    two libraries. qmlkit follows the true geometry; PennyLane follows an
    approximation to it.

    .. math::  g_{ij} = \mathrm{Re}\langle \partial_i\psi | \partial_j\psi \rangle
               - \langle \partial_i\psi|\psi\rangle\langle\psi|\partial_j\psi\rangle

    The derivative states are exact whenever every parameterised gate declares a
    closed-form derivative, which every built-in gate does. ``eps`` is only consulted
    on the fallback path, for a custom gate registered without a ``dmatrix``.
    """
    from qmlkit.core.execute import statevector
    from qmlkit.gradients.adjoint import supports_adjoint

    arr = np.asarray(theta, dtype=float).ravel()
    p = arr.size

    if supports_adjoint(spec, backend):
        psi, derivatives = _exact_derivative_states(spec, arr, backend)
    else:
        psi = statevector(spec.bind(arr), backend=backend)
        derivatives = np.empty((p, psi.size), dtype=complex)
        for k in range(p):
            plus, minus = arr.copy(), arr.copy()
            plus[k] += eps
            minus[k] -= eps
            derivatives[k] = (
                statevector(spec.bind(plus), backend=backend)
                - statevector(spec.bind(minus), backend=backend)
            ) / (2 * eps)

    overlaps = derivatives @ psi.conj()
    g = np.real(derivatives.conj() @ derivatives.T) - np.real(np.outer(overlaps.conj(), overlaps))
    if approx == "diag":
        return np.diag(np.diag(g))
    if approx in ("block-diag", None):
        return g
    raise ValueError(f"unknown approx {approx!r}; expected 'diag', 'block-diag' or None")


def quantum_fisher_information(
    spec: CircuitSpec, theta: Sequence[float], backend: BackendLike = None
) -> npt.NDArray[Any]:
    """QFIM — exactly ``4 x`` the Fubini–Study metric.

    Distinct from the *classical* Fisher information in :mod:`qmlkit.metrics`, which
    describes the output distribution and is what effective dimension uses.
    """
    return 4.0 * metric_tensor(spec, theta, None, backend)


def qng_step(
    spec: CircuitSpec,
    theta: Sequence[float],
    obs: Observable | None = None,
    lr: float = 0.1,
    approx: str = "block-diag",
    regularization: float = 1e-6,
    backend: BackendLike = None,
) -> npt.NDArray[Any]:
    r"""One natural-gradient step: ``theta <- theta - lr * g^+ grad``.

    The pseudo-inverse of the metric rescales each direction by how much the *state*
    actually moves, rather than how much the parameter does.
    """
    from qmlkit.gradients.dispatch import grad

    obs = Z(0) if obs is None else obs
    arr = np.asarray(theta, dtype=float).ravel()
    g = metric_tensor(spec, arr, approx, backend)
    gradient = grad(spec, arr, obs, backend=backend)
    natural = np.linalg.pinv(g + regularization * np.eye(g.shape[0])) @ gradient
    return arr - lr * natural


def minimize_qng(
    spec: CircuitSpec,
    theta0: Sequence[float],
    obs: Observable | None = None,
    n_steps: int = 50,
    lr: float = 0.1,
    approx: str = "block-diag",
    backend: BackendLike = None,
    callback: Callable[[int, npt.NDArray[Any], float], None] | None = None,
) -> tuple[npt.NDArray[Any], list[float]]:
    """Minimise ``<obs>`` by quantum natural gradient descent."""
    obs = Z(0) if obs is None else obs
    theta = np.asarray(theta0, dtype=float).ravel().copy()
    history: list[float] = []
    for step in range(n_steps):
        value = expval(spec, obs, theta=theta, backend=backend)
        history.append(value)
        if callback is not None:
            callback(step, theta, value)
        theta = qng_step(spec, theta, obs, lr, approx, backend=backend)
    history.append(expval(spec, obs, theta=theta, backend=backend))
    return theta, history


# re-exported so the shot-budget helper is where an optimiser user looks for it
from qmlkit.utils.shots import shots_for_precision  # noqa: E402
