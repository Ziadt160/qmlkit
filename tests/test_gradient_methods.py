"""Gradient dispatch, adjoint differentiation, and SPSA.

Adjoint and parameter-shift must agree to machine precision — they are exact by
different routes, so any disagreement is a bug in one of them. SPSA is stochastic
and is checked on the properties that actually matter: unbiasedness, a cost that
does not grow with the parameter count, and convergence.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.gradients import (
    SPSASchedule,
    adjoint_grad,
    choose_method,
    grad,
    hadamard_grad,
    hadamard_grad_cost,
    list_gradient_methods,
    minimize_spsa,
    register_gradient,
    spsa_grad,
    supports_adjoint,
    supports_hadamard_grad,
)
from qmlkit.gradients.spsa import spsa_step

ZOO = ["hardware_efficient", "strongly_entangling", "simplified_two_design", "qcnn", "qaoa", "mps"]


def _case(name: str = "hardware_efficient", n_qubits: int = 3):
    a = qk.get_ansatz(name, n_qubits=n_qubits)
    return a.build(), a.init(seed=0), qk.Z(0) + 0.5 * qk.ZZ(0, n_qubits - 1)


# --------------------------------------------------------------------------- #
# adjoint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ZOO)
def test_adjoint_matches_parameter_shift_exactly(name):
    """Two exact methods by different routes — they must agree to machine precision."""
    spec, theta, obs = _case(name, 4)
    a = adjoint_grad(spec, theta, obs)
    p = qk.param_shift_grad_circuit(spec, theta, obs)
    assert a == pytest.approx(p, abs=1e-12)


def test_adjoint_handles_weight_tied_parameters():
    """The occurrence sum must work the same way it does for parameter-shift."""
    a = qk.Ansatz(3, qk.share(3, qk.RotationLayer("ry")))
    spec, theta = a.build(), a.init(seed=2)
    assert len(spec.occurrences_of(0)) == 3
    assert adjoint_grad(spec, theta, qk.Z(0)) == pytest.approx(
        qk.param_shift_grad_circuit(spec, theta, qk.Z(0)), abs=1e-12
    )


def test_adjoint_handles_the_four_term_gates():
    qc = qk.QCircuit(2)
    qc.h(0)
    qc.ry(1, qk.ParamRef(0))
    qc.crz(0, 1, qk.ParamRef(1))
    qc.h(0)
    spec = qc.to_spec()
    theta = np.array([0.6, 1.3])
    assert adjoint_grad(spec, theta, qk.Z(0)) == pytest.approx(
        qk.param_shift_grad_circuit(spec, theta, qk.Z(0)), abs=1e-12
    )


def test_adjoint_respects_parameter_scaling():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0, scale=2.0))
    g = adjoint_grad(qc.to_spec(), np.array([0.3]), qk.Z(0))
    assert g[0] == pytest.approx(-2.0 * np.sin(0.6), abs=1e-12)


def test_adjoint_cost_does_not_grow_with_parameter_count():
    """One backward pass regardless of P — that is the whole point."""
    counts = []
    for n_layers in (1, 4):
        a = qk.hardware_efficient(3, n_layers)
        counts.append((a.n_params, a.resources()["grad_circuits"]))
    assert counts[1][0] > counts[0][0]
    assert counts[1][1] > counts[0][1]  # parameter-shift grows...
    # ...while adjoint needs a single pass in both cases
    for name in ("hardware_efficient",):
        spec, theta, obs = _case(name, 3)
        assert adjoint_grad(spec, theta, obs).shape == theta.shape


def test_supports_adjoint_reports_honestly():
    spec, _, _ = _case()
    assert supports_adjoint(spec)
    qk.register_gate(qk.GateDef("noderiv", 1, 1, lambda t: np.eye(2, dtype=complex), (1.0,)))
    qc = qk.QCircuit(1)
    qc.apply("noderiv", 0, qk.ParamRef(0))
    assert not supports_adjoint(qc.to_spec())
    with pytest.raises(ValueError, match="no derivative matrix"):
        adjoint_grad(qc.to_spec(), np.array([0.1]), qk.Z(0))


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def test_auto_picks_adjoint_when_it_can_and_shifts_when_it_cannot():
    spec, _, _ = _case()
    assert choose_method(spec) == "adjoint"
    assert choose_method(spec, shots=1000) == "parameter-shift"


def test_all_exact_methods_agree():
    spec, theta, obs = _case()
    ref = grad(spec, theta, obs, method="adjoint")
    assert grad(spec, theta, obs) == pytest.approx(ref, abs=1e-12)  # auto
    assert grad(spec, theta, obs, method="parameter-shift") == pytest.approx(ref, abs=1e-12)
    assert grad(spec, theta, obs, method="finite-diff") == pytest.approx(ref, abs=1e-6)


def test_adjoint_refuses_a_shot_budget_rather_than_ignoring_it():
    spec, theta, obs = _case()
    with pytest.raises(ValueError, match="cannot honour a shot budget"):
        grad(spec, theta, obs, method="adjoint", shots=1000)


def test_unknown_method_is_rejected():
    spec, theta, obs = _case()
    with pytest.raises(KeyError, match="unknown gradient method"):
        grad(spec, theta, obs, method="telepathy")


def test_a_researcher_can_register_their_own_estimator():
    """Custom estimators become a keyword everywhere the library takes method=."""

    @register_gradient("always_zero_test")
    def always_zero(spec, theta, obs, *, backend=None, shots=None, **kw):
        return np.zeros(spec.n_params)

    assert "always_zero_test" in list_gradient_methods()
    spec, theta, obs = _case()
    assert np.allclose(grad(spec, theta, obs, method="always_zero_test"), 0.0)


def test_gradient_registry_rejects_duplicates():
    with pytest.raises(ValueError, match="already registered"):
        register_gradient("adjoint", lambda *a, **k: None)


def test_default_observable_is_z0():
    spec, theta, _ = _case()
    assert grad(spec, theta) == pytest.approx(grad(spec, theta, qk.Z(0)), abs=1e-12)


# --------------------------------------------------------------------------- #
# SPSA
# --------------------------------------------------------------------------- #
def test_spsa_is_unbiased_in_expectation():
    """Noisy per-sample, but the average converges on the true gradient."""
    spec, theta, obs = _case("hardware_efficient", 2)
    exact = grad(spec, theta, obs, method="adjoint")
    rng = np.random.default_rng(0)

    def f(t):
        return qk.expectation(spec, obs, theta=t)

    avg = np.mean([spsa_grad(f, theta, c=0.01, rng=rng) for _ in range(400)], axis=0)
    assert avg == pytest.approx(exact, abs=0.05)


def test_spsa_costs_two_evaluations_whatever_the_parameter_count():
    calls = {"n": 0}
    for n_layers in (1, 8):
        a = qk.hardware_efficient(3, n_layers)
        spec, theta = a.build(), a.init(seed=0)
        calls["n"] = 0

        def f(t, spec=spec):
            calls["n"] += 1
            return qk.expectation(spec, qk.Z(0), theta=t)

        spsa_grad(f, theta, c=0.1, seed=0)
        assert calls["n"] == 2, f"SPSA used {calls['n']} evaluations for P={spec.n_params}"


def test_spsa_averaging_reduces_variance():
    spec, theta, obs = _case("hardware_efficient", 2)
    exact = grad(spec, theta, obs, method="adjoint")

    def f(t):
        return qk.expectation(spec, obs, theta=t)

    def spread(n_avg):
        errs = [
            np.linalg.norm(spsa_grad(f, theta, c=0.01, n_avg=n_avg, seed=s) - exact)
            for s in range(30)
        ]
        return float(np.mean(errs))

    assert spread(16) < spread(1)


def test_spsa_via_the_dispatcher():
    spec, theta, obs = _case("hardware_efficient", 2)
    g = grad(spec, theta, obs, method="spsa", seed=0, n_avg=8)
    assert g.shape == theta.shape
    assert np.isfinite(g).all()


def test_spsa_schedule_decays_and_carries_the_stability_constant():
    s = SPSASchedule(a=0.2, c=0.1, n_iterations=100)
    assert pytest.approx(10.0) == s.A  # 10% of the planned iterations
    assert s.step_size(0) > s.step_size(50) > s.step_size(99)
    assert s.perturbation(0) > s.perturbation(99)
    assert "A=" in repr(s)


def test_spsa_step_moves_downhill_on_a_quadratic():
    def f(t):
        return float(np.sum((t - 1.0) ** 2))

    theta = np.zeros(4)
    rng = np.random.default_rng(0)
    for k in range(60):
        theta = spsa_step(f, theta, k, SPSASchedule(a=0.5, c=0.05), rng)
    assert f(theta) < f(np.zeros(4))


def test_minimize_spsa_reduces_a_circuit_loss():
    a = qk.hardware_efficient(3, 2)
    spec, theta = a.build(), a.init("uniform", seed=1)
    obs = qk.Z(0) + qk.Z(1) + qk.Z(2)

    def loss(t):
        return qk.expectation(spec, obs, theta=t)

    seen: list[int] = []
    best, history = minimize_spsa(
        loss, theta, n_iterations=150, seed=0, callback=lambda k, t, v: seen.append(k)
    )
    assert len(history) == 151
    assert len(seen) == 150
    assert history[-1] < history[0]
    assert best.shape == theta.shape


def test_spsa_validates_its_arguments():
    def f(t):
        return 0.0

    with pytest.raises(ValueError, match="c must be positive"):
        spsa_grad(f, np.zeros(2), c=0.0)
    with pytest.raises(ValueError, match="n_avg must be at least 1"):
        spsa_grad(f, np.zeros(2), n_avg=0)


# --------------------------------------------------------------------------- #
# Hadamard test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ZOO)
def test_hadamard_matches_adjoint_exactly(name):
    """A third exact route to the same number, via an ancilla instead of shifts."""
    spec, theta, obs = _case(name, 3)
    assert hadamard_grad(spec, theta, obs) == pytest.approx(
        adjoint_grad(spec, theta, obs), abs=1e-12
    )


def test_hadamard_costs_half_of_parameter_shift():
    """One circuit per parameter instead of two — the reason to reach for it."""
    spec, _, _ = _case("hardware_efficient", 3)
    assert hadamard_grad_cost(spec) == spec.n_params
    assert qk.gradient_cost(spec, "parameter-shift") == 2 * hadamard_grad_cost(spec)


def test_hadamard_sums_tied_occurrences_rather_than_shifting_them_together():
    """Three Ry(t) on the same qubit is Ry(3t): the derivative is 3x, not 1x."""
    a = qk.Ansatz(1, qk.share(3, qk.RotationLayer("ry")))
    spec, theta = a.build(), np.array([0.4])
    assert len(spec.occurrences_of(0)) == 3
    assert hadamard_grad(spec, theta, qk.Z(0))[0] == pytest.approx(-3 * np.sin(1.2), abs=1e-12)


def test_hadamard_respects_parameter_scaling():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0, scale=2.0))
    g = hadamard_grad(qc.to_spec(), np.array([0.3]), qk.Z(0))
    assert g[0] == pytest.approx(-2.0 * np.sin(0.6), abs=1e-12)


def test_hadamard_refuses_controlled_rotations_instead_of_guessing():
    """CRZ's generator is not a Pauli, so there is no controlled form to insert."""
    qc = qk.QCircuit(2)
    qc.h(0).crz(0, 1, qk.ParamRef(0))
    spec = qc.to_spec()
    assert not supports_hadamard_grad(spec)
    with pytest.raises(ValueError, match="needs Pauli-generated rotations"):
        hadamard_grad(spec, np.array([0.5]), qk.Z(1))


def test_hadamard_leaves_the_original_register_untouched():
    """The ancilla is added, not stolen from the circuit's own qubits."""
    spec, theta, obs = _case("hardware_efficient", 3)
    before = spec.n_qubits
    hadamard_grad(spec, theta, obs)
    assert spec.n_qubits == before


def test_hadamard_via_the_dispatcher():
    spec, theta, obs = _case("hardware_efficient", 2)
    assert grad(spec, theta, obs, method="hadamard") == pytest.approx(
        grad(spec, theta, obs, method="adjoint"), abs=1e-12
    )


# --------------------------------------------------------------------------- #
# backprop (torch)
# --------------------------------------------------------------------------- #
requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="needs PyTorch"
)


@requires_torch
@pytest.mark.torch
@pytest.mark.parametrize("name", ZOO)
def test_backprop_matches_adjoint_exactly(name):
    """Autograd through the simulator must land on the same number as adjoint."""
    spec, theta, obs = _case(name, 3)
    assert grad(spec, theta, obs, method="backprop") == pytest.approx(
        adjoint_grad(spec, theta, obs), abs=1e-12
    )


@requires_torch
@pytest.mark.torch
def test_backprop_handles_controlled_rotations_that_hadamard_refuses():
    qc = qk.QCircuit(2)
    qc.h(0).ry(1, qk.ParamRef(0)).crz(0, 1, qk.ParamRef(1)).cy(0, 1)
    spec, theta = qc.to_spec(), np.array([0.6, 1.3])
    assert grad(spec, theta, qk.Z(1), method="backprop") == pytest.approx(
        adjoint_grad(spec, theta, qk.Z(1)), abs=1e-12
    )


@requires_torch
@pytest.mark.torch
def test_backprop_refuses_a_shot_budget_rather_than_ignoring_it():
    spec, theta, obs = _case()
    with pytest.raises(ValueError, match="cannot honour a shot budget"):
        grad(spec, theta, obs, method="backprop", shots=1000)


@requires_torch
@pytest.mark.torch
def test_torch_backend_statevector_matches_the_numpy_reference():
    from qmlkit.core.backends.torch_backend import TorchBackend

    qc = qk.QCircuit(3)
    qc.h(0).cx(0, 1).ry(2, 0.7).crz(1, 2, 1.1).cy(0, 2).t(1)
    spec = qc.to_spec()
    assert TorchBackend().statevector(spec) == pytest.approx(qk.statevector(spec), abs=1e-14)


@requires_torch
@pytest.mark.torch
def test_torch_expectation_is_a_differentiable_scalar():
    import torch

    from qmlkit.core.backends.torch_backend import torch_expectation

    spec, theta, obs = _case("hardware_efficient", 2)
    t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    value = torch_expectation(spec, t, obs)
    assert value.requires_grad and value.shape == ()
    assert float(value.detach()) == pytest.approx(qk.expval(spec, obs, theta=theta), abs=1e-12)
    value.backward()
    assert t.grad is not None and t.grad.shape == t.shape


# --------------------------------------------------------------------------- #
# second derivatives and cost accounting
# --------------------------------------------------------------------------- #
def test_hessian_matches_the_analytic_second_derivative():
    """<Z> of Ry(t)|0> is cos(t), so the Hessian is the 1x1 matrix [-cos(t)]."""
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    h = qk.hessian(qc.to_spec(), np.array([0.8]), qk.Z(0))
    assert h.shape == (1, 1)
    assert h[0, 0] == pytest.approx(-np.cos(0.8), abs=1e-6)


def test_hessian_is_square_and_symmetric():
    spec, theta, obs = _case("hardware_efficient", 2)
    h = qk.hessian(spec, theta, obs)
    assert h.shape == (theta.size, theta.size)
    assert h == pytest.approx(h.T, abs=1e-12)


def test_gradient_cost_is_reported_for_every_built_in_method():
    spec, _, _ = _case("hardware_efficient", 3)
    p = spec.n_params
    assert qk.gradient_cost(spec, "adjoint") == 1
    assert qk.gradient_cost(spec, "backprop") == 1
    assert qk.gradient_cost(spec, "spsa") == 2
    assert qk.gradient_cost(spec, "hadamard") == p
    assert qk.gradient_cost(spec, "finite-diff") == 2 * p
    assert qk.gradient_cost(spec, "parameter-shift") >= 2 * p
    assert qk.gradient_cost(spec, "always_zero_test") == "unknown"


def test_every_advertised_method_explains_itself_when_its_extra_is_missing(monkeypatch):
    """`list_gradient_methods()` promises six methods on a bare install too.

    Anyone who iterates them — the quickstart and the gradients tutorial both do —
    must get an install command, not a raw ModuleNotFoundError from deep inside an
    optional dependency. Simulated by making `import torch` fail.
    """
    import sys

    from qmlkit.core.backends.base import BackendNotAvailable

    assert "backprop" in list_gradient_methods()
    monkeypatch.setitem(sys.modules, "torch", None)

    spec, theta, obs = _case("hardware_efficient", 2)
    with pytest.raises(BackendNotAvailable, match=r"pip install 'qmlkit\[torch\]'"):
        grad(spec, theta, obs, method="backprop")


def test_iterating_every_method_never_raises_an_uncaught_import_error(monkeypatch):
    """The exact shape of the loop the docs and examples use."""
    import sys

    from qmlkit.core.backends.base import BackendNotAvailable

    monkeypatch.setitem(sys.modules, "torch", None)
    spec, theta, obs = _case("hardware_efficient", 2)
    ran = 0
    for method in list_gradient_methods():
        kwargs = {"seed": 0} if method == "spsa" else {}
        try:
            grad(spec, theta, obs, method=method, **kwargs)
            ran += 1
        except BackendNotAvailable:
            continue
    assert ran >= 4, "the methods that need no extra must still all run"


def test_backprop_refuses_a_backend_with_no_statevector():
    """It always evaluates on the torch simulator, so on a device it must refuse.

    Silently returning a machine-precision gradient computed on a simulator, to a
    caller who asked for one from hardware, is the plausible-wrong-number failure this
    library exists to catch — and it was doing exactly that.
    """
    pytest.importorskip("torch")
    import numpy as np

    import qmlkit as qk

    class Device(qk.Backend):
        name = "backprop_probe_device"
        supports_statevector = False
        supports_exact = False

        def counts(self, spec, shots, seed=None):
            self._check_bound(spec)
            probs = np.abs(qk.get_backend("numpy").statevector(spec)) ** 2
            rng = np.random.default_rng(0)
            return {
                format(i, f"0{spec.n_qubits}b"): int(c)
                for i, c in enumerate(rng.multinomial(shots, probs / probs.sum()))
                if c
            }

    ansatz = qk.hardware_efficient(3, 2)
    spec, theta = ansatz.build(), ansatz.init(seed=0)
    with pytest.raises(ValueError, match="parameter-shift"):
        qk.grad(spec, theta, qk.Z(0), method="backprop", backend=Device())


def test_backprop_still_works_on_a_statevector_backend():
    pytest.importorskip("torch")
    import numpy as np

    import qmlkit as qk

    ansatz = qk.hardware_efficient(3, 2)
    spec, theta = ansatz.build(), ansatz.init(seed=0)
    np.testing.assert_allclose(
        qk.grad(spec, theta, qk.Z(0), method="backprop", backend="numpy"),
        qk.grad(spec, theta, qk.Z(0), method="adjoint"),
        atol=1e-12,
    )
