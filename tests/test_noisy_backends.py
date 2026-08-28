"""Mixed-state backends.

The load-bearing test here is the *noiseless* one. A density-matrix simulator with
no noise model must reproduce the pure-state backends to machine precision — if it
does not, every noisy number it later produces is measuring the translation rather
than the noise. Everything else in this file rests on that.

The second theme is refusal. A noisy backend has no statevector, so the gradients
that differentiate one must decline rather than quietly evaluate a noiseless circuit
and hand back a machine-precision answer to someone who asked about a noisy one.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.noisy import NoisyBackend
from qmlkit.core.backends.registry import NOISY_BACKENDS, is_available

BACKENDS = [
    pytest.param(
        name,
        marks=[
            pytest.mark.noisy,
            pytest.mark.skipif(
                not is_available(name), reason=f"{name} needs an SDK that is not installed"
            ),
        ],
    )
    for name in NOISY_BACKENDS
]


def bound(ansatz):
    """An ansatz at a fixed, non-trivial parameter vector of the right length."""
    return ansatz.build(np.linspace(0.1, 1.4, ansatz.n_params))


#: circuits chosen to exercise the paths that differ from the pure-state backends:
#: several qubits, idle qubits, entanglement, and a non-trivial angle
CIRCUITS = [
    lambda: qk.angle_encode([0.7]),
    lambda: qk.angle_encode([0.7, 0.3, 1.1]),
    lambda: bound(qk.hardware_efficient(3, 2)),
]

#: X and Y terms are the reason the exact path cannot just read a diagonal: they force
#: the basis rotation that the sampled path also uses
OBSERVABLES = [
    lambda: qk.Z(0),
    lambda: qk.X(0),
    lambda: qk.Y(0),
    lambda: qk.Z(0) + 0.5 * qk.ZZ(0, 2) + 0.25 * qk.X(1),
]


def depolarizing(backend_name: str, p: float):
    """The same physical channel, expressed the way each SDK expects."""
    if backend_name == "cirq-density":
        import cirq

        return cirq.depolarize(p)
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(p, 1), ["rx", "ry", "rz", "u", "sx", "x", "h", "id"]
    )
    return model


# --------------------------------------------------------------------------- #
# The reference agreement everything else depends on
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("make_circuit", CIRCUITS)
@pytest.mark.parametrize("make_obs", OBSERVABLES)
def test_noiseless_matches_the_reference(backend_name, make_circuit, make_obs):
    """With no noise model, a density matrix is |psi><psi| and must agree exactly."""
    spec, obs = make_circuit(), make_obs()
    if qk.core.observables.required_qubits(obs) > spec.n_qubits:
        pytest.skip("observable is wider than the circuit")
    got = qk.expectation(spec, obs, backend=qk.get_backend(backend_name))
    expected = qk.expectation(spec, obs, backend="numpy")
    assert got == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_probabilities_match_the_reference(backend_name):
    spec = bound(qk.hardware_efficient(3, 2))
    got = qk.get_backend(backend_name).probabilities(spec)
    expected = qk.get_backend("numpy").probabilities(spec)
    assert got == pytest.approx(expected, abs=1e-10)
    assert got.sum() == pytest.approx(1.0)
    assert (got >= 0).all()


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_purity_is_one_without_noise(backend_name):
    spec = qk.angle_encode([0.7, 0.3])
    assert qk.get_backend(backend_name).purity(spec) == pytest.approx(1.0, abs=1e-10)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_statevector_is_refused(backend_name):
    backend = qk.get_backend(backend_name)
    assert isinstance(backend, NoisyBackend)
    assert backend.supports_statevector is False
    assert backend.supports_exact is True
    with pytest.raises(NotImplementedError, match="density matrix"):
        backend.statevector(qk.angle_encode([0.7]))


@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("method", ["adjoint", "backprop"])
def test_state_based_gradients_are_refused(backend_name, method):
    """The plausible-wrong-number failure: a noiseless gradient for a noisy question."""
    ansatz = qk.hardware_efficient(2, 1)
    theta = np.linspace(0.1, 0.9, ansatz.n_params)
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.02))
    with pytest.raises(ValueError, match="statevector"):
        qk.grad(ansatz.build(), theta, qk.Z(0), method=method, backend=backend)


# --------------------------------------------------------------------------- #
# What still works, because it is measurement-only
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend_name", BACKENDS)
def test_parameter_shift_works_under_noise(backend_name):
    ansatz = qk.hardware_efficient(2, 2)
    theta = np.linspace(0.1, 1.2, ansatz.n_params)
    spec = ansatz.build()
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.05))

    noisy = qk.grad(spec, theta, qk.Z(0), method="parameter-shift", backend=backend)
    exact = qk.grad(spec, theta, qk.Z(0), method="parameter-shift", backend="numpy")

    assert noisy.shape == exact.shape
    assert np.isfinite(noisy).all()
    # depolarizing noise contracts the whole state toward the identity, so every
    # gradient component shrinks while the direction survives. Both halves matter:
    # the first is why noise kills trainability, the second is why it still points
    # the right way when there is enough signal left to measure.
    assert np.linalg.norm(noisy) < np.linalg.norm(exact)
    assert np.corrcoef(noisy, exact)[0, 1] > 0.99


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_batched_gradient_routes_through_the_same_backend(backend_name):
    ansatz = qk.hardware_efficient(2, 1)
    spec = ansatz.build()
    thetas = np.stack([np.linspace(0.1, 0.9, ansatz.n_params)] * 2) * [[1.0], [0.9]]
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.03))

    batched = qk.grad_batch(spec, thetas, qk.Z(0), method="parameter-shift", backend=backend)
    for row, theta in zip(batched, thetas, strict=True):
        single = qk.grad(spec, theta, qk.Z(0), method="parameter-shift", backend=backend)
        assert row == pytest.approx(single, abs=1e-10)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_exact_batched_expectation_matches_the_loop(backend_name):
    """``expectation_over`` has no state to stack, so it must fall back, not fail."""
    ansatz = qk.hardware_efficient(2, 1)
    spec = ansatz.build()
    thetas = np.array([np.linspace(0.1, 0.9, ansatz.n_params) * s for s in (1.0, 0.8, 0.6)])
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.03))

    batched = backend.expectation_over(spec, thetas, qk.Z(0))
    one_at_a_time = [backend.expectation(spec.bind(t), qk.Z(0)) for t in thetas]
    assert batched == pytest.approx(one_at_a_time, abs=1e-12)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_shots_stack_on_top_of_noise(backend_name):
    """Two different error sources, and asking for one must not force the other."""
    spec = qk.angle_encode([0.7])
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.05))
    exact_under_noise = backend.expectation(spec, qk.Z(0))
    sampled = backend.expectation(spec, qk.Z(0), shots=100_000, seed=0)
    assert sampled == pytest.approx(exact_under_noise, abs=0.02)


# --------------------------------------------------------------------------- #
# The noise is real, and it is the noise that was asked for
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend_name", BACKENDS)
@pytest.mark.parametrize("p", [0.01, 0.05, 0.2])
def test_noise_reduces_purity_monotonically(backend_name, p):
    spec = bound(qk.hardware_efficient(2, 2))
    pure = qk.get_backend(backend_name).purity(spec)
    noisy = qk.get_backend(backend_name, noise=depolarizing(backend_name, p)).purity(spec)
    assert noisy < pure
    assert 0.0 < noisy <= 1.0


@pytest.mark.noisy
@pytest.mark.skipif(not is_available("cirq-density"), reason="needs Cirq")
@pytest.mark.parametrize("p", [0.0, 0.01, 0.05, 0.1])
def test_depolarizing_matches_the_analytic_value(p):
    """A closed form, so this checks the physics and not merely self-consistency.

    ``cirq.depolarize(p)`` applies a uniformly random non-identity Pauli with
    probability ``p``, which contracts any Pauli expectation by ``1 - 4p/3``. One
    gate means one noise moment, so the answer is exactly ``cos(theta)(1 - 4p/3)``.
    """
    import cirq

    theta = 0.7
    backend = qk.get_backend("cirq-density", noise=cirq.depolarize(p))
    got = qk.expectation(qk.angle_encode([theta]), qk.Z(0), backend=backend)
    assert got == pytest.approx(np.cos(theta) * (1 - 4 * p / 3), abs=1e-12)


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_noise_breaks_the_unit_diagonal_a_fidelity_kernel_assumes(backend_name):
    """``k(x, x) == 1`` is a fact about a *noiseless* compute-uncompute circuit.

    Under noise the circuit does not return to ``|0>``, so the diagonal drops below
    one. Anything that fills the diagonal in by assumption is writing down a number
    it did not measure — which is worth a test precisely because it still runs.
    """
    x = np.array([0.4, 0.5])
    backend = qk.get_backend(backend_name, noise=depolarizing(backend_name, 0.05))
    kernel = qk.QuantumKernel(qk.ZZFeatureMap(2), backend=backend)
    self_similarity = float(np.atleast_2d(kernel(x, x))[0, 0])
    assert self_similarity < 0.999
    assert qk.QuantumKernel(qk.ZZFeatureMap(2), backend="numpy")(x, x) == pytest.approx(
        1.0, abs=1e-10
    )


# --------------------------------------------------------------------------- #
# Noise never picks a backend for you
# --------------------------------------------------------------------------- #
def test_noise_without_a_named_backend_is_refused():
    with pytest.raises(ValueError, match="Name a mixed-state backend explicitly"):
        qk.get_backend(None, noise="anything")


@pytest.mark.parametrize("pure_backend", ["numpy", "qiskit", "cirq"])
def test_noise_on_a_pure_state_backend_is_refused(pure_backend):
    if not is_available(pure_backend):
        pytest.skip(f"{pure_backend} is not installed")
    with pytest.raises(ValueError, match="evolves a pure state"):
        qk.get_backend(pure_backend, noise="anything")


def test_the_refusal_names_the_backends_that_would_work():
    with pytest.raises(ValueError) as excinfo:
        qk.get_backend("numpy", noise="anything")
    message = str(excinfo.value)
    for name in NOISY_BACKENDS:
        assert name in message


@pytest.mark.parametrize("backend_name", BACKENDS)
def test_registered_and_reported(backend_name):
    assert backend_name in qk.list_backends()
    assert backend_name in qk.backend_report()
    assert qk.get_backend(backend_name).name == backend_name


# --------------------------------------------------------------------------- #
# diagnose() on a mixed-state backend
#
# The structure probes compare statevectors, which a density-matrix backend has
# none of. Pointing diagnose() at one used to raise NotImplementedError from four
# frames down - the diagnostics being the thing that breaks is the worst version
# of this bug, because it is what the caller reached for to find out.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("backend_name", ["cirq-density", "qiskit-aer"])
def test_diagnose_runs_on_a_mixed_state_backend(backend_name):
    if not is_available(backend_name):
        pytest.skip(f"{backend_name} is not installed")
    ansatz = qk.hardware_efficient(3, 2)
    report = qk.diagnose(ansatz, backend=backend_name, seed=0, n_samples=6, probes=1)
    assert report.codes == ()


@pytest.mark.parametrize("backend_name", ["cirq-density", "qiskit-aer"])
def test_diagnose_names_the_backend_that_could_not_answer(backend_name):
    """Substituting the reference silently would be its own plausible-wrong-number."""
    if not is_available(backend_name):
        pytest.skip(f"{backend_name} is not installed")
    report = qk.diagnose(qk.hardware_efficient(3, 2), backend=backend_name, seed=0, n_samples=6)
    assert "numpy reference" in report.subject
    assert backend_name in report.subject


@pytest.mark.parametrize("backend_name", ["cirq-density", "qiskit-aer"])
def test_a_notice_is_not_a_finding(backend_name):
    """``if qk.diagnose(model):`` has to keep meaning "something is wrong"."""
    if not is_available(backend_name):
        pytest.skip(f"{backend_name} is not installed")
    report = qk.diagnose(qk.hardware_efficient(3, 2), backend=backend_name, seed=0, n_samples=6)
    assert not report
    assert len(report) == 0


def test_diagnose_still_finds_structure_through_a_mixed_state_backend():
    """The substitution must not cost the findings it was made to preserve."""
    if not is_available("cirq-density"):
        pytest.skip("cirq is not installed")
    fmap = qk.AngleFeatureMap(2, rotation="ry")
    model = qk.Ansatz(2, qk.repeat(3, qk.EncodingLayer(fmap) + qk.RotationLayer("ry")), n_inputs=2)
    on_reference = qk.diagnose(model, backend="numpy", seed=0, n_samples=6, probes=1)
    on_noisy = qk.diagnose(model, backend="cirq-density", seed=0, n_samples=6, probes=1)
    assert "ENCODING_COMMUTES" in on_reference.codes
    assert "ENCODING_COMMUTES" in on_noisy.codes


def test_flat_gradients_does_not_claim_exactness_under_noise():
    """`supports_exact` is true on a mixed-state backend and is the wrong flag to ask.

    It means "shot-free", not "undisturbed": the number is exact *given the noise
    model*. What the finding is about to claim needs a pure state, and only
    `supports_statevector` says that.
    """
    cirq = pytest.importorskip("cirq")
    backend = qk.get_backend("cirq-density", noise=cirq.depolarize(0.1))
    assert backend.supports_exact is True  # the flag that would have been asked

    ansatz = qk.hardware_efficient(4, 5)
    observable = qk.Z(0) * qk.Z(1) * qk.Z(2) * qk.Z(3)
    report = qk.diagnose(ansatz, backend=backend, obs=observable, seed=0, n_samples=6, probes=1)

    flat = [f for f in report if f.code == "FLAT_GRADIENTS"]
    assert flat, "expected a flat-gradient finding on a deep ansatz under 10% depolarizing"
    assert "Gradients are exact here" not in flat[0].message
    assert "carries a noise model" in flat[0].message
