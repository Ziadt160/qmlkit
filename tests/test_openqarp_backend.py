"""OpenQARP-specific behaviour.

``test_cross_backend.py`` already asserts that this backend agrees with the NumPy
reference on statevectors, probabilities, expectations, seeded sampling, gradients and
registered gates. What it cannot ask is the question this backend raises by existing:
it computes expectations *inside* the SDK, so there are two routes to a number here
where every other statevector backend has one.

So these tests point the two routes at each other. ``native_expectations=False``
derives the answer from the statevector with this library's own contraction;
the default derives it in the C++ kernel from a state that never crosses the boundary.
They share the circuit translation and nothing else, and they are asserted to agree to
the last bit rather than to a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.registry import is_available

pytestmark = [
    pytest.mark.openqarp,
    pytest.mark.skipif(not is_available("openqarp"), reason="openqarp not installed"),
]

OBSERVABLE = qk.Z(0) + 0.5 * qk.X(1) + qk.ZZ(0, 2) + 0.25 * qk.Y(2)


def _ansatz(n_qubits: int = 3, reps: int = 2) -> qk.CircuitSpec:
    qc = qk.QCircuit(n_qubits)
    for _ in range(reps):
        qc.rotation_layer(("ry", "rz")).entangle("ring")
    return qc.to_spec()


def _bound(n_qubits: int = 3, reps: int = 2, seed: int = 17) -> qk.CircuitSpec:
    spec = _ansatz(n_qubits, reps)
    return spec.bind(np.random.default_rng(seed).uniform(-np.pi, np.pi, spec.n_params))


def _both() -> tuple[qk.Backend, qk.Backend]:
    """The contracted route and the statevector route, as two backends."""
    return (
        qk.get_backend("openqarp"),
        qk.get_backend("openqarp", native_expectations=False),
    )


# --------------------------------------------------------------------------- #
# the two routes
# --------------------------------------------------------------------------- #
def test_the_contracted_expectation_equals_the_statevector_one() -> None:
    """One backend, one circuit, two arithmetics, one number."""
    native, derived = _both()
    spec = _bound()
    for obs in (qk.Z(0), qk.X(1), qk.Y(2), qk.ZZ(0, 2), OBSERVABLE, 2.0 * qk.I()):
        assert native.expectation(spec, obs) == pytest.approx(
            derived.expectation(spec, obs), abs=1e-12
        ), f"the two routes disagree on <{obs}>"


def test_the_sweep_equals_the_row_by_row_path_exactly() -> None:
    """The sweep binds inside C++; the fallback binds in Python and re-translates.

    Bit-identical rather than merely close, because it is the same circuit and the
    same kernel either way - only the boundary moves. A tolerance here would hide the
    failure this test is for: a slot bound to the wrong symbol still lands close.
    """
    native, derived = _both()
    spec = _ansatz()
    rows = np.random.default_rng(3).uniform(-np.pi, np.pi, (40, len(spec.slots())))

    swept = native.expectation_over_slots(spec, rows, OBSERVABLE)
    row_by_row = np.array(
        [native.expectation(spec.with_slot_angles(r), OBSERVABLE) for r in rows], dtype=float
    )
    assert np.array_equal(swept, row_by_row)
    assert swept == pytest.approx(derived.expectation_over_slots(spec, rows, OBSERVABLE), abs=1e-12)


def test_the_sweep_binds_slots_and_not_parameters() -> None:
    """A weight-tied parameter fills several slots, and they move independently.

    The sweep names one symbol per *slot*. Named per logical parameter instead, this
    circuit would still build, still run, and still return numbers in range - with
    every occurrence of theta[0] forced to the same shift. That is the failure the
    slot abstraction exists to prevent, so it is asserted where the binding happens.
    """
    qc = qk.QCircuit(3)
    qc.ry(0, qk.ParamRef(0)).ry(1, qk.ParamRef(0)).crx(0, 2, qk.ParamRef(1))
    qc.rz(2, qk.ParamRef(2, scale=2.0, offset=0.3))
    spec = qc.to_spec()
    assert len(spec.occurrences_of(0)) == 2  # the tie this test is about

    rows = np.random.default_rng(5).uniform(-np.pi, np.pi, (16, len(spec.slots())))
    got = qk.get_backend("openqarp").expectation_over_slots(spec, rows, OBSERVABLE)
    ref = qk.get_backend("numpy").expectation_over_slots(spec, rows, OBSERVABLE)
    assert got == pytest.approx(ref, abs=1e-12)

    thetas = np.random.default_rng(6).uniform(-np.pi, np.pi, (16, spec.n_params))
    assert qk.get_backend("openqarp").expectation_over(spec, thetas, OBSERVABLE) == pytest.approx(
        qk.get_backend("numpy").expectation_over(spec, thetas, OBSERVABLE), abs=1e-12
    )


def test_a_batched_gradient_agrees_with_the_reference() -> None:
    """The call the sweep exists for: one batch of shifted circuits, one sweep."""
    spec = _ansatz(3, 1)
    thetas = np.random.default_rng(11).uniform(-np.pi, np.pi, (8, spec.n_params))
    got = qk.param_shift_grad_batch(spec, thetas, OBSERVABLE, backend="openqarp")
    ref = qk.param_shift_grad_batch(spec, thetas, OBSERVABLE, backend="numpy")
    assert got == pytest.approx(ref, abs=1e-9)


# --------------------------------------------------------------------------- #
# where the native route stands aside
# --------------------------------------------------------------------------- #
def test_a_non_hermitian_observable_raises_rather_than_returning_its_real_part() -> None:
    """``QarpSimulator.expectation`` returns ``Re <psi|H|psi>`` whatever it is given.

    A complex coefficient therefore comes back as a perfectly plausible real number
    instead of the error it is. The backend hands those to the base class, which is
    where the refusal lives.
    """
    spec = _bound()
    obs = (1.0 + 0.5j) * qk.Z(0)
    with pytest.raises(ValueError, match="not Hermitian"):
        qk.get_backend("openqarp").expectation(spec, obs)


def test_sampling_stays_with_the_shared_estimator() -> None:
    """Shots are the base class's, so a seed reproduces the reference's own counts."""
    spec = _bound()
    native = qk.get_backend("openqarp")
    assert native.counts(spec, shots=4096, seed=7) == qk.get_backend("numpy").counts(
        spec, shots=4096, seed=7
    )
    sampled = native.expectation(spec, OBSERVABLE, shots=200_000, seed=7)
    assert sampled == pytest.approx(native.expectation(spec, OBSERVABLE), abs=0.02)


def test_a_parametric_registered_gate_falls_back_and_still_agrees() -> None:
    """A registered gate reaches qarp as a matrix, and a matrix carries no symbol.

    So a custom gate whose angle is a *parameter* cannot be swept, and the backend
    gives way to the base class's loop instead of binding the wrong thing. The answer
    has to survive that handover.
    """
    from qmlkit.core.gates import _REGISTRY

    def _xy(t: float) -> np.ndarray:
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array(
            [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, 1]], dtype=complex
        )

    qk.register_gate(qk.GateDef("xy_sweep_test", 2, 1, _xy, frequencies=(1.0,)))
    try:
        qc = qk.QCircuit(3)
        qc.ry(0, qk.ParamRef(0)).apply("xy_sweep_test", (0, 2), qk.ParamRef(1))
        spec = qc.to_spec()
        rows = np.random.default_rng(13).uniform(-np.pi, np.pi, (6, len(spec.slots())))
        assert qk.get_backend("openqarp").expectation_over_slots(
            spec, rows, OBSERVABLE
        ) == pytest.approx(
            qk.get_backend("numpy").expectation_over_slots(spec, rows, OBSERVABLE), abs=1e-12
        )
    finally:
        _REGISTRY.pop("xy_sweep_test", None)


# --------------------------------------------------------------------------- #
# the native circuit, and the plumbing around it
# --------------------------------------------------------------------------- #
def test_the_native_block_is_a_usable_qarp_object() -> None:
    """``to_openqarp`` hands back a real block, and it agrees on its own terms.

    qarp indexes amplitudes LSB-first, so a block that had merely been translated
    would disagree with this backend here. It does not, because the translation maps
    qmlkit qubit ``i`` onto qarp qubit ``n-1-i`` and the two index conventions then
    describe the same vector.
    """
    backend = qk.get_backend("openqarp")
    spec = _bound()
    block = backend.to_openqarp(spec)
    assert block.n_gates() == len(spec.ops)
    assert np.allclose(np.asarray(block.statevector()), backend.statevector(spec), atol=1e-12)


def test_an_unbound_circuit_is_refused_by_name() -> None:
    spec = _ansatz()
    with pytest.raises(ValueError, match="free parameters"):
        qk.get_backend("openqarp").to_openqarp(spec)


def test_a_slot_count_mismatch_is_refused_before_the_sweep() -> None:
    """qarp would say 'unresolved symbolic parameter'; the caller asked about slots."""
    spec = _ansatz()
    rows = np.zeros((4, len(spec.slots()) - 1))
    with pytest.raises(ValueError, match="slot angles"):
        qk.get_backend("openqarp").expectation_over_slots(spec, rows, OBSERVABLE)


def test_the_sweep_reports_its_progress() -> None:
    """A Gram matrix is one call from outside and thousands of circuits from inside."""
    backend = qk.get_backend("openqarp")  # a fresh instance, so this mutation is local
    backend.sweep_rows = 4  # small enough that a 10-row batch reports three times
    spec = _ansatz(2, 1)
    rows = np.zeros((10, len(spec.slots())))
    with qk.progress(live=False) as run:
        backend.expectation_over_slots(spec, rows, qk.Z(0))
    assert [(r.label, r.items) for r in run.records] == [("openqarp circuits", 10)]


def test_repr_says_which_route_it_is_on() -> None:
    assert "native_expectations" not in repr(qk.get_backend("openqarp"))
    derived = qk.get_backend("openqarp", native_expectations=False)
    assert "native_expectations=False" in repr(derived)
