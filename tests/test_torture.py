"""Property-based torture tests: invariants that hold by mathematics, not by design.

The suite's other fuzzers draw from a fixed-seed RNG. They explore, but a failure
arrives as a 40-gate circuit and a wall of angles, and the same seed is the only way
back to it. These use Hypothesis, which *shrinks*: a failure comes back as the
smallest circuit that still exhibits it, and is replayed automatically thereafter.

Every property here is one that must hold for any correct implementation. None of
them is a design choice, so a failure is always a bug and never a disagreement about
what the library meant to do:

* the exact gradient routes are four derivations of the same derivative
* the backends are five implementations of the same linear algebra
* a batched call is an unrolled loop
* a tied weight's gradient is the sum over its occurrences (the chain rule)
* an adjoint undoes its circuit, probabilities are a distribution, expectation is
  linear in the observable

These are deliberately *not* comparisons against a stored expected value. A test
that pins today's output catches a change; a test that pins an invariant catches a
mistake, and only the second kind is worth running against randomly generated input.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import densesim
import qmlkit as qk
from qmlkit.core.gates import get_gate
from qmlkit.core.ir import CircuitSpec, Op, ParamRef

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

#: Gates carrying a differentiable angle, and the parameter-free ones. Kept explicit
#: rather than filtered from the registry at import: another module registering a
#: throwaway gate at run time must not silently widen what these tests fuzz over.
PARAMETRIC = ("rx", "ry", "rz", "phase", "crx", "cry", "crz")
STATIC = ("h", "x", "y", "z", "s", "sdg", "t", "tdg", "i", "cx", "cy", "cz", "swap")

#: How many circuits each property is tried on. The default keeps CI honest without
#: making it slow; set QMLKIT_TORTURE_EXAMPLES far higher before a release and let it
#: run for an hour. Hypothesis persists failures either way, so a deep campaign that
#: finds something turns into a fast regression test on the next ordinary run.
TORTURE_EXAMPLES = int(os.environ.get("QMLKIT_TORTURE_EXAMPLES", "150"))

TORTURE = settings(
    max_examples=TORTURE_EXAMPLES,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# --------------------------------------------------------------------------- #
# generating circuits
# --------------------------------------------------------------------------- #
@st.composite
def circuits(draw, min_qubits: int = 1, max_qubits: int = 4, max_ops: int = 12):
    """A random parametric circuit over the whole gate set.

    Angles are `ParamRef`s so the result can be differentiated; the caller binds.
    Two-qubit gates are only drawn when there are two wires to put them on, and the
    wire pair is ordered randomly so control-before-target is never assumed.
    """
    n_qubits = draw(st.integers(min_qubits, max_qubits))
    n_ops = draw(st.integers(1, max_ops))
    ops: list[Op] = []
    n_params = 0
    for _ in range(n_ops):
        pool = (
            PARAMETRIC + STATIC
            if n_qubits > 1
            else tuple(g for g in PARAMETRIC + STATIC if get_gate(g).n_qubits == 1)
        )
        name = draw(st.sampled_from(sorted(pool)))
        gate = get_gate(name)
        if gate.n_qubits > n_qubits:
            continue
        wires = tuple(
            draw(st.permutations(range(n_qubits)).map(lambda w: tuple(w)))[: gate.n_qubits]
        )
        params: tuple[object, ...] = ()
        if gate.n_params:
            params = tuple(ParamRef(n_params + i) for i in range(gate.n_params))
            n_params += gate.n_params
        ops.append(Op(name, wires, params))

    # Pad untouched wires with an explicit identity. Cirq has no declared register, so
    # a qubit no operation touches is simply not in its circuit - which is correct and
    # documented, and would make the round-trip properties below compare a 3-qubit
    # state against a 2-qubit one. Padding keeps the unitary and keeps the register.
    touched = {q for op in ops for q in op.qubits}
    ops.extend(Op("i", (q,)) for q in range(n_qubits) if q not in touched)
    return CircuitSpec(n_qubits=n_qubits, ops=tuple(ops), n_params=n_params)


@st.composite
def circuit_and_angles(draw, **kwargs):
    spec = draw(circuits(**kwargs))
    theta = np.array(
        draw(
            st.lists(
                st.floats(-np.pi, np.pi, allow_nan=False, allow_infinity=False),
                min_size=spec.n_params,
                max_size=spec.n_params,
            )
        ),
        dtype=float,
    )
    return spec, theta


def _observable(n_qubits: int):
    """A multi-term observable that actually spans the register."""
    obs = qk.Z(0)
    if n_qubits > 1:
        obs = obs + 0.5 * qk.ZZ(0, n_qubits - 1) + 0.25 * qk.X(n_qubits - 1)
    return obs


# --------------------------------------------------------------------------- #
# the gradient routes are four derivations of one derivative
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles())
def test_every_exact_gradient_route_agrees(case):
    """adjoint, parameter-shift and hadamard compute the same number or the library
    is wrong somewhere, and which one is wrong is not knowable from one of them."""
    spec, theta = case
    if spec.n_params == 0:
        return
    obs = _observable(spec.n_qubits)
    reference = qk.grad(spec, theta, obs, method="adjoint")
    for method in ("parameter-shift", "hadamard"):
        try:
            other = qk.grad(spec, theta, obs, method=method)
        except (NotImplementedError, ValueError):
            continue  # a route that refuses by name is not a disagreement
        assert other == pytest.approx(reference, abs=1e-9), f"{method} disagrees with adjoint"


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_finite_differences_agree_to_their_own_accuracy(case):
    """The one route with no shared machinery: if every closed form were wrong the
    same way, this is what would notice."""
    spec, theta = case
    if spec.n_params == 0:
        return
    obs = _observable(spec.n_qubits)
    exact = qk.grad(spec, theta, obs, method="adjoint")
    numeric = qk.grad(spec, theta, obs, method="finite-diff")
    assert numeric == pytest.approx(exact, abs=2e-5)


# --------------------------------------------------------------------------- #
# the chain rule, where slot-space bugs live
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles(min_qubits=1, max_qubits=3, max_ops=6))
def test_a_tied_weight_accumulates_its_occurrences(case):
    """Bind every gate to ONE logical parameter. Its gradient must equal the sum of
    what each occurrence would contribute - a shift rule moves one occurrence, so a
    gradient computed in logical space rather than slot space is silently wrong here
    and nowhere else."""
    spec, theta = case
    if spec.n_params < 2:
        return
    tied_ops = tuple(
        Op(op.gate, op.qubits, tuple(ParamRef(0) for _ in op.params)) for op in spec.ops
    )
    tied = CircuitSpec(n_qubits=spec.n_qubits, ops=tied_ops, n_params=1)
    obs = _observable(spec.n_qubits)
    value = float(theta[0])

    tied_grad = qk.grad(tied, np.array([value]), obs, method="adjoint")
    shift_grad = qk.grad(tied, np.array([value]), obs, method="parameter-shift")
    assert tied_grad == pytest.approx(shift_grad, abs=1e-9)

    # and against the untied circuit evaluated with every slot at the same angle
    untied_grad = qk.grad(spec, np.full(spec.n_params, value), obs, method="adjoint")
    assert float(tied_grad[0]) == pytest.approx(float(untied_grad.sum()), abs=1e-9)


# --------------------------------------------------------------------------- #
# the backends are implementations of one linear algebra
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_every_installed_backend_agrees_with_the_reference(case):
    from qmlkit.core.backends.registry import available_backends

    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    obs = _observable(spec.n_qubits)
    reference = qk.expectation(bound, obs, backend="numpy")
    for name in available_backends():
        # torch is NOT excluded. It was, on the reasoning that a differentiable
        # simulator is a different kind of backend - and that exclusion is exactly
        # what let a wrong `moveaxis` permute untouched wires for two-qubit gates on
        # descending wire pairs. An agreement test that skips a backend tests nothing
        # about that backend.
        if name == "numpy":
            continue
        tolerance = 1e-7 if name == "spinqit" else 1e-10
        assert qk.expectation(bound, obs, backend=name) == pytest.approx(
            reference, abs=tolerance
        ), f"{name} disagrees with the NumPy reference"


# --------------------------------------------------------------------------- #
# a batched call is an unrolled loop
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_batched_expectation_equals_the_loop(case):
    spec, theta = case
    if spec.n_params == 0:
        return
    obs = _observable(spec.n_qubits)
    batch = np.stack([theta, theta * 0.5, np.zeros_like(theta)])
    batched = qk.expectation_over(spec, batch, obs)
    looped = [qk.expectation(spec.bind(row), obs) for row in batch]
    assert batched == pytest.approx(looped, abs=1e-11)


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=6))
def test_batched_gradient_equals_the_loop(case):
    spec, theta = case
    if spec.n_params == 0:
        return
    obs = _observable(spec.n_qubits)
    batch = np.stack([theta, theta * 0.5])
    batched = qk.grad_batch(spec, batch, obs)
    for row, angles in zip(batched, batch, strict=True):
        assert row == pytest.approx(qk.grad(spec, angles, obs), abs=1e-10)


# --------------------------------------------------------------------------- #
# structural invariants
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles())
def test_the_state_is_normalised_and_the_probabilities_are_a_distribution(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    state = qk.statevector(bound)
    assert float(np.vdot(state, state).real) == pytest.approx(1.0, abs=1e-10)
    probabilities = qk.probabilities(bound)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-10)
    assert (probabilities >= -1e-12).all()


@TORTURE
@given(circuit_and_angles())
def test_a_circuit_composed_with_its_adjoint_is_the_identity(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    undone = bound.compose(bound.adjoint())
    state = qk.statevector(undone)
    expected = np.zeros(2**spec.n_qubits, dtype=complex)
    expected[0] = 1.0
    assert np.abs(state - expected).max() == pytest.approx(0.0, abs=1e-9)


@TORTURE
@given(circuit_and_angles(max_qubits=3))
def test_expectation_is_linear_in_the_observable(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    a, b = qk.Z(0), qk.X(0)
    combined = qk.expectation(bound, 2.0 * a + 3.0 * b)
    separate = 2.0 * qk.expectation(bound, a) + 3.0 * qk.expectation(bound, b)
    assert combined == pytest.approx(separate, abs=1e-11)


@TORTURE
@given(circuit_and_angles(max_qubits=3))
def test_every_expectation_lies_inside_the_observables_spectrum(case):
    """A Pauli sum's expectation cannot exceed the sum of |coefficients|."""
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    obs = _observable(spec.n_qubits)
    bound_max = sum(abs(term.coeff.real) for term in qk.core.observables.iter_terms(obs))
    assert abs(qk.expectation(bound, obs)) <= bound_max + 1e-10


# --------------------------------------------------------------------------- #
# interop is a round trip, not a hope
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_qasm_round_trip_reproduces_the_statevector(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    qiskit = pytest.importorskip("qiskit")  # noqa: F841
    returned = qk.from_qiskit(qk.get_backend("qiskit").to_qiskit(bound))
    assert qk.statevector(returned) == pytest.approx(qk.statevector(bound), abs=1e-10)


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_cirq_round_trip_reproduces_the_statevector(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    pytest.importorskip("cirq")
    returned = qk.from_cirq(qk.get_backend("cirq").to_cirq(bound))
    assert qk.statevector(returned) == pytest.approx(qk.statevector(bound), abs=1e-10)


# --------------------------------------------------------------------------- #
# sampling converges on the exact answer
# --------------------------------------------------------------------------- #
@settings(
    max_examples=max(10, TORTURE_EXAMPLES // 6),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(circuit_and_angles(max_qubits=3, max_ops=6))
def test_sampling_converges_on_the_exact_expectation(case):
    """Shot noise has a known size; anything outside it is bias, not variance."""
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    obs = qk.Z(0)
    exact = qk.expectation(bound, obs)
    sampled = qk.expectation(bound, obs, shots=40_000, seed=0)
    # five standard errors of a bounded-[-1,1] estimator at 40k shots
    assert abs(sampled - exact) < 5 * (1.0 / np.sqrt(40_000)) + 1e-9


# --------------------------------------------------------------------------- #
# an opinion that shares nothing with the library
#
# Every other property here is internal consistency: four gradient derivations
# agreeing, five backends agreeing, batched matching looped. All of that passes if
# they share one wrong convention. `densesim` inherited nothing - hand-written gate
# matrices, hand-derived derivatives, reads only spec.ops - so it is the only check
# in this file that can catch a mistake they would all make together.
# --------------------------------------------------------------------------- #
@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_the_dense_reference_agrees_on_the_statevector(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    assert qk.statevector(bound) == pytest.approx(densesim.state(spec, theta), abs=1e-11)


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_the_dense_reference_agrees_on_the_expectation(case):
    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    obs = _observable(spec.n_qubits)
    assert qk.expectation(bound, obs) == pytest.approx(densesim.expval(spec, obs, theta), abs=1e-11)


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=6))
def test_the_dense_reference_agrees_on_the_gradient(case):
    """Hand-derived product rule against the adjoint sweep, on random circuits."""
    spec, theta = case
    if spec.n_params == 0:
        return
    obs = _observable(spec.n_qubits)
    assert qk.grad(spec, theta, obs) == pytest.approx(densesim.grad(spec, theta, obs), abs=1e-10)


@TORTURE
@given(circuit_and_angles(max_qubits=3, max_ops=8))
def test_the_dense_reference_agrees_on_every_backend(case):
    """The property that would have caught the torch permutation bug on day one."""
    from qmlkit.core.backends.registry import available_backends

    spec, theta = case
    bound = spec.bind(theta) if spec.n_params else spec
    obs = _observable(spec.n_qubits)
    truth = densesim.expval(spec, obs, theta)
    for name in available_backends():
        tolerance = 1e-7 if name == "spinqit" else 1e-10
        assert qk.expectation(bound, obs, backend=name) == pytest.approx(truth, abs=tolerance), (
            f"{name} disagrees with the independent dense reference"
        )
