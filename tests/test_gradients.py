"""Gradient tests.

Two of these exist specifically to catch the corrections made during the design
review — a per-gate rule lookup, and per-occurrence shifting for weight-tied
parameters. Both bugs produce a plausible wrong number rather than an exception,
so only a test catches them.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.gradients.rules import rule_for_frequencies, second_derivative_rule


# --------------------------------------------------------------------------- #
# shift rules, derived rather than remembered
# --------------------------------------------------------------------------- #
def test_two_term_rule_is_the_familiar_one():
    rule = qk.two_term_rule()
    assert sorted(rule.shifts) == pytest.approx([-np.pi / 2, np.pi / 2])
    coeff = dict(zip(rule.shifts, rule.coeffs, strict=False))
    assert coeff[np.pi / 2] == pytest.approx(0.5)
    assert coeff[-np.pi / 2] == pytest.approx(-0.5)


def test_four_term_rule_reproduces_the_textbook_constants():
    """Derived rule must match the published controlled-rotation coefficients."""
    rule = qk.four_term_rule()
    got = dict(zip(np.round(rule.shifts, 12), rule.coeffs, strict=False))
    d1 = (np.sqrt(2) + 1) / (4 * np.sqrt(2))
    d2 = (np.sqrt(2) - 1) / (4 * np.sqrt(2))
    assert got[round(np.pi / 2, 12)] == pytest.approx(d1)
    assert got[round(-np.pi / 2, 12)] == pytest.approx(-d1)
    assert got[round(3 * np.pi / 2, 12)] == pytest.approx(-d2)
    assert got[round(-3 * np.pi / 2, 12)] == pytest.approx(d2)


@pytest.mark.parametrize("freqs", [(1.0,), (0.5, 1.0), (1.0, 2.0), (0.5, 1.0, 1.5)])
def test_shift_rule_differentiates_its_own_fourier_family(freqs):
    """A rule for frequencies W must differentiate every series built from W."""
    rng = np.random.default_rng(0)
    a0 = rng.normal()
    a = rng.normal(size=len(freqs))
    b = rng.normal(size=len(freqs))

    def f(t):
        return a0 + sum(
            ai * np.cos(w * t) + bi * np.sin(w * t) for w, ai, bi in zip(freqs, a, b, strict=False)
        )

    def fprime(t):
        return sum(
            -ai * w * np.sin(w * t) + bi * w * np.cos(w * t)
            for w, ai, bi in zip(freqs, a, b, strict=False)
        )

    rule = rule_for_frequencies(freqs)
    for theta in np.linspace(-2.0, 2.0, 7):
        got = sum(c * f(theta + s) for s, c in zip(rule.shifts, rule.coeffs, strict=False))
        assert got == pytest.approx(fprime(theta), abs=1e-9)


def test_rule_lookup_refuses_a_gate_with_no_declared_frequencies():
    qk.register_gate(qk.GateDef("mystery", 1, 1, lambda t: np.eye(2, dtype=complex)))
    with pytest.raises(ValueError, match="no generator frequencies"):
        qk.rule_for_gate("mystery")


# --------------------------------------------------------------------------- #
# the analytic reference: <Z> of Ry(theta)|0> is cos(theta)
# --------------------------------------------------------------------------- #
def test_param_shift_matches_analytic_derivative_of_cos():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    spec = qc.to_spec()

    for theta in np.linspace(0, 2 * np.pi, 13):
        g = qk.param_shift_grad_circuit(spec, [theta], qk.Z(0))
        assert g[0] == pytest.approx(-np.sin(theta), abs=1e-12)


def test_param_shift_matches_finite_differences_on_a_layered_circuit():
    qc = qk.QCircuit(3)
    qc.rotation_layer(("ry", "rz")).entangle("ring").rotation_layer(("ry",))
    spec = qc.to_spec()
    rng = np.random.default_rng(3)
    theta = rng.uniform(-np.pi, np.pi, spec.n_params)
    obs = qk.Z(0) + 0.5 * qk.ZZ(1, 2)

    exact = qk.param_shift_grad_circuit(spec, theta, obs)

    def f(t):
        return qk.expectation(spec, obs, theta=t)

    approx = qk.finite_diff_grad(f, theta, eps=1e-5)
    assert exact == pytest.approx(approx, abs=1e-6)


# --------------------------------------------------------------------------- #
# correction (a): the shift rule is a property of the gate, not the call
# --------------------------------------------------------------------------- #
def _mixed_ry_crz_circuit():
    """One- and two-frequency generators in one circuit.

    The Hadamards matter. A CRZ only exposes its frequency-1/2 component when
    something creates coherence *across* the control qubit; without them the
    measured value has frequency 1 only, both rules agree, and the test would
    pass while proving nothing.
    """
    qc = qk.QCircuit(2)
    qc.h(0)
    qc.ry(1, qk.ParamRef(0))
    qc.crz(0, 1, qk.ParamRef(1))
    qc.h(0)
    qc.ry(1, qk.ParamRef(2))
    return qc.to_spec()


def test_crz_really_carries_a_half_frequency_component():
    """Guard the guard: confirm the test circuit can discriminate the two rules."""
    spec = _mixed_ry_crz_circuit()
    theta = np.array([0.9, 0.0, 0.5])
    ts = np.linspace(0, 4 * np.pi, 256, endpoint=False)
    vals = []
    for t in ts:
        th = theta.copy()
        th[1] = t
        vals.append(qk.expectation(spec, qk.Z(0), theta=th))
    amp = np.abs(np.fft.rfft(vals)) / len(ts)
    assert amp[1] > 0.1, "no frequency-1/2 content: the circuit cannot tell the rules apart"


def test_mixed_ry_and_crz_needs_two_different_rules():
    """Applying one uniform rule returns a plausible wrong gradient, with no error.

    Swept over the CRZ angle rather than probed at one random point: the two rules
    coincide wherever the gradient happens to pass through zero, so a single sample
    can pass by luck.
    """
    spec = _mixed_ry_crz_circuit()
    obs = qk.Z(0) + qk.Z(1)
    uniform_rules = {i: qk.two_term_rule() for i in range(len(spec.slots()))}

    worst_gap = 0.0
    for crz_angle in np.linspace(0.2, 3.0, 8):
        theta = np.array([0.9, crz_angle, 0.5])

        exact = qk.param_shift_grad_circuit(spec, theta, obs)

        def f(t):
            return qk.expectation(spec, obs, theta=t)

        # the per-gate rules are exact, everywhere on the sweep
        assert exact == pytest.approx(qk.finite_diff_grad(f, theta, eps=1e-6), abs=1e-6)

        uniform = qk.param_shift_grad(
            lambda a: qk.expectation(spec.with_slot_angles(a), obs),
            spec,
            theta,
            rules=uniform_rules,
        )
        # the Ry entries agree — they are single-frequency either way
        assert uniform[0] == pytest.approx(exact[0], abs=1e-9)
        assert uniform[2] == pytest.approx(exact[2], abs=1e-9)
        worst_gap = max(worst_gap, abs(uniform[1] - exact[1]))

    # ...but the CRZ entry is materially wrong, so the assertions above are not vacuous
    assert worst_gap > 0.05


def test_grad_circuit_cost_is_not_a_flat_2P():
    """A CRZ costs four evaluations, not two."""
    qc = qk.QCircuit(2)
    qc.ry(0, qk.ParamRef(0))
    qc.crz(0, 1, qk.ParamRef(1))
    spec = qc.to_spec()
    assert spec.n_params == 2
    assert qk.grad_circuit_cost(spec) == 2 + 4


# --------------------------------------------------------------------------- #
# correction (b): shared parameters shift one occurrence at a time
# --------------------------------------------------------------------------- #
def test_weight_tied_parameter_sums_over_occurrences():
    """One logical parameter driving several gates — the QCNN's shared block.

    Shifting all occurrences together computes a different derivative entirely.
    """
    qc = qk.QCircuit(3)
    shared = qc.param()
    qc.rotation_layer(("ry",), shared=shared)  # one parameter, three gates
    qc.entangle("chain")
    spec = qc.to_spec()

    assert spec.n_params == 1
    assert len(spec.occurrences_of(0)) == 3

    obs = qk.Z(0) + qk.Z(1) + qk.Z(2)
    for theta in np.linspace(-1.5, 1.5, 7):
        exact = qk.param_shift_grad_circuit(spec, [theta], obs)

        def f(t):
            return qk.expectation(spec, obs, theta=t)

        approx = qk.finite_diff_grad(f, [theta], eps=1e-5)
        assert exact[0] == pytest.approx(approx[0], abs=1e-6)


def test_naive_simultaneous_shift_gets_the_tied_case_wrong():
    """Guard against a regression to the simultaneous-shift bug.

    Two Ry gates sharing a parameter *on the same qubit* compose to ``Ry(2t)``, so
    the measured value has frequency 2. Shifting both occurrences together applies
    a frequency-1 rule to a frequency-2 function and returns exactly zero.

    Note the sibling case: two tied rotations on *different* qubits sum to
    ``2cos(t)``, still frequency 1, where the naive shift accidentally agrees. A
    test built on that circuit would pass while proving nothing.
    """
    qc = qk.QCircuit(1)
    shared = qc.param()
    qc.ry(0, shared)
    qc.ry(0, shared)
    spec = qc.to_spec()
    assert len(spec.occurrences_of(0)) == 2

    def f(t):
        return qk.expectation(spec, qk.Z(0), theta=t)

    for theta in (0.4, 0.9, 1.6):
        correct = qk.param_shift_grad_circuit(spec, [theta], qk.Z(0))[0]
        assert correct == pytest.approx(-2 * np.sin(2 * theta), abs=1e-10)

        naive = 0.5 * (f([theta + np.pi / 2]) - f([theta - np.pi / 2]))
        assert naive == pytest.approx(0.0, abs=1e-12)  # catastrophically wrong
        assert naive != pytest.approx(correct, abs=1e-3)


def test_tied_rotations_on_separate_qubits_stay_single_frequency():
    """The benign sibling of the case above — documented so it is not mistaken for a bug."""
    qc = qk.QCircuit(2)
    shared = qc.param()
    qc.rotation_layer(("ry",), shared=shared)
    spec = qc.to_spec()
    obs = qk.Z(0) + qk.Z(1)
    assert qk.param_shift_grad_circuit(spec, [0.7], obs)[0] == pytest.approx(
        -2 * np.sin(0.7), abs=1e-10
    )


def test_param_ref_scaling_applies_the_chain_rule():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0, scale=2.0))
    spec = qc.to_spec()
    theta = np.array([0.3])
    g = qk.param_shift_grad_circuit(spec, theta, qk.Z(0))
    assert g[0] == pytest.approx(-2.0 * np.sin(2.0 * 0.3), abs=1e-12)


# --------------------------------------------------------------------------- #
# gradients with respect to the *inputs* — the dressed-circuit fix
# --------------------------------------------------------------------------- #
def test_gradient_flows_through_the_encoding():
    """df/dx exists, which is what a classical pre-net needs to train at all."""
    spec = qk.angle_encode([0.4, 1.1], trainable=True)
    x = np.array([0.4, 1.1])
    g = qk.param_shift_grad_circuit(spec, x, qk.Z(0))
    assert g[0] == pytest.approx(-np.sin(0.4), abs=1e-12)
    assert g[1] == pytest.approx(0.0, abs=1e-12)


def test_second_derivative_rule_on_a_pauli_rotation():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    spec = qc.to_spec()
    rule = second_derivative_rule()
    assert rule.needs_unshifted

    for theta in np.linspace(0.1, 2.0, 5):
        second = qk.param_shift_grad(
            lambda a: qk.expectation(spec.with_slot_angles(a), qk.Z(0)),
            spec,
            [theta],
            rules={0: rule},
            f0=qk.expectation(spec, qk.Z(0), theta=[theta]),
        )
        assert second[0] == pytest.approx(-np.cos(theta), abs=1e-10)
