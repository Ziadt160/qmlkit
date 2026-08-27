"""The circuit budget.

The arithmetic has to be right in the way a caller will check it: the total must be
exactly samples x steps x per-gradient x measurement settings, and the reductions
must be the real cost of the alternative method rather than a rule of thumb. Both
are asserted against :func:`~qmlkit.gradients.dispatch.gradient_cost` itself, so a
change to a shift rule moves this test rather than quietly invalidating it.
"""

from __future__ import annotations

import pytest

import qmlkit as qk
from qmlkit.budget import plan


def test_total_is_the_product_it_claims_to_be():
    ansatz = qk.hardware_efficient(4, 3)
    spec = ansatz.build()
    per = qk.gradient_cost(spec, "parameter-shift")
    got = plan(ansatz, n_samples=100, steps=50, obs=qk.Z(0))
    assert got.circuits == per * 1 * 100 * 50  # Z(0) is a single measurement setting
    assert got.circuits_per_gradient == per


def test_qubit_wise_commuting_grouping_is_counted_not_assumed():
    """Z0, Z1 and Z0Z1 are all diagonal in Z, so they share one circuit."""
    ansatz = qk.hardware_efficient(3, 2)
    obs = qk.Z(0) + qk.Z(1) + qk.ZZ(0, 1)
    got = plan(ansatz, n_samples=10, steps=10, obs=obs)
    assert got.observable_terms == 3
    assert got.measurement_settings == 1
    assert any("grouping is already saving" in note for note in got.notes)


def test_non_commuting_terms_cost_separate_settings():
    ansatz = qk.hardware_efficient(2, 1)
    got = plan(ansatz, obs=qk.X(0) + qk.Z(0))
    assert got.observable_terms == 2
    assert got.measurement_settings == 2
    assert got.circuits == 2 * qk.gradient_cost(ansatz.build(), "parameter-shift")


def test_adjoint_is_offered_as_the_cheaper_route_with_its_trade():
    got = plan(qk.hardware_efficient(4, 3), n_samples=100, steps=50)
    adjoint = next(r for r in got.reductions if r.name == "adjoint")
    assert adjoint.circuits < got.circuits
    assert adjoint.factor > 1.0
    assert "simulator only" in adjoint.trade
    # the reduction is the real cost of that method, not an estimate
    assert adjoint.circuits == (
        qk.gradient_cost(qk.hardware_efficient(4, 3).build(), "adjoint")
        * got.measurement_settings
        * 100
        * 50
    )


def test_planning_from_adjoint_offers_no_cheaper_alternative():
    got = plan(qk.hardware_efficient(3, 2), n_samples=5, steps=5, method="adjoint")
    assert got.reductions == ()  # nothing on a simulator beats one backward pass


def test_shots_multiply_through_and_exact_reports_none():
    ansatz = qk.hardware_efficient(3, 2)
    sampled = plan(ansatz, n_samples=4, steps=4, shots=1024)
    exact = plan(ansatz, n_samples=4, steps=4)
    assert sampled.shots_total == sampled.circuits * 1024
    assert exact.shots_total is None
    assert "exact, no shots" in str(exact)


def test_hours_scale_linearly_with_the_queue():
    got = plan(qk.hardware_efficient(3, 2), n_samples=10, steps=10)
    assert got.hours(2.0) == pytest.approx(2 * got.hours(1.0))
    assert got.hours(0.5) == pytest.approx(got.circuits * 0.5 / 3600)


def test_a_circuit_spec_and_a_model_are_both_accepted():
    spec = qk.hardware_efficient(3, 2).build()
    assert plan(spec, n_samples=2, steps=2).circuits > 0

    torch = pytest.importorskip("torch")  # noqa: F841
    model = qk.VQC(n_features=3, n_classes=2, seed=0)
    got = plan(model, n_samples=20, steps=10)
    assert got.circuits > 0
    assert got.n_params > 0


def test_a_type_it_cannot_read_says_what_it_takes():
    with pytest.raises(TypeError, match="Ansatz"):
        plan("not a circuit")


def test_the_printout_states_the_totals_and_the_trades():
    text = str(plan(qk.hardware_efficient(4, 3), n_samples=100, steps=50, shots=1024))
    assert "circuits" in text
    assert "hours" in text
    assert "cheaper, and what it costs" in text
    assert text.isascii()  # this prints to a Windows console as often as a notebook
