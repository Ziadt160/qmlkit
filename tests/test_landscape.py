"""Gradient statistics, plateau attribution, curvature, minima, overparametrisation.

Every finding here is tested by asserting the thing it claims is **true**, never that
it fired. A landscape tool that cries plateau on a healthy circuit teaches people to
ignore it, so the false-positive guards below matter more than the positive cases.
"""

from __future__ import annotations

import operator
import warnings
from functools import reduce

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.ansatz import Ansatz, EncodingLayer, EntanglerLayer, RotationLayer, repeat
from qmlkit.landscape import (
    gradient_stats,
    hessian_spectrum,
    landscape,
    minima_scan,
    overparametrisation,
    plateau_mechanism,
)
from qmlkit.metrics import effective_dimension, fisher_information, gradient_variance


def _global_z(n: int) -> qk.PauliString:
    return reduce(operator.mul, [qk.Z(i) for i in range(n)])


def _reupload(n_qubits: int = 3, layers: int = 2) -> Ansatz:
    fmap = qk.AngleFeatureMap(n_qubits, rotation="ry")
    block = EncodingLayer(fmap) + RotationLayer(("ry", "rz")) + EntanglerLayer("cz", "ring")
    return Ansatz(n_qubits, repeat(layers, block), name="reupload")


# --------------------------------------------------------------------------- #
# gradient statistics
# --------------------------------------------------------------------------- #
def test_per_parameter_variance_agrees_with_the_single_parameter_probe():
    """Cross-validation against the older one-at-a-time implementation.

    The batched path reaches BLAS and the scalar one does not, so agreement here is
    the check that the speedup did not change the number.
    """
    ansatz = qk.hardware_efficient(3, 2)
    stats = gradient_stats(ansatz, qk.Z(0), n_samples=64, seed=5)
    for index in (0, 1, 8):
        with warnings.catch_warnings():
            # Index 8 is silent against Z(0), and gradient_variance warns about exactly
            # that. Here the warning is the behaviour under test, not a surprise.
            warnings.simplefilter("ignore", UserWarning)
            reference = gradient_variance(ansatz, qk.Z(0), n_samples=64, param_index=index, seed=5)
        assert stats.variance[index] == pytest.approx(reference, rel=1e-9, abs=1e-12)


def test_silent_parameters_really_have_no_gradient():
    """Assert the claim, not the flag: those entries are zero at fresh random points."""
    ansatz = qk.hardware_efficient(4, 2)
    stats = gradient_stats(ansatz, qk.Z(0), n_samples=40, seed=1)
    assert stats.silent.size, "this ansatz is the fixture precisely because it has some"

    spec = ansatz.build()
    rng = np.random.default_rng(99)  # a different seed from the one measured with
    for _ in range(5):
        g = qk.grad(spec, rng.uniform(-np.pi, np.pi, ansatz.n_params), qk.Z(0))
        assert np.allclose(g[stats.silent], 0.0, atol=1e-12)
        assert np.any(np.abs(g[stats.live]) > 1e-9)


def test_live_and_silent_partition_every_parameter():
    stats = gradient_stats(qk.hardware_efficient(3, 2), n_samples=20, seed=0)
    assert stats.live.size + stats.silent.size == stats.n_params
    assert not set(stats.live.tolist()) & set(stats.silent.tolist())


def test_variance_carries_its_sampling_error():
    """A variance from 30 draws is good to about 26%, and says so."""
    stats = gradient_stats(qk.hardware_efficient(3, 1), n_samples=30, seed=0)
    assert stats.relative_error == pytest.approx(np.sqrt(2 / 29))
    assert 0.2 < stats.relative_error < 0.3


# --------------------------------------------------------------------------- #
# mechanism attribution
# --------------------------------------------------------------------------- #
def test_a_healthy_shallow_ansatz_is_reported_clean():
    """The false-positive guard. A trainable circuit must produce no mechanism."""
    assert plateau_mechanism(qk.hardware_efficient(4, 1), qk.Z(0), n_samples=80, seed=0) == []


def test_a_local_cost_on_a_deep_circuit_is_not_called_global():
    codes = [f.code for f in plateau_mechanism(qk.hardware_efficient(5, 6), qk.Z(0), n_samples=80)]
    assert "PLATEAU_COST_GLOBALITY" not in codes


def test_cost_globality_fires_and_the_claim_it_makes_is_true():
    """The finding says a local observable would do better. Check that it would."""
    n, ansatz = 6, qk.hardware_efficient(6, 1)
    found = plateau_mechanism(ansatz, _global_z(n), n_samples=150, seed=0)
    globality = [f for f in found if f.code == "PLATEAU_COST_GLOBALITY"]
    assert globality, "a global cost on a shallow circuit is the textbook case"

    glob = gradient_stats(ansatz, _global_z(n), n_samples=150, seed=0).typical_variance
    local = gradient_stats(ansatz, qk.Z(0), n_samples=150, seed=0).typical_variance
    assert local > glob, "the fix the finding recommends has to actually help"
    assert globality[0].value == pytest.approx(local / glob, rel=1e-6)


def test_the_globality_gap_widens_with_width():
    """The mechanism's claim is about scaling, so the test is about scaling.

    Measured: the local variance is flat in ``n`` while the global one halves every
    couple of qubits, so the ratio has to grow.
    """
    ratios = []
    for n in (3, 7):
        ansatz = qk.hardware_efficient(n, 1)
        glob = gradient_stats(ansatz, _global_z(n), n_samples=200, seed=0).typical_variance
        local = gradient_stats(ansatz, qk.Z(0), n_samples=200, seed=0).typical_variance
        ratios.append(local / glob)
    assert ratios[1] > 2 * ratios[0]


# --------------------------------------------------------------------------- #
# curvature
# --------------------------------------------------------------------------- #
def test_a_converged_minimum_has_no_descent_direction():
    ansatz = qk.hardware_efficient(3, 2)
    scan = minima_scan(ansatz, qk.Z(0), n_starts=2, n_steps=600, seed=0, classify=False)
    point = hessian_spectrum(ansatz.build(), scan.best_theta, qk.Z(0))
    assert point.is_stationary
    assert point.kind == "minimum"
    assert point.index == 0
    assert np.all(point.eigenvalues > -1e-6 * max(point.scale, 1.0))


def test_a_point_still_on_a_slope_is_not_classified_as_a_minimum():
    """The honest answer to an unconverged run is 'not stationary', not 'minimum'."""
    ansatz = qk.hardware_efficient(3, 2)
    rng = np.random.default_rng(4)
    point = hessian_spectrum(ansatz.build(), rng.uniform(-np.pi, np.pi, ansatz.n_params), qk.Z(0))
    assert not point.is_stationary
    assert point.kind == "not-stationary"


def test_flat_directions_are_counted_at_the_minimum():
    """A 16-parameter circuit whose solution manifold is 14-dimensional has 14 flat."""
    ansatz = qk.hardware_efficient(4, 2)
    scan = minima_scan(ansatz, qk.Z(0), n_starts=2, n_steps=600, seed=0, classify=False)
    point = hessian_spectrum(ansatz.build(), scan.best_theta, qk.Z(0))
    assert point.n_flat > 0
    assert point.n_flat + point.index <= ansatz.n_params


# --------------------------------------------------------------------------- #
# minima
# --------------------------------------------------------------------------- #
def test_basin_tolerance_is_relative_to_the_loss_scale():
    """An absolute tolerance splits one minimum at -6.1154 into four. A relative one
    does not, and still separates a genuine gap of 0.11."""
    from qmlkit.landscape import MinimaScan

    tight = MinimaScan(losses=np.array([-6.1154, -6.1153, -6.1150]), kinds=(), n_steps=1)
    assert tight.n_basins == 1

    split = MinimaScan(losses=np.array([-6.1154, -6.1153, -6.0012]), kinds=(), n_steps=1)
    assert split.n_basins == 2


def test_convergence_is_unknown_when_endpoints_were_not_classified():
    scan = minima_scan(qk.hardware_efficient(3, 2), n_starts=2, n_steps=40, classify=False)
    assert scan.converged is None


def test_a_scan_that_ran_out_of_steps_reports_itself_unconverged():
    scan = minima_scan(qk.hardware_efficient(4, 2), n_starts=3, n_steps=15, seed=0)
    assert "not-stationary" in scan.kinds
    assert scan.converged is False


def test_disagreement_without_convergence_is_inconclusive_not_spurious_minima():
    """A run still on a slope has not reached a minimum, so its final loss is wherever
    the budget ran out. Calling that a spurious minimum is the false positive this
    module exists to avoid."""
    report = landscape(qk.hardware_efficient(4, 3), n_samples=20, seed=0, n_starts=4, n_steps=15)
    assert report.minima is not None and report.minima.converged is False
    if not report.minima.benign:
        assert "MINIMA_SCAN_INCONCLUSIVE" in report.codes
        assert "SPURIOUS_MINIMA" not in report.codes


def test_a_converged_disagreement_is_reported_as_spurious_minima():
    """The genuine case: every endpoint classifies as a minimum and they differ."""
    rng = np.random.default_rng(11)
    hamiltonian = (
        sum(qk.ZZ(i, (i + 1) % 4) for i in range(4))
        + sum(float(rng.uniform(-1, 1)) * qk.X(i) for i in range(4))
        + sum(float(rng.uniform(-1, 1)) * qk.Z(i) for i in range(4))
    )
    scan = minima_scan(qk.hardware_efficient(4, 1), hamiltonian, n_starts=12, n_steps=500, seed=3)
    assert scan.converged is True, "the fixture only works if every start converged"
    assert not scan.benign
    assert set(scan.kinds) == {"minimum"}


def test_stationarity_is_judged_against_the_curvature_not_an_absolute_cut():
    """The same point must not change verdict because the observable got bigger.

    Scaling an observable by 4 scales the gradient and the Hessian by 4, so an
    absolute threshold on |g| calls one converged and the other not.
    """
    from qmlkit.landscape import HessianSpectrum

    small = HessianSpectrum(eigenvalues=np.array([1.0, 0.5]), gradient_norm=5e-4)
    large = HessianSpectrum(eigenvalues=np.array([4.0, 2.0]), gradient_norm=2e-3)
    assert small.is_stationary == large.is_stationary


def test_the_optimiser_settles_rather_than_orbiting_the_minimum():
    """Fixed-step Adam kept |g| at ~1e-2 while the loss had been stable for 2,600
    steps -- oscillation in a basin. The decayed schedule has to actually land."""
    ansatz = qk.hardware_efficient(4, 2)
    scan = minima_scan(ansatz, qk.Z(0), n_starts=3, n_steps=900, seed=0)
    assert scan.converged is True
    assert set(scan.kinds) == {"minimum"}


def test_a_stable_spread_across_budgets_is_reported_as_real_minima():
    """The use case that found the bug: basins whose spread did not move between 400
    and 5,000 steps are genuine, and must not be hidden behind 'inconclusive'.

    Two layers rather than the three the use case ran: same verdict, wider separation,
    and 9 seconds instead of 58. A test this slow is a cost the whole suite pays.
    """
    fmap = qk.AngleFeatureMap(4, rotation="ry")
    block = EncodingLayer(fmap) + RotationLayer(("ry", "rz")) + EntanglerLayer("cz", "ring")
    ansatz = Ansatz(4, repeat(2, block), name="reupload_4q")
    obs = sum(qk.Z(i) for i in range(4))
    x = np.array([0.4, -0.6, 0.4, -0.6])
    scan = minima_scan(ansatz, obs, x=x, n_starts=4, n_steps=600, seed=0)
    assert scan.converged is True
    assert not scan.benign
    assert scan.n_basins >= 2
    # Comfortably above the basin tolerance, so this is separation and not residual.
    # The magnitude itself depends on x -- 0.260 here, 0.190 at a real moons sample --
    # which is the whole reason minima_scan refuses to pick a point for you.
    assert scan.spread > 20 * scan.tolerance


def test_starts_agree_on_a_landscape_with_one_minimum():
    scan = minima_scan(qk.hardware_efficient(3, 3), qk.Z(0), n_starts=6, n_steps=500, seed=0)
    assert scan.benign
    assert scan.n_basins == 1
    assert scan.reached_best == 1.0


# --------------------------------------------------------------------------- #
# overparametrisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits", [2, 3])
def test_qfim_rank_saturates_at_the_dimension_of_projective_hilbert_space(n_qubits):
    """``2 * 2^n - 2`` is the real dimension of CP^(2^n - 1), and a universal ansatz
    reaches exactly that -- the value is a theoretical maximum, not a fitted one."""
    scan = overparametrisation(
        lambda layers: qk.hardware_efficient(n_qubits, layers), range(1, 2 * n_qubits + 3)
    )
    assert scan.saturated
    assert scan.saturated_rank == 2 * 2**n_qubits - 2


def test_rank_never_exceeds_the_parameter_count():
    scan = overparametrisation(lambda layers: qk.hardware_efficient(3, layers), range(1, 5))
    for params, rank in zip(scan.n_params, scan.ranks, strict=True):
        assert rank <= params


def test_a_scan_that_has_not_saturated_says_so_rather_than_guessing():
    """One width, still climbing: the threshold is above the scan and None is the
    honest answer for it."""
    scan = overparametrisation(lambda layers: qk.hardware_efficient(4, layers), [1])
    assert not scan.saturated
    assert scan.verdict(8) == "unknown"


def test_below_the_threshold_the_ansatz_cannot_reach_the_ground_state():
    """The claim the threshold makes, checked against exact diagonalisation.

    At one layer the rank is 4 of a possible 6 and the best reachable energy is
    -1.65803; at two the rank saturates and the ansatz hits the true -1.77652.
    """
    hamiltonian = 0.8 * qk.ZZ(0, 1) + 0.6 * qk.X(0) + 0.5 * qk.X(1) + 0.3 * qk.Z(0) - 0.4 * qk.Z(1)
    ground = -1.7765214

    scan = overparametrisation(lambda layers: qk.hardware_efficient(2, layers), range(1, 4))
    assert scan.threshold == 8  # two layers

    under = minima_scan(
        qk.hardware_efficient(2, 1), hamiltonian, n_starts=8, n_steps=600, seed=7, classify=False
    )
    over = minima_scan(
        qk.hardware_efficient(2, 2), hamiltonian, n_starts=8, n_steps=600, seed=7, classify=False
    )
    assert under.best > ground + 0.1, "an underparametrised ansatz cannot get there"
    assert over.best == pytest.approx(ground, abs=1e-4), "a saturated one lands on it"


# --------------------------------------------------------------------------- #
# the one call
# --------------------------------------------------------------------------- #
def test_landscape_reports_unmeasurable_rather_than_dead_for_a_live_parameter():
    """The two have different fixes, so conflating them sends readers to the wrong edit.

    Every silent parameter of ``hardware_efficient`` against ``Z(0)`` does change the
    state, so it is unmeasurable and must not be reported as dead.
    """
    report = landscape(qk.hardware_efficient(4, 2), n_samples=40, seed=0, minima=False)
    assert "UNMEASURABLE_PARAMETERS" in report.codes
    assert "DEAD_PARAMETERS" not in report.codes


def test_a_model_with_input_slots_refuses_to_scan_without_a_data_point():
    """Optimising the input slots answers a different question, silently."""
    with pytest.raises(ValueError, match="input slots"):
        minima_scan(_reupload(), qk.Z(0), n_starts=2, n_steps=10)


def test_a_data_point_holds_the_encoding_fixed_while_the_weights_move():
    ansatz = _reupload()
    x = np.array([0.3, -0.7, 0.2])
    scan = minima_scan(ansatz, qk.Z(0), x=x, n_starts=2, n_steps=40, classify=False)
    assert scan.best_theta.size == ansatz.n_params
    assert np.allclose(scan.best_theta[: ansatz.n_inputs], ansatz.angles(x))


def test_curvature_is_judged_on_the_free_coordinates_only():
    """A slot nobody optimised has a nonzero gradient at the optimum, so measuring the
    full vector would call every converged point 'not stationary'."""
    ansatz = _reupload()
    x = np.array([0.3, -0.7, 0.2])
    scan = minima_scan(ansatz, qk.Z(0), x=x, n_starts=2, n_steps=800, classify=False)
    free = range(ansatz.n_inputs, ansatz.n_params)
    restricted = hessian_spectrum(ansatz.build(), scan.best_theta, qk.Z(0), free=free)
    whole = hessian_spectrum(ansatz.build(), scan.best_theta, qk.Z(0))
    assert restricted.is_stationary
    assert restricted.gradient_norm < whole.gradient_norm


def test_landscape_skips_the_scan_rather_than_faking_it():
    report = landscape(_reupload(), n_samples=20, seed=0, n_starts=2, n_steps=20)
    assert "MINIMA_SCAN_SKIPPED" in report.codes
    assert report.minima is None


def test_landscape_runs_the_scan_when_given_a_point():
    report = landscape(
        _reupload(), x=np.array([0.3, -0.7, 0.2]), n_samples=20, seed=0, n_starts=2, n_steps=40
    )
    assert "MINIMA_SCAN_SKIPPED" not in report.codes
    assert report.minima is not None


def test_widening_the_readout_really_does_revive_the_parameters():
    """The finding recommends measuring every wire. Check that it helps, and by how
    much -- 11 of 16 against Z(0), 4 of 16 against the sum."""
    ansatz = qk.hardware_efficient(4, 2)
    narrow = gradient_stats(ansatz, qk.Z(0), n_samples=40, seed=0).silent.size
    wide = gradient_stats(ansatz, sum(qk.Z(i) for i in range(4)), n_samples=40, seed=0).silent.size
    assert narrow == 11
    assert wide == 4


def test_a_narrow_readout_is_reported_as_the_cause_not_as_an_ansatz_defect():
    """Against Z(0) most parameters of any stock ansatz are unmeasurable. That is the
    readout, so it is info and names the readout -- a warning here would fire on
    everything and teach people to skip the section."""
    report = landscape(qk.hardware_efficient(4, 2), n_samples=40, seed=0, minima=False)
    (finding,) = [f for f in report.findings if f.code == "UNMEASURABLE_PARAMETERS"]
    assert finding.severity == "info"
    assert "watches 1 of 4 wires" in finding.message


def test_a_full_readout_that_still_cannot_see_them_is_a_warning():
    """A trailing Rz layer before a Z readout is the circuit's fault, not the
    observable's, and is reported as such."""
    block = RotationLayer("ry") + EntanglerLayer("cz", "ring") + RotationLayer("rz")
    ansatz = Ansatz(3, repeat(2, block), name="trailing_rz")
    report = landscape(ansatz, sum(qk.Z(i) for i in range(3)), n_samples=40, seed=0, minima=False)
    (finding,) = [f for f in report.findings if f.code == "UNMEASURABLE_PARAMETERS"]
    assert finding.severity == "warning"
    assert "every wire" in finding.message


def test_landscape_runs_on_a_reuploading_model_with_input_slots():
    report = landscape(_reupload(), n_samples=30, seed=0, minima=False)
    assert report.gradients.n_params == _reupload().n_params
    assert report.gradients.best_variance > 0


def test_landscape_str_mentions_every_section():
    text = str(landscape(qk.hardware_efficient(3, 2), n_samples=20, n_starts=2, n_steps=60))
    assert "gradient statistics" in text
    assert "random starts" in text


# --------------------------------------------------------------------------- #
# the Fisher information fix
# --------------------------------------------------------------------------- #
def test_fisher_information_depends_on_the_data():
    """It averaged over rows it never read, so every dataset gave the same rank-1 matrix."""
    ansatz = _reupload()
    weights = np.random.default_rng(0).uniform(-np.pi, np.pi, ansatz.n_weights)
    a = fisher_information(ansatz, np.random.default_rng(1).uniform(-1, 1, (30, 3)), weights)
    b = fisher_information(ansatz, np.random.default_rng(2).uniform(-1, 1, (30, 3)), weights)
    assert not np.allclose(a, b)


def test_fisher_information_gains_rank_from_more_data():
    ansatz = _reupload()
    weights = np.random.default_rng(0).uniform(-np.pi, np.pi, ansatz.n_weights)
    rows = np.random.default_rng(3).uniform(-1, 1, (60, 3))
    one = np.linalg.matrix_rank(fisher_information(ansatz, rows[:1], weights))
    many = np.linalg.matrix_rank(fisher_information(ansatz, rows, weights))
    assert one == 1, "a single input contributes a single outer product"
    assert many > one


def test_fisher_information_is_symmetric_positive_semidefinite():
    ansatz = _reupload()
    weights = np.random.default_rng(0).uniform(-np.pi, np.pi, ansatz.n_weights)
    f = fisher_information(ansatz, np.random.default_rng(4).uniform(-1, 1, (20, 3)), weights)
    assert np.allclose(f, f.T)
    assert np.all(np.linalg.eigvalsh(f) > -1e-10)


def test_effective_dimension_stays_within_the_parameter_count():
    ansatz = _reupload()
    weights = np.random.default_rng(0).uniform(-np.pi, np.pi, ansatz.n_weights)
    f = fisher_information(ansatz, np.random.default_rng(5).uniform(-1, 1, (40, 3)), weights)
    assert 0.0 < effective_dimension(f) <= ansatz.n_weights


def test_fisher_information_refuses_an_ansatz_the_data_cannot_reach():
    """No input slots means X changes nothing, and a silently rank-1 matrix is worse
    than a refusal that says why."""
    ansatz = qk.hardware_efficient(3, 2)
    weights = np.zeros(ansatz.n_params)
    with pytest.raises(ValueError, match="no input slots"):
        fisher_information(ansatz, np.zeros((5, 3)), weights)


def test_fisher_information_rejects_a_full_length_parameter_vector():
    """`theta` is the weights; passing inputs too is the easy mistake to make."""
    ansatz = _reupload()
    with pytest.raises(ValueError, match="weight"):
        fisher_information(ansatz, np.zeros((5, 3)), np.zeros(ansatz.n_params))
