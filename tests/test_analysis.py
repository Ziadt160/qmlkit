"""Metrics, Fourier analysis, the QML-specific optimisers, quantum info, datasets, drawing."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.fourier import (
    dominant_frequency,
    fourier_coefficients,
    model_spectrum,
    reachable_frequencies,
    reconstruct,
    spectrum,
)
from qmlkit.info import (
    bloch_vector,
    concurrence,
    density_matrix,
    mutual_info,
    purity,
    reduced_dm,
    state_fidelity,
    vn_entropy,
)
from qmlkit.metrics import (
    AnsatzReport,
    barren_plateau_scan,
    compare_ansatze,
    effective_dimension,
    entangling_capability,
    expressibility,
    generalization_bound,
    gradient_variance,
    haar_fidelity_pdf,
    meyer_wallach,
    noise_survival,
    samples_for_gap,
)
from qmlkit.optim import (
    metric_tensor,
    minimize_qng,
    minimize_rotosolve,
    quantum_fisher_information,
    rotosolve_step,
)


def _bell() -> qk.CircuitSpec:
    qc = qk.QCircuit(2)
    qc.h(0).cx(0, 1)
    return qc.to_spec()


# --------------------------------------------------------------------------- #
# quantum information
# --------------------------------------------------------------------------- #
def test_bell_state_is_maximally_entangled():
    bell = _bell()
    assert purity(bell, [0]) == pytest.approx(0.5, abs=1e-12)
    assert vn_entropy(bell, [0], base=2) == pytest.approx(1.0, abs=1e-12)
    assert concurrence(bell) == pytest.approx(1.0, abs=1e-12)
    assert mutual_info(bell, [0], [1]) == pytest.approx(2 * np.log(2), abs=1e-12)


def test_product_state_has_no_entanglement():
    spec = qk.angle_encode([0.4, 1.1])
    assert purity(spec, [0]) == pytest.approx(1.0, abs=1e-12)
    assert vn_entropy(spec, [0]) == pytest.approx(0.0, abs=1e-10)
    assert concurrence(spec) == pytest.approx(0.0, abs=1e-10)


def test_reduced_dm_shape_and_trace():
    bell = _bell()
    rho = reduced_dm(bell, [0])
    assert rho.shape == (2, 2)
    assert np.trace(rho) == pytest.approx(1.0)
    assert reduced_dm(bell, [0, 1]).shape == (4, 4)


def test_reduced_dm_validates_wires():
    with pytest.raises(ValueError, match="out of range"):
        reduced_dm(_bell(), [5])


def test_bloch_vector_of_basis_and_superposition():
    assert np.allclose(bloch_vector(qk.QCircuit(1).to_spec()), [0, 0, 1], atol=1e-12)
    qc = qk.QCircuit(1)
    qc.h(0)
    assert np.allclose(bloch_vector(qc.to_spec()), [1, 0, 0], atol=1e-12)


def test_state_fidelity_and_density_matrix():
    a = qk.angle_encode([0.5])
    assert state_fidelity(a, a) == pytest.approx(1.0)
    rho = density_matrix(a)
    assert np.trace(rho) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="different widths"):
        state_fidelity(a, qk.angle_encode([0.5, 0.5]))


def test_concurrence_is_two_qubit_only():
    with pytest.raises(ValueError, match="two qubits"):
        concurrence(qk.angle_encode([0.1, 0.2, 0.3]))


def test_mutual_info_rejects_overlapping_subsystems():
    with pytest.raises(ValueError, match="overlap"):
        mutual_info(_bell(), [0], [0])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_meyer_wallach_endpoints():
    assert meyer_wallach(qk.statevector(qk.angle_encode([0.4, 1.1]))) == pytest.approx(
        0.0, abs=1e-10
    )
    assert meyer_wallach(qk.statevector(_bell())) == pytest.approx(1.0, abs=1e-10)


def test_haar_pdf_integrates_to_one():
    # plain Riemann sum rather than np.trapezoid, which only exists on NumPy 2.x --
    # SpinQit pins numpy<2, so the suite has to run on both
    f = np.linspace(0, 1, 200001)
    dx = f[1] - f[0]
    for n in (1, 2, 3):
        assert float(np.sum(haar_fidelity_pdf(f, n)) * dx) == pytest.approx(1.0, abs=1e-3)


def test_deeper_ansatz_is_more_expressive():
    """Lower KL from Haar means more of state space is reachable."""
    shallow = expressibility(qk.hardware_efficient(3, 1), n_samples=600, seed=0)
    deep = expressibility(qk.hardware_efficient(3, 4), n_samples=600, seed=0)
    assert deep < shallow


def test_entangling_capability_ordering():
    """An ansatz with no entanglers cannot entangle."""
    none = qk.Ansatz(3, qk.RotationLayer(("ry", "rz")))
    some = qk.hardware_efficient(3, 2)
    assert entangling_capability(none, n_samples=40, seed=0) == pytest.approx(0.0, abs=1e-9)
    assert entangling_capability(some, n_samples=40, seed=0) > 0.3


def test_gradient_variance_is_positive_and_finite():
    v = gradient_variance(qk.hardware_efficient(3, 2), n_samples=40, seed=0)
    assert v > 0 and np.isfinite(v)


def test_global_cost_plateaus_where_a_local_one_does_not():
    """At fixed shallow depth, cost locality is what decides trainability."""

    def global_cost(n):
        return qk.PauliString(tuple((q, "Z") for q in range(n)))

    local = barren_plateau_scan(
        lambda n: qk.hardware_efficient(n, 2), [2, 4, 6], n_samples=40, seed=0
    )
    glob = barren_plateau_scan(
        lambda n: qk.hardware_efficient(n, 2),
        [2, 4, 6],
        obs_factory=global_cost,
        n_samples=40,
        seed=0,
    )
    assert glob["decay_per_qubit"] < local["decay_per_qubit"]
    assert glob["looks_exponential"]


def test_generalization_bound_behaviour():
    assert generalization_bound(50, 1000) > generalization_bound(10, 1000)
    assert generalization_bound(50, 10000) < generalization_bound(50, 1000)
    assert generalization_bound(50, 1000, with_log=False) < generalization_bound(50, 1000)
    assert samples_for_gap(10, 0.1) > samples_for_gap(10, 0.5)
    with pytest.raises(ValueError, match="n_samples must be positive"):
        generalization_bound(10, 0)
    with pytest.raises(ValueError, match="gap must be positive"):
        samples_for_gap(10, 0.0)


def test_effective_dimension_is_bounded_by_the_parameter_count():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6))
    fisher = a @ a.T
    d = effective_dimension(fisher, n_samples=1000)
    assert 0.0 <= d <= 6.0
    assert effective_dimension(np.zeros((4, 4))) == 0.0


def test_noise_survival_compounds():
    assert noise_survival(0.99, 100) == pytest.approx(0.99**100)


def test_ansatz_report_and_comparison():
    report = AnsatzReport(qk.hardware_efficient(2, 1), n_samples=120, seed=0)
    for key in ("expressibility", "entangling_capability", "gradient_variance", "depth"):
        assert key in report.results
    assert "hardware_efficient" in str(report)
    rows = compare_ansatze([qk.hardware_efficient(2, 1), qk.mps_ansatz(2)], n_samples=80)
    assert len(rows) == 2 and rows[0]["name"] != rows[1]["name"]


# --------------------------------------------------------------------------- #
# Fourier
# --------------------------------------------------------------------------- #
def test_coefficients_recover_a_known_series():
    def f(x):
        return 0.3 + 0.8 * np.cos(x) - 0.5 * np.sin(2 * x)

    c = fourier_coefficients(f, degree=3)
    xs = np.linspace(0, 2 * np.pi, 11)
    assert np.allclose(reconstruct(c, xs), [f(x) for x in xs], atol=1e-10)
    s = spectrum(f, 4)
    assert s[0] == pytest.approx(0.3, abs=1e-9)
    assert s[1] == pytest.approx(0.8, abs=1e-9)
    assert s[2] == pytest.approx(0.5, abs=1e-9)
    assert 3 not in s
    assert dominant_frequency(f) == 1


def test_fourier_validates_its_grid():
    with pytest.raises(ValueError, match="degree must be non-negative"):
        fourier_coefficients(np.cos, degree=-1)
    with pytest.raises(ValueError, match="cannot resolve"):
        fourier_coefficients(np.cos, degree=4, n_samples=5)


def test_reachable_frequencies():
    assert reachable_frequencies(3) == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="cannot be negative"):
        reachable_frequencies(-1)


def test_re_uploading_buys_the_spectrum_it_claims():
    """L uploads reach frequencies 0..L — with a non-commuting trainable block."""
    rng = np.random.default_rng(0)
    for L in (2, 3):
        enc = qk.DataReuploadEncoder(
            1, n_uploads=L, rotations=("rz", "ry", "rz"), entanglement=None
        )
        theta = rng.uniform(-np.pi, np.pi, enc.n_weights)
        present = set(model_spectrum(enc, theta, degree=L + 3))
        assert present <= set(range(L + 1)), f"reached beyond frequency {L}: {present}"
        assert L in present, f"L={L} uploads did not reach frequency {L}"


def test_a_commuting_block_collapses_the_spectrum_and_warns():
    """Ry(x)Ry(t)Ry(x)... = Ry(Lx + sum t): one frequency, and the weights do nothing."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enc = qk.DataReuploadEncoder(1, n_uploads=3, rotations=("ry",), entanglement=None)
    assert caught, "a commuting trainable block must warn"
    assert "commutes" in str(caught[0].message)

    theta = np.random.default_rng(0).uniform(-np.pi, np.pi, enc.n_weights)
    assert set(model_spectrum(enc, theta, degree=6)) == {3}


# --------------------------------------------------------------------------- #
# optimisers
# --------------------------------------------------------------------------- #
def test_rotosolve_needs_no_learning_rate_and_reaches_the_minimum():
    a = qk.hardware_efficient(3, 2)
    spec = a.build()
    obs = qk.Z(0) + qk.Z(1) + qk.Z(2)

    def loss(t):
        return qk.expval(spec, obs, theta=t)

    _, history = minimize_rotosolve(loss, a.init("uniform", seed=1), n_sweeps=10)
    assert history[-1] < history[0]
    assert history[-1] < -2.9  # the minimum is -3


def test_rotosolve_step_is_monotone_on_a_single_rotation():
    qc = qk.QCircuit(1)
    qc.ry(0, qk.ParamRef(0))
    spec = qc.to_spec()

    def loss(t):
        return qk.expval(spec, qk.Z(0), theta=t)

    best = rotosolve_step(loss, np.array([0.3]))
    assert loss(best) == pytest.approx(-1.0, abs=1e-9)  # exact minimum of cos


def test_metric_tensor_is_symmetric_and_psd():
    a = qk.hardware_efficient(3, 1)
    g = metric_tensor(a.build(), a.init(seed=0))
    assert g.shape == (a.n_params, a.n_params)
    assert np.allclose(g, g.T, atol=1e-6)
    assert np.linalg.eigvalsh(g).min() > -1e-6


def test_quantum_fisher_is_four_times_the_metric():
    a = qk.hardware_efficient(2, 1)
    theta = a.init(seed=0)
    assert np.allclose(
        quantum_fisher_information(a.build(), theta), 4 * metric_tensor(a.build(), theta, None)
    )


def test_diag_approximation_keeps_only_the_diagonal():
    a = qk.hardware_efficient(2, 1)
    g = metric_tensor(a.build(), a.init(seed=0), approx="diag")
    assert np.allclose(g, np.diag(np.diag(g)))
    with pytest.raises(ValueError, match="unknown approx"):
        metric_tensor(a.build(), a.init(seed=0), approx="sideways")


def test_qng_converges_at_least_as_well_as_plain_gradient_descent():
    a = qk.hardware_efficient(3, 2)
    spec, theta0 = a.build(), a.init("uniform", seed=1)
    obs = qk.Z(0) + qk.Z(1) + qk.Z(2)

    _, qng_history = minimize_qng(spec, theta0, obs, n_steps=25, lr=0.15)

    theta = theta0.copy()
    for _ in range(25):
        theta = theta - 0.15 * qk.grad(spec, theta, obs)
    plain = qk.expval(spec, obs, theta=theta)

    assert qng_history[-1] <= plain + 1e-9


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def test_ad_hoc_data_is_labelled_by_a_quantum_witness():
    X, y = qk.datasets.ad_hoc_data(n_samples=20, n_features=2, gap=0.4, seed=0)
    assert X.shape == (20, 2)
    assert set(np.unique(y)) == {0, 1}


def test_bars_and_stripes_patterns():
    bas = qk.datasets.bars_and_stripes(2)
    assert bas.shape[1] == 4
    strings = {"".join(map(str, row)) for row in bas}
    assert {"0000", "1111", "0011", "1100", "0101", "1010"} == strings
    with pytest.raises(ValueError, match="size must be at least 1"):
        qk.datasets.bars_and_stripes(0)


@pytest.mark.parametrize(
    "fn", [qk.datasets.make_moons, qk.datasets.make_circles, qk.datasets.make_blobs]
)
def test_toy_datasets_are_angle_scaled(fn):
    X, y = fn(n_samples=30, seed=0)
    assert X.shape[0] == 30
    assert X.min() >= 0.0 and X.max() <= np.pi + 1e-9
    assert len(np.unique(y)) >= 2


def test_parity_dataset_labels_are_parity():
    X, y = qk.datasets.make_parity(20, 4, seed=0)
    bits = (X > 1).astype(int)
    assert np.array_equal(y, bits.sum(axis=1) % 2)


def test_train_test_split():
    X, y = qk.datasets.make_blobs(n_samples=20, seed=0)
    a, b, ya, yb = qk.datasets.train_test_split(X, y, test_size=0.25, seed=0)
    assert len(a) == 15 and len(b) == 5
    assert len(ya) == 15 and len(yb) == 5
    with pytest.raises(ValueError, match="test_size must be"):
        qk.datasets.train_test_split(X, y, test_size=1.5)


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def test_draw_shows_every_wire_and_gate():
    text = qk.draw(qk.hardware_efficient(3, 1).build())
    assert text.count("\n") == 2  # three wires
    for token in ("q0:", "q1:", "q2:", "RY", "RZ"):
        assert token in text


def test_draw_marks_two_qubit_gates():
    qc = qk.QCircuit(2)
    qc.h(0).cx(0, 1)
    text = qk.draw(qc.to_spec())
    assert "@" in text and "X" in text


def test_draw_truncates_a_very_wide_circuit():
    text = qk.draw(qk.hardware_efficient(2, 40).build(), max_width=60)
    assert all(len(line) <= 60 for line in text.split("\n"))
    assert "..." in text


def test_specs_reports_tied_parameters():
    s = qk.specs(qk.qcnn_ansatz(8).build())
    assert s["weight_tied_parameters"] > 0
    assert s["grad_passes_adjoint"] == 1
    assert s["grad_circuits_parameter_shift"] > 1
