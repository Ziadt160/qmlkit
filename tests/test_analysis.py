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


# --------------------------------------------------------------------------- #
# A dead parameter is not a barren plateau
#
# `gradient_variance` probes ONE parameter, and theta_0 on several stock ansaetze is
# a leading Rz on |0>, whose gradient against Z is identically zero. AnsatzReport
# printed 1.4e-32 under "higher = more trainable", which reads as a catastrophic
# plateau and is really "you probed a parameter that does nothing" - while
# diagnose() called the same ansatz DEAD_WEIGHTS. Two of this library's own tools
# contradicting each other is the worst place for it to happen.
# --------------------------------------------------------------------------- #
def test_a_machine_zero_gradient_variance_warns_that_it_is_not_a_plateau():
    ansatz = qk.strongly_entangling(4, 3)
    with pytest.warns(UserWarning, match="machine zero rather than a small number"):
        value = qk.metrics.gradient_variance(ansatz, qk.Z(0), n_samples=10, seed=0)
    assert value < 1e-28
    # the number itself is unchanged - only the silence around it was the bug
    assert value >= 0.0


def test_a_genuinely_small_variance_does_not_warn():
    """The warning must not fire on every deep circuit, or it means nothing."""
    ansatz = qk.hardware_efficient(3, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        value = qk.metrics.gradient_variance(ansatz, qk.Z(0), n_samples=10, seed=0)
    assert value > 1e-28


def test_ansatz_report_probes_a_live_parameter():
    """The reported number has to be about the ansatz, not about index 0."""
    ansatz = qk.strongly_entangling(4, 3)
    report = qk.metrics.AnsatzReport(ansatz, n_samples=60, seed=0)
    index = report["gradient_param_index"]
    assert index != 0, "theta_0 is dead on this ansatz, so the report must move off it"
    assert float(report["gradient_variance"]) > 1e-28
    assert f"parameter {index}" in str(report)


def test_ansatz_report_agrees_with_diagnose_about_dead_weights():
    """Both tools look at the same ansatz; they must not contradict each other."""
    ansatz = qk.strongly_entangling(4, 3)
    report = qk.metrics.AnsatzReport(ansatz, n_samples=60, seed=0)
    findings = qk.diagnose(ansatz, seed=0, n_samples=10, probes=1)
    assert "DEAD_WEIGHTS" in findings.codes
    # diagnose says some weights are dead; the report no longer reads as though the
    # whole ansatz were untrainable because it happened to land on one of them
    assert float(report["gradient_variance"]) > 1e-28


def test_first_live_parameter_falls_back_when_everything_is_dead():
    from qmlkit.metrics import _first_live_parameter

    ansatz = qk.Ansatz(1, qk.RotationLayer("rz"))  # Rz on |0> moves nothing measurable
    assert _first_live_parameter(ansatz) == 0


# --------------------------------------------------------------------------- #
# Printing a circuit on a console that cannot encode box-drawing glyphs
#
# `print(qk.draw(spec))` used to raise UnicodeEncodeError on a Windows console,
# whose default code page is cp1252. It happened after the model had already
# trained, on the first thing a new user does, with a traceback naming the codec
# rather than qk.draw.
# --------------------------------------------------------------------------- #
class _Cp1252Stream:
    """Stands in for a Windows console: reports an encoding that lacks the glyphs."""

    encoding = "cp1252"


def _bound(n_qubits=3, layers=1):
    ansatz = qk.hardware_efficient(n_qubits, layers)
    return ansatz.build(np.linspace(0.1, 1.2, ansatz.n_params))


def test_draw_degrades_when_stdout_cannot_encode_the_glyphs(monkeypatch):
    spec = _bound()
    monkeypatch.setattr("sys.stdout", _Cp1252Stream())
    diagram = qk.draw(spec)
    diagram.encode("cp1252")  # the whole point: this must not raise
    assert "─" not in diagram
    assert "-" in diagram


def test_draw_keeps_the_glyphs_on_a_utf8_stream(monkeypatch):
    class Utf8Stream:
        encoding = "utf-8"

    monkeypatch.setattr("sys.stdout", Utf8Stream())
    assert "─" in qk.draw(_bound())


def test_ascii_overrides_the_detection_both_ways():
    spec = _bound()
    assert "─" not in qk.draw(spec, ascii=True)
    assert "─" in qk.draw(spec, ascii=False)


def test_the_ascii_fallback_lines_up_exactly():
    """Same column widths, or the fallback is unreadable rather than merely plain."""
    spec = _bound(3, 2)
    unicode_lines = qk.draw(spec, ascii=False).splitlines()
    ascii_lines = qk.draw(spec, ascii=True).splitlines()
    assert [len(line) for line in unicode_lines] == [len(line) for line in ascii_lines]


def test_a_dagger_gate_survives_the_fallback():
    from qmlkit.core.builder import QCircuit

    qc = QCircuit(1)
    qc.sdg(0)
    plain = qk.draw(qc.to_spec(), ascii=True)
    plain.encode("cp1252")
    assert "†" not in plain


def test_probabilities_bar_degrades_too(monkeypatch):
    """The histogram's block glyph dies on cp1252 just as loudly as the wires."""
    from qmlkit.draw import probabilities_bar

    spec = _bound()
    monkeypatch.setattr("sys.stdout", _Cp1252Stream())
    bars = probabilities_bar(qk.probabilities(spec), 3, top=3)
    bars.encode("cp1252")
    assert "█" not in bars
    assert "#" in bars


def test_detection_survives_a_stream_with_no_encoding(monkeypatch):
    """A redirected or wrapped stdout may report nothing; assume the narrow case."""

    class NoEncoding:
        encoding = None

    monkeypatch.setattr("sys.stdout", NoEncoding())
    qk.draw(_bound()).encode("cp1252")


# --------------------------------------------------------------------------- #
# ENCODING_COMMUTES claimed a Fourier fact it had only inferred
#
# `Ry(x) Ry(t) Ry(x) Ry(t)` collapses on ONE WIRE WITH NOTHING IN BETWEEN. Any
# entanglement breaks it, and the check was purely structural - it could see neither
# an entangler in the trainable block nor an entangling feature map. It reported an
# *error* on architectures whose measured band was 0..2L. A false positive in the
# honesty layer is worse than a false negative: it teaches people to ignore the tool.
# --------------------------------------------------------------------------- #
def _reupload(fmap, block, n_layers=2):
    return qk.reupload(fmap, n_layers=n_layers, block=block)


def _live_frequencies(model, n_inputs, degree=6):
    """The frequencies the model actually reaches, measured rather than inferred."""
    weights = model.init(seed=0)

    def response(x):
        return qk.expval(model.bind(np.full(n_inputs, x), weights), qk.Z(0))

    return sorted(k for k, power in qk.fourier.spectrum(response, degree).items() if power > 1e-6)


def _flags_collapse(model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return "ENCODING_COMMUTES" in qk.diagnose(model, seed=0, n_samples=6, probes=1).codes


def test_a_genuine_collapse_is_still_caught():
    """One wire, matching rotations, nothing in between: the identity really holds."""
    model = _reupload(qk.AngleFeatureMap(1, rotation="ry", entangle=False), qk.RotationLayer("ry"))
    assert len(_live_frequencies(model, 1)) == 1
    assert _flags_collapse(model)


def test_a_collapse_across_unentangled_wires_is_still_caught():
    model = _reupload(qk.AngleFeatureMap(2, rotation="ry", entangle=False), qk.RotationLayer("ry"))
    assert len(_live_frequencies(model, 2)) == 1
    assert _flags_collapse(model)


def test_an_entangler_in_the_block_defeats_the_collapse_and_is_not_reported():
    model = _reupload(
        qk.AngleFeatureMap(2, rotation="ry", entangle=False),
        qk.RotationLayer("ry") + qk.EntanglerLayer("cx", "ring"),
    )
    assert len(_live_frequencies(model, 2)) > 1, "the entangler must break the collapse"
    assert not _flags_collapse(model)


def test_an_entangling_feature_map_also_defeats_it():
    """The structural check could not see this one at all - the entanglement is
    upstream of the trainable block, in the encoding itself."""
    model = _reupload(qk.AngleFeatureMap(2, rotation="ry"), qk.RotationLayer("ry"))
    assert len(_live_frequencies(model, 2)) > 1
    assert not _flags_collapse(model)


def test_a_non_commuting_block_is_never_reported():
    model = _reupload(
        qk.AngleFeatureMap(2, rotation="ry", entangle=False),
        qk.RotationLayer(("rz", "ry", "rz")),
    )
    assert not _flags_collapse(model)


def test_the_finding_agrees_with_the_measurement_it_asserts():
    """The property that was actually violated: report if and only if it is true."""
    cases = [
        (qk.AngleFeatureMap(1, rotation="ry", entangle=False), qk.RotationLayer("ry"), 1),
        (qk.AngleFeatureMap(2, rotation="ry", entangle=False), qk.RotationLayer("ry"), 2),
        (qk.AngleFeatureMap(2, rotation="ry"), qk.RotationLayer("ry"), 2),
        (
            qk.AngleFeatureMap(2, rotation="ry", entangle=False),
            qk.RotationLayer("ry") + qk.EntanglerLayer("cx", "ring"),
            2,
        ),
    ]
    for fmap, block, n_inputs in cases:
        model = _reupload(fmap, block)
        collapsed = len(_live_frequencies(model, n_inputs)) == 1
        assert _flags_collapse(model) == collapsed


# --------------------------------------------------------------------------- #
# Two smaller ones from the same session
# --------------------------------------------------------------------------- #
def test_list_baselines_does_not_repeat_a_name_registered_for_both_tasks():
    names = qk.list_baselines()
    assert len(names) == len(set(names))
    # the name that exposed it is registered for classification and regression alike
    assert "rbf-kernel-ridge" in qk.list_baselines("classification")
    assert "rbf-kernel-ridge" in qk.list_baselines("regression")


def test_a_near_miss_metric_key_is_answered_not_silently_dropped():
    """`get("precision")` returning None turned a typo into a numpy TypeError later."""
    scores = qk.evaluate.classification(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]))
    with pytest.raises(KeyError, match="precision_macro"):
        scores.get("precision")


def test_an_explicit_default_is_still_honoured():
    """The caller has said what they want; do not second-guess them."""
    scores = qk.evaluate.classification(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]))
    assert scores.get("precision", 0.0) == 0.0
    assert scores.get("nonsense", None) is None


def test_a_real_key_still_reads_normally():
    scores = qk.evaluate.classification(np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]))
    assert scores.get("precision_macro") == scores["precision_macro"]


# --------------------------------------------------------------------------- #
# Adam
#
# The gap that mattered: the circuit-level optimisers were gradient-descent,
# rotosolve and spsa, so anyone training a variational model outside the torch
# bridge hand-rolled their own -- and a hand-rolled optimiser that diverges looks
# exactly like a method that does not work.
# --------------------------------------------------------------------------- #
def test_one_step_matches_the_closed_form():
    """Adam is four lines of algebra; assert them rather than trusting the loop."""
    from qmlkit.optim import AdamState, adam_step

    theta = np.array([1.0, 2.0, 3.0])
    gradient = np.array([0.1, -0.2, 0.3])
    lr, b1, b2, eps = 0.1, 0.9, 0.999, 1e-8

    stepped, _ = adam_step(theta, gradient, AdamState.for_parameters(3), lr, b1, b2, eps)

    m = (1 - b1) * gradient
    v = (1 - b2) * gradient**2
    expected = theta - lr * (m / (1 - b1)) / (np.sqrt(v / (1 - b2)) + eps)
    assert stepped == pytest.approx(expected)


def test_the_first_step_is_bias_corrected():
    """Without correction the first steps are damped by ~(1 - beta) and Adam looks
    like it is barely moving."""
    from qmlkit.optim import AdamState, adam_step

    gradient = np.array([1.0])
    stepped, _ = adam_step(np.array([0.0]), gradient, AdamState.for_parameters(1), lr=0.1)
    # corrected, the first step is almost exactly -lr * sign(grad)
    assert stepped[0] == pytest.approx(-0.1, abs=1e-6)


def test_the_state_is_not_mutated():
    """Callers driving one step at a time keep trajectories; mutation loses them."""
    from qmlkit.optim import AdamState, adam_step

    state = AdamState.for_parameters(2)
    before = (state.m.copy(), state.v.copy(), state.t)
    adam_step(np.zeros(2), np.ones(2), state)
    assert state.t == before[2]
    assert state.m == pytest.approx(before[0])
    assert state.v == pytest.approx(before[1])


def test_the_step_counter_advances_with_the_returned_state():
    from qmlkit.optim import AdamState, adam_step

    state = AdamState.for_parameters(2)
    for expected_t in (1, 2, 3):
        _, state = adam_step(np.zeros(2), np.ones(2), state)
        assert state.t == expected_t


def test_a_mismatched_gradient_is_refused_by_name():
    from qmlkit.optim import AdamState, adam_step

    with pytest.raises(ValueError, match="3 entries but there are 2 parameters"):
        adam_step(np.zeros(2), np.ones(3), AdamState.for_parameters(2))


def test_adam_beats_plain_descent_when_the_scales_differ():
    """The property Adam exists for, on a landscape where it is unambiguous.

    A quadratic whose curvature spans four orders of magnitude: one learning rate
    either crawls on the flat direction or diverges on the steep one. Dividing by the
    running gradient magnitude makes the step per-parameter, which is the fix.
    """
    from qmlkit.optim import minimize_adam

    curvature = np.array([1.0, 1e-4])

    def loss(t):
        return float(np.sum(curvature * t**2))

    def grad(t):
        return 2 * curvature * np.asarray(t, dtype=float)

    start = np.array([1.0, 1.0])
    _, adam_history = minimize_adam(loss, start, grad, n_steps=200, lr=0.05)

    theta = start.copy()
    for _ in range(200):
        theta = theta - 0.05 * grad(theta)
    descent_final = loss(theta)

    assert adam_history[-1] < descent_final, "Adam should win where the scales differ"
    assert adam_history[-1] < adam_history[0]


def test_minimize_adam_lowers_a_real_expectation_value():
    from qmlkit.optim import minimize_adam

    ansatz = qk.hardware_efficient(3, 2)
    spec, obs = ansatz.build(), qk.Z(0)
    theta0 = ansatz.init("small", seed=0)
    _, history = minimize_adam(
        lambda t: qk.expval(spec, obs, theta=t),
        theta0,
        lambda t: qk.grad(spec, t, obs),
        n_steps=30,
    )
    assert history[-1] < history[0]


def test_the_callback_and_tolerance_are_honoured():
    from qmlkit.optim import minimize_adam

    seen = []
    _, history = minimize_adam(
        lambda t: float(np.sum(np.asarray(t) ** 2)),
        np.array([1.0]),
        lambda t: 2 * np.asarray(t, dtype=float),
        n_steps=50,
        callback=lambda step, theta, value: seen.append(step),
    )
    assert seen == list(range(50))
    assert len(history) == 51

    _, stopped = minimize_adam(
        lambda t: float(np.sum(np.asarray(t) ** 2)),
        np.array([1.0]),
        lambda t: 2 * np.asarray(t, dtype=float),
        n_steps=500,
        tol=1e-6,
    )
    assert len(stopped) < 501, "a tolerance should stop before the full budget"


def test_adam_is_reachable_by_name_like_every_other_optimiser():
    from qmlkit.algorithms.vqe import OPTIMIZERS

    assert "adam" in OPTIMIZERS
    loss = lambda t: float(np.sum(np.asarray(t) ** 2))  # noqa: E731
    grad = lambda t: 2 * np.asarray(t, dtype=float)  # noqa: E731
    theta, history = OPTIMIZERS["adam"](loss, np.array([1.0, -1.0]), grad=grad, n_steps=40)
    assert history[-1] < history[0]
    assert theta.shape == (2,)


# --------------------------------------------------------------------------- #
# Shot-noise error bars for an observable with more than one term
#
# `sqrt((1 - z^2)/shots)` is the single-Pauli formula. Applied to a sum it reported
# exactly 0.00000 once |<O>| reached 1 -- so every molecular Hamiltonian came back
# with an error bar that looked converged and did not move with the shot count.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "observable",
    [
        qk.Z(0),
        2.0 * qk.Z(0),
        qk.Z(0) + qk.Z(1),
        qk.Z(0) + 0.5 * qk.ZZ(0, 1),
        qk.Z(0) + qk.X(1),
    ],
    ids=["single", "scaled", "sum", "weighted-sum", "two-bases"],
)
def test_the_reported_error_bar_matches_the_empirical_spread(observable):
    """The only check that matters: does it predict how much the number moves?"""
    spec = qk.angle_encode([0.7, 0.3])
    shots = 2000
    _, reported = qk.expectation(spec, observable, shots=shots, seed=0, return_std=True)
    draws = [qk.expectation(spec, observable, shots=shots, seed=s) for s in range(200)]
    empirical = float(np.std(draws))
    assert reported == pytest.approx(empirical, rel=0.20)


def test_a_multi_term_error_bar_is_not_zero():
    """The specific failure: |<O>| >= 1 drove the old formula's variance negative."""
    spec = qk.angle_encode([0.05, 0.05])  # both Z near +1, so <Z0+Z1> is near 2
    value, reported = qk.expectation(spec, qk.Z(0) + qk.Z(1), shots=4000, seed=0, return_std=True)
    assert abs(value) > 1.0, "the case only arises once the expectation exceeds one"
    assert reported > 0.0


def test_the_error_bar_shrinks_as_one_over_root_shots():
    spec = qk.angle_encode([0.7, 0.3])
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 1)
    _, few = qk.expectation(spec, obs, shots=1000, seed=0, return_std=True)
    _, many = qk.expectation(spec, obs, shots=100_000, seed=0, return_std=True)
    assert few / many == pytest.approx(10.0, rel=0.05)


def test_exact_mode_still_reports_no_error():
    spec = qk.angle_encode([0.7, 0.3])
    value, err = qk.expectation(spec, qk.Z(0) + qk.Z(1), return_std=True)
    assert err == 0.0
    assert value == pytest.approx(qk.expectation(spec, qk.Z(0) + qk.Z(1)))


# --------------------------------------------------------------------------- #
# purity was told which backend to use and ignored it
# --------------------------------------------------------------------------- #
def test_purity_on_a_mixed_state_backend_is_not_hard_coded_to_one():
    cirq = pytest.importorskip("cirq")
    spec = qk.angle_encode([0.7, 0.3])
    backend = qk.get_backend("cirq-density", noise=cirq.depolarize(0.5))
    assert qk.purity(spec, backend=backend) == pytest.approx(backend.purity(spec))
    assert qk.purity(spec, backend=backend) < 0.9, "a heavily depolarised state is mixed"


def test_purity_of_a_statevector_is_still_one():
    assert qk.purity(qk.angle_encode([0.7, 0.3])) == pytest.approx(1.0)
