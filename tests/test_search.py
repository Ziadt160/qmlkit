"""The grid search: identical folds, honest verdicts, and broken points skipped.

The behaviour worth pinning is not that it finds a winner — any loop over a grid does
that. It is that a configuration the diagnostics already condemn is *not fitted*, is
reported with its reason rather than ranked last, and that a lead inside the fold
spread is refused rather than announced.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.search import SearchResult, SearchRow, search

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture
def moons():
    return qk.datasets.make_moons(n_samples=60, seed=0)


@pytest.fixture
def wide():
    """Four features, so a four-qubit circuit is reachable.

    The pruning cases need width: `basic_entangler` only goes flat once the CNOT ring
    is deep enough, which is a property of the circuit rather than of the data.
    """
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (60, 4))
    return X, (X[:, 0] + X[:, 1] > np.pi).astype(int)


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #
def test_a_bare_value_counts_as_a_one_point_axis(moons):
    X, y = moons
    assert len(search(X, y, n_layers=1, cv=2, dry_run=True).rows) == 1


def test_the_grid_is_the_product_of_its_axes(moons):
    X, y = moons
    result = search(X, y, n_layers=[1, 2], lr=[0.05, 0.1], epochs=[10], cv=2, dry_run=True)
    assert len(result.rows) == 4
    assert set(result.varied) == {"n_layers", "lr"}  # epochs had a single value


def test_only_the_axes_that_varied_are_labelled(moons):
    X, y = moons
    result = search(X, y, n_layers=[1, 2], cv=2, dry_run=True)
    assert all("n_layers=" in r.label(result.varied) for r in result.rows)
    assert all("batch_size" not in r.label(result.varied) for r in result.rows)


def test_a_typo_in_an_axis_name_is_refused_with_a_suggestion(moons):
    """A silently ignored axis is a sweep that varies nothing."""
    X, y = moons
    with pytest.raises(ValueError, match="n_layers"):
        search(X, y, n_layer=[1, 2], cv=2, dry_run=True)


def test_n_qubits_is_read_off_the_data_and_said_so(moons):
    X, y = moons  # two features
    result = search(X, y, cv=2, dry_run=True)
    assert result.rows[0].config["n_qubits"] == 2
    assert any("n_qubits was not given" in note for note in result.notes)


def test_asking_for_more_qubits_than_features_is_an_error(moons):
    X, y = moons
    with pytest.raises(ValueError, match="cannot invent them"):
        search(X, y, n_qubits=[8], cv=2, dry_run=True)


def test_max_configs_samples_reproducibly(moons):
    X, y = moons
    kwargs = dict(n_layers=[1, 2, 3], lr=[0.05, 0.1], cv=2, dry_run=True, max_configs=3)
    first, second = search(X, y, **kwargs), search(X, y, **kwargs)
    assert len(first.rows) == 3
    assert [r.config for r in first.rows] == [r.config for r in second.rows]
    assert any("sampled 3 of 6" in note for note in first.notes)


# --------------------------------------------------------------------------- #
# pruning: the part that needs the diagnostics
# --------------------------------------------------------------------------- #
def test_untrainable_configurations_are_skipped_with_their_reason(wide):
    X, y = wide
    result = search(
        X, y, ansatz=["hardware_efficient", "basic_entangler"], n_layers=[2, 3], cv=2,
        dry_run=True, prune="untrainable",
    )
    skipped = {(r.config["ansatz"], r.config["n_layers"]) for r in result.pruned}
    # d<Z0>/dtheta_0 is exactly zero for a 4-qubit, 3-layer CNOT ring
    assert ("basic_entangler", 3) in skipped
    assert not any(a == "hardware_efficient" for a, _ in skipped)
    assert all("FLAT_GRADIENTS" in r.pruned for r in result.pruned)


def test_prune_none_fits_everything(wide):
    X, y = wide
    result = search(
        X, y, ansatz=["hardware_efficient", "basic_entangler"], n_layers=[2, 3], cv=2,
        dry_run=True, prune="none",
    )
    assert not result.pruned


def test_every_row_carries_its_diagnosis_whether_or_not_it_was_pruned(moons):
    """A configuration that scores well *and* carries DEAD_WEIGHTS is worth seeing."""
    X, y = moons
    result = search(X, y, cv=2, dry_run=True, prune="error")
    assert not result.pruned
    assert any(r.findings for r in result.rows)


def test_prune_accepts_an_explicit_list_of_codes(wide):
    X, y = wide
    result = search(
        X, y, ansatz=["hardware_efficient"], cv=2, dry_run=True,
        prune=["UNMEASURABLE_WEIGHTS"],
    )
    assert len(result.pruned) == 1
    assert "UNMEASURABLE_WEIGHTS" in result.pruned[0].pruned


def test_an_unknown_prune_level_is_refused(moons):
    X, y = moons
    with pytest.raises(ValueError, match="untrainable"):
        search(X, y, cv=2, dry_run=True, prune="agressive")


def test_a_configuration_that_cannot_be_built_is_reported_not_raised(moons):
    X, y = moons
    result = search(X, y, ansatz=["no_such_ansatz"], cv=2, dry_run=True)
    assert len(result.pruned) == 1
    assert "could not be built" in result.pruned[0].pruned


def test_a_pruned_reason_does_not_split_a_decimal_in_half():
    """`message.split('.')[0]` turns "variance 1.14e-33" into "variance 1"."""
    from qmlkit.search import _first_sentence

    assert _first_sentence("variance 1.14e-33 for weight 0. And more.") == (
        "variance 1.14e-33 for weight 0."
    )
    assert _first_sentence("x" * 200).endswith("...")


# --------------------------------------------------------------------------- #
# a real run
# --------------------------------------------------------------------------- #
def test_a_real_search_ranks_and_rebuilds_the_winner(moons):
    X, y = moons
    result = search(X, y, n_layers=[1, 2], epochs=8, cv=2, verbose=False)
    assert len(result.ran) == 2
    assert result.best is not None
    assert set(result.best_config) == set(qk.AXES)
    assert type(result.model()).__name__ == "VQC"
    assert result.model(n_layers=3) is not None  # overrides apply


def test_every_configuration_is_scored_on_identical_folds(moons):
    X, y = moons
    first = search(X, y, n_layers=[1, 2], epochs=5, cv=2, verbose=False)
    second = search(X, y, n_layers=[1, 2], epochs=5, cv=2, verbose=False)
    for (a_train, a_test), (b_train, b_test) in zip(
        first.extras["folds"], second.extras["folds"], strict=True
    ):
        np.testing.assert_array_equal(a_train, b_train)
        np.testing.assert_array_equal(a_test, b_test)


def test_dry_run_fits_nothing_but_still_prunes(wide):
    X, y = wide
    result = search(
        X, y, ansatz=["hardware_efficient", "basic_entangler"], n_layers=[2, 3], cv=2,
        dry_run=True, prune="untrainable",
    )
    assert not result.ran
    assert result.pruned  # pruning still happened
    assert "would run" in str(result)
    with pytest.raises(ValueError, match="no configuration ran"):
        _ = result.best_config


def test_the_printout_is_ascii(moons):
    """It goes to a Windows console as often as to a notebook."""
    X, y = moons
    assert str(search(X, y, n_layers=[1, 2], epochs=5, cv=2, verbose=False)).isascii()


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #
def test_a_lead_inside_the_fold_spread_is_not_called_a_winner():
    result = SearchResult(
        "classification", "balanced_accuracy", 60, 3, ("n_layers",),
        (SearchRow({"n_layers": 2}, 0.81, 0.05), SearchRow({"n_layers": 1}, 0.80, 0.05)),
    )
    assert "has not separated them" in result.verdict


def test_a_lead_outside_the_fold_spread_is():
    result = SearchResult(
        "classification", "balanced_accuracy", 60, 3, ("n_layers",),
        (SearchRow({"n_layers": 2}, 0.95, 0.01), SearchRow({"n_layers": 1}, 0.70, 0.01)),
    )
    assert "wins at 0.950" in result.verdict


def test_a_verdict_with_nothing_fitted_says_which_kind_of_nothing():
    everything_pruned = SearchResult(
        "classification", "balanced_accuracy", 60, 3, ("n_layers",),
        (SearchRow({"n_layers": 1}, pruned="FLAT_GRADIENTS", fitted=False),),
    )
    assert "pruned" in everything_pruned.verdict

    dry = SearchResult(
        "classification", "balanced_accuracy", 60, 3, ("n_layers",),
        (SearchRow({"n_layers": 1}, fitted=False),),
    )
    assert "would run" in dry.verdict


# --------------------------------------------------------------------------- #
# the feature-map registry
# --------------------------------------------------------------------------- #
def test_feature_maps_are_reachable_by_name_and_extensible():
    assert {"angle", "z", "zz", "pauli"} <= set(qk.list_feature_maps())
    qk.register_feature_map("probe_map", lambda n_qubits: qk.AngleFeatureMap(n_qubits))
    try:
        assert "probe_map" in qk.list_feature_maps()
    finally:
        from qmlkit.search import _FEATURE_MAPS

        _FEATURE_MAPS.pop("probe_map", None)


def test_an_unknown_feature_map_is_reported_not_raised(moons):
    X, y = moons
    result = search(X, y, feature_map=["zzz"], cv=2, dry_run=True)
    assert "could not be built" in result.pruned[0].pruned
