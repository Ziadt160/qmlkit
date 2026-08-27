"""Generic molecules, the feature pipeline, and fitting into scikit-learn.

Three things that decide whether anyone outside this repository can use the library:
a Hamiltonian for *their* molecule, a reduction step that does not leak the test set,
and estimators that survive ``clone``.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.algorithms import (
    Molecule,
    exact_ground_energy,
    from_integrals,
    h2_hamiltonian,
    hydrogen_chain,
    hydrogen_ring,
    molecular_hamiltonian,
)


# --------------------------------------------------------------------------- #
# molecules
# --------------------------------------------------------------------------- #
def test_the_general_path_reproduces_the_h2_specific_one():
    """Two independent routes to the same Hamiltonian must agree.

    The H2 builder fixes the orbitals by symmetry; the general one runs a real SCF.
    If they disagree, one of them is wrong.
    """
    general, info = molecular_hamiltonian(Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.735))]))
    specific, _ = h2_hamiltonian(0.735)
    assert info.n_qubits == 4
    assert exact_ground_energy(general, 4) == pytest.approx(
        exact_ground_energy(specific, 4), abs=1e-7
    )


def test_hartree_fock_is_never_below_the_full_solution():
    """The variational principle one level down: correlation energy is negative."""
    for molecule in (
        Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.735))]),
        hydrogen_chain(4, 0.9),
        hydrogen_ring(4, 1.0),
    ):
        hamiltonian, info = molecular_hamiltonian(molecule)
        fci = exact_ground_energy(hamiltonian, info.n_qubits)
        assert info.hartree_fock_energy is not None
        assert fci <= info.hartree_fock_energy + 1e-9


@pytest.mark.parametrize(
    ("molecule", "n_qubits"),
    [
        (Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.74))]), 4),
        (hydrogen_chain(4, 0.9), 8),
        (Molecule([("He", (0, 0, 0)), ("H", (0, 0, 0.775))], charge=1), 4),
    ],
)
def test_a_molecule_uses_two_qubits_per_spatial_orbital(molecule, n_qubits):
    _, info = molecular_hamiltonian(molecule)
    assert info.n_qubits == n_qubits
    assert info.n_qubits == 2 * info.n_orbitals


def test_an_unsupported_element_says_what_to_do_instead():
    """The boundary is stated, with the route past it."""
    with pytest.raises(ValueError, match="from_integrals"):
        Molecule([("C", (0, 0, 0)), ("O", (0, 0, 1.1))])


def test_from_integrals_accepts_integrals_computed_elsewhere():
    """The general route — what PySCF or OpenFermion output would take."""
    from qmlkit.algorithms.molecule import _ao_integrals, _rhf

    molecule = Molecule([("H", (0, 0, 0)), ("H", (0, 0, 0.735))])
    general, _ = molecular_hamiltonian(molecule)

    overlap, core, eri, repulsion = _ao_integrals(molecule)
    coefficients, _ = _rhf(overlap, core, eri, 2)
    one_body = coefficients.T @ core @ coefficients
    two_body = np.einsum(
        "pi,qj,rk,sl,pqrs->ijkl",
        coefficients,
        coefficients,
        coefficients,
        coefficients,
        eri,
        optimize=True,
    )
    rebuilt, info = from_integrals(
        one_body, two_body, n_electrons=2, nuclear_repulsion=float(repulsion)
    )
    assert info.n_electrons == 2
    assert exact_ground_energy(rebuilt, 4) == pytest.approx(
        exact_ground_energy(general, 4), abs=1e-10
    )


def test_an_active_space_shrinks_the_register():
    """The usual way to fit a molecule onto the machine you actually have."""
    full, full_info = molecular_hamiltonian(hydrogen_chain(4, 0.9))
    reduced, reduced_info = molecular_hamiltonian(hydrogen_chain(4, 0.9), active_space=(0, 1))
    assert full_info.n_qubits == 8
    assert reduced_info.n_qubits == 4
    assert reduced_info.active_space == (0, 1)
    assert len(reduced.terms) < len(full.terms)


def test_integrals_of_the_wrong_shape_are_refused():
    with pytest.raises(ValueError, match="must be square"):
        from_integrals(np.zeros((2, 3)), np.zeros((2, 2, 2, 2)), 2)
    with pytest.raises(ValueError, match="must have shape"):
        from_integrals(np.zeros((2, 2)), np.zeros((3, 3, 3, 3)), 2)


def test_a_register_too_large_to_build_is_refused_with_advice():
    with pytest.raises(ValueError, match="active_space"):
        from_integrals(np.zeros((8, 8)), np.zeros((8,) * 4), 8)


# --------------------------------------------------------------------------- #
# the feature pipeline
# --------------------------------------------------------------------------- #
def test_the_pipeline_never_refits_on_transform():
    """Fitting the reducer on the test set is the classic silent leak."""
    rng = np.random.default_rng(0)
    train, test = rng.normal(size=(60, 12)), rng.normal(size=(20, 12)) + 5.0
    pipeline = qk.FeaturePipeline(n_qubits=4).fit(train)
    first = pipeline.transform(test)
    pipeline.transform(train)  # transforming other data must not disturb the fit
    assert pipeline.transform(test) == pytest.approx(first)


def test_the_pipeline_reports_what_the_reduction_cost():
    rng = np.random.default_rng(0)
    pipeline = qk.FeaturePipeline(n_qubits=3).fit(rng.normal(size=(80, 15)))
    assert pipeline.explained_variance_ is not None
    assert 0.0 < pipeline.explained_variance_ < 1.0


def test_the_pipeline_lands_inside_the_requested_angle_range():
    rng = np.random.default_rng(0)
    pipeline = qk.FeaturePipeline(n_qubits=4, angle_range=(0.0, np.pi))
    out = pipeline.fit_transform(rng.normal(size=(50, 10)))
    assert out.shape == (50, 4)
    assert out.min() >= -1e-9
    assert out.max() <= np.pi + 1e-9


def test_the_pipeline_refuses_to_invent_features():
    with pytest.raises(ValueError, match="cannot map 3 features onto 8 qubits"):
        qk.FeaturePipeline(n_qubits=8).fit(np.zeros((10, 3)))


def test_a_constant_column_does_not_divide_by_zero():
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.normal(size=40), np.ones(40)])
    assert np.isfinite(qk.FeaturePipeline(n_qubits=2).fit_transform(data)).all()


def test_the_pipeline_matches_doing_it_by_hand():
    """It is a convenience, not a different computation."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 8))
    by_hand = qk.AngleScaler().fit_transform(
        qk.PCAReducer(3).fit_transform((data - data.mean(0)) / data.std(0))
    )
    assert qk.FeaturePipeline(n_qubits=3).fit_transform(data) == pytest.approx(by_hand)


# --------------------------------------------------------------------------- #
# scikit-learn interoperability
# --------------------------------------------------------------------------- #
def test_estimators_survive_a_clone():
    """``clone`` is what Pipeline, GridSearchCV and cross_val_score are built on."""
    base = pytest.importorskip("sklearn.base")
    estimator = qk.QSVC(qk.ZZFeatureMap(2, reps=2), C=2.0)
    copy = base.clone(estimator)
    assert copy.C == 2.0
    assert type(copy.feature_map) is type(estimator.feature_map)
    assert base.is_classifier(estimator)
    assert base.is_regressor(qk.QSVR(qk.AngleFeatureMap(2)))


def test_a_quantum_estimator_runs_inside_a_sklearn_pipeline():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    X, y = qk.datasets.ad_hoc_data(n_samples=24, n_features=2, gap=0.4, seed=0)
    pipeline = Pipeline(
        [("prep", qk.FeaturePipeline(n_qubits=2)), ("qsvc", qk.QSVC(qk.ZZFeatureMap(2, reps=2)))]
    )
    scores = cross_val_score(pipeline, X, y, cv=3)
    assert len(scores) == 3
    assert np.isfinite(scores).all()


def test_grid_search_can_tune_a_quantum_estimator():
    pytest.importorskip("sklearn")
    from sklearn.model_selection import GridSearchCV

    X, y = qk.datasets.ad_hoc_data(n_samples=24, n_features=2, gap=0.4, seed=0)
    search = GridSearchCV(qk.QSVC(qk.ZZFeatureMap(2, reps=2)), {"C": [0.5, 2.0]}, cv=2)
    search.fit(X, y)
    assert search.best_params_["C"] in (0.5, 2.0)


def test_get_params_reads_the_constructor_signature():
    pipeline = qk.FeaturePipeline(n_qubits=5, method="truncate", standardize=False)
    params = pipeline.get_params(deep=False)
    assert params == {
        "n_qubits": 5,
        "method": "truncate",
        "standardize": False,
        "angle_range": (0.0, 2 * np.pi),
    }


def test_set_params_rejects_a_name_it_does_not_have():
    with pytest.raises(ValueError, match="invalid parameter"):
        qk.FeaturePipeline(n_qubits=2).set_params(nonsense=1)
