"""Observable algebra, basis rotation, and measurement grouping."""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import NumpyBackend
from qmlkit.core.observables import (
    PauliString,
    as_sum,
    basis_rotation,
    diagonal_eigenvalues,
    expectation_from_counts,
    group_qubit_wise_commuting,
    iter_terms,
    observable_support,
    required_qubits,
)


# ---------------------------------------------------------------- construction
def test_from_label_skips_identities():
    p = PauliString.from_label("ZIX")
    assert p.paulis == ((0, "Z"), (2, "X"))
    assert p.support() == (0, 2)


def test_rejects_an_unknown_pauli():
    with pytest.raises(ValueError, match="unknown Pauli"):
        PauliString(((0, "Q"),))


def test_rejects_a_repeated_qubit():
    with pytest.raises(ValueError, match="twice"):
        PauliString(((0, "Z"), (0, "X")))


def test_rejects_a_negative_qubit():
    with pytest.raises(ValueError, match="negative qubit"):
        PauliString(((-1, "Z"),))


# --------------------------------------------------------------------- algebra
def test_scalar_multiplication_both_ways():
    assert (2.0 * qk.Z(0)).coeff == 2.0
    assert (qk.Z(0) * 2.0).coeff == 2.0
    assert (-qk.Z(0)).coeff == -1.0


def test_product_of_disjoint_strings():
    p = qk.Z(0) * qk.X(1)
    assert dict(p.paulis) == {0: "Z", 1: "X"}


def test_overlapping_product_is_refused_rather_than_guessed():
    with pytest.raises(ValueError, match="overlap"):
        qk.Z(0) * qk.X(0)


def test_sum_construction_and_scaling():
    s = qk.Z(0) + qk.Z(1) + qk.X(2)
    assert len(s.terms) == 3
    assert observable_support(s) == (0, 1, 2)
    scaled = 0.5 * s
    assert all(t.coeff == 0.5 for t in scaled.terms)


def test_as_sum_and_iter_terms_normalise_a_single_string():
    assert len(as_sum(qk.Z(0)).terms) == 1
    assert len(list(iter_terms(qk.Z(0) + qk.Z(1)))) == 2


def test_required_qubits():
    assert required_qubits(qk.Z(3)) == 4
    assert required_qubits(qk.I()) == 1


def test_repr_is_readable():
    assert repr(qk.Z(0)) == "Z0"
    assert repr(2.0 * qk.Z(1)) == "2*Z1"
    assert repr(qk.I()) == "I"
    assert repr(qk.Z(0) + qk.Z(1)) == "Z0 + Z1"


# ------------------------------------------------------------ basis rotations
def test_basis_rotation_gates_per_pauli():
    assert basis_rotation(qk.Z(0)) == []
    assert basis_rotation(qk.X(0)) == [("h", (0,))]
    assert basis_rotation(qk.Y(1)) == [("sdg", (1,)), ("h", (1,))]


def test_identity_term_expectation_is_its_coefficient():
    spec = qk.angle_encode([0.3])
    obs = 2.0 * qk.I()
    assert qk.expectation(spec, obs) == pytest.approx(2.0)
    assert qk.expectation(spec, obs, shots=64, seed=0) == pytest.approx(2.0)


# ------------------------------------------------------------------ from counts
def test_diagonal_eigenvalues_follow_the_bit_convention():
    # qubit 0 is the most significant bit
    ev = diagonal_eigenvalues(qk.Z(0), 2)
    assert ev.tolist() == [1.0, 1.0, -1.0, -1.0]
    ev1 = diagonal_eigenvalues(qk.Z(1), 2)
    assert ev1.tolist() == [1.0, -1.0, 1.0, -1.0]


def test_diagonal_eigenvalues_refuses_a_non_z_string():
    with pytest.raises(ValueError, match="Z-only"):
        diagonal_eigenvalues(qk.X(0), 1)


def test_expectation_from_counts_uses_parity():
    counts = {"00": 50, "11": 50}  # perfectly correlated
    assert expectation_from_counts(qk.ZZ(0, 1), counts, 2) == pytest.approx(1.0)
    assert expectation_from_counts(qk.Z(0), counts, 2) == pytest.approx(0.0)


def test_expectation_from_counts_rejects_a_width_mismatch():
    with pytest.raises(ValueError, match="does not match"):
        expectation_from_counts(qk.Z(0), {"0": 10}, 2)


def test_expectation_from_zero_shots_is_an_error():
    with pytest.raises(ValueError, match="zero shots"):
        expectation_from_counts(qk.Z(0), {}, 1)


# ------------------------------------------------------- qubit-wise grouping
def test_commuting_terms_share_one_measurement_circuit():
    obs = qk.Z(0) + qk.Z(1) + qk.ZZ(0, 1)
    groups = group_qubit_wise_commuting(obs)
    assert len(groups) == 1


def test_anticommuting_terms_are_split():
    obs = qk.Z(0) + qk.X(0)
    groups = group_qubit_wise_commuting(obs)
    assert len(groups) == 2


def test_grouping_partitions_every_term_exactly_once():
    obs = qk.Z(0) + qk.X(0) + qk.Y(1) + qk.Z(1) + qk.ZZ(0, 1)
    groups = group_qubit_wise_commuting(obs)
    assert sum(len(g) for g in groups) == len(as_sum(obs).terms)


# --------------------------------------------------------------- correctness
def test_sampled_multi_term_observable_matches_exact():
    qc = qk.QCircuit(2)
    qc.ry(0, 0.8).cx(0, 1).rz(1, 0.4)
    spec = qc.to_spec()
    obs = qk.Z(0) + 0.5 * qk.X(1) + qk.ZZ(0, 1)
    exact = qk.expectation(spec, obs)
    sampled = qk.expectation(spec, obs, shots=200_000, seed=5)
    assert sampled == pytest.approx(exact, abs=0.02)


def test_non_hermitian_observable_is_rejected():
    spec = qk.angle_encode([0.5])
    bad = PauliString(((0, "Z"),), coeff=1j)
    with pytest.raises(ValueError, match="not Hermitian"):
        qk.expectation(spec, bad)


def test_probabilities_sum_to_one():
    qc = qk.QCircuit(3)
    qc.rotation_layer(("ry",)).entangle("ring")
    spec = qc.to_spec()
    p = qk.probabilities(spec, theta=np.linspace(0.1, 1.0, spec.n_params))
    assert p.sum() == pytest.approx(1.0)
    assert p.shape == (8,)


# --------------------------------------------------------------------------- #
# qubit-wise-commuting grouping: one circuit per measurement setting
# --------------------------------------------------------------------------- #
class _CountingBackend(NumpyBackend):
    """A NumPy backend that records how many circuits it was asked to sample."""

    name = "counting"

    def __init__(self, seed=None):
        super().__init__(seed)
        self.circuits = 0

    def counts(self, spec, shots, seed=None):
        self.circuits += 1
        return super().counts(spec, shots, seed)


@pytest.mark.parametrize(
    ("obs_fn", "n_terms", "n_settings"),
    [
        (lambda: qk.Z(0), 1, 1),
        (lambda: qk.Z(0) + 0.5 * qk.ZZ(0, 2), 2, 1),
        (lambda: qk.Z(0) + qk.Z(1) + qk.Z(2) + qk.ZZ(0, 2), 4, 1),
        # different qubits never conflict, so these still share one setting
        (lambda: qk.Z(0) + 0.3 * qk.X(1), 2, 1),
        # ...but Z0Z2 wants Z on qubit 2 while Y2 wants Y, so that pair does conflict
        (lambda: qk.Z(0) + 0.5 * qk.ZZ(0, 2) + 0.3 * qk.X(1) + 0.2 * qk.Y(2), 4, 2),
        # the same qubit wanting two different bases is what actually costs a circuit
        (lambda: qk.Z(0) + 0.3 * qk.X(0), 2, 2),
        (lambda: qk.Z(0) + qk.X(0) + qk.Y(0), 3, 3),
    ],
)
def test_sampling_costs_one_circuit_per_commuting_group(obs_fn, n_terms, n_settings):
    """Circuit count is what binds on hardware, so it is asserted, not assumed."""
    obs = obs_fn()
    assert len(as_sum(obs).terms) == n_terms
    assert len(group_qubit_wise_commuting(obs)) == n_settings

    ansatz = qk.hardware_efficient(3, 2)
    spec, theta = ansatz.build(), ansatz.init(seed=0)
    backend = _CountingBackend(seed=0)
    qk.expectation(spec, obs, theta=theta, shots=1000, seed=0, backend=backend)
    assert backend.circuits == n_settings


@pytest.mark.parametrize("seed", range(4))
def test_grouped_sampling_still_converges_on_the_exact_value(seed):
    """Grouping must not change the physics, only the number of circuits."""
    ansatz = qk.hardware_efficient(3, 2)
    spec, theta = ansatz.build(), ansatz.init(seed=seed)
    obs = qk.Z(0) + qk.Z(1) + 0.5 * qk.ZZ(0, 2) + 0.3 * qk.X(1)

    exact = qk.expval(spec, obs, theta=theta)
    sampled = qk.expectation(spec, obs, theta=theta, shots=400_000, seed=seed)
    assert sampled == pytest.approx(exact, abs=0.02)


def test_identity_terms_need_no_circuit_at_all():
    ansatz = qk.hardware_efficient(2, 1)
    spec, theta = ansatz.build(), ansatz.init(seed=0)
    backend = _CountingBackend(seed=0)
    value = qk.expectation(spec, 2.5 * qk.I(), theta=theta, shots=1000, backend=backend)
    assert value == pytest.approx(2.5)
    assert backend.circuits == 0
