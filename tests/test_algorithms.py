"""The algorithms, checked against something that is not themselves.

Every one of these has an independent oracle: VQE and ADAPT against dense
diagonalisation, QAOA against the true MaxCut, the autoencoder against an actual
round trip, shadows against exact expectation values, q-means against known cluster
labels, and the policy against a bandit whose optimum is 1.0 by construction. A
variational result that is only compared to itself is not evidence of anything.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.algorithms import (
    QAOA,
    VQE,
    AdaptVQE,
    ContextualBandit,
    QMeans,
    QuantumAutoencoder,
    QuantumPolicy,
    exact_ground_energy,
    hamiltonian_matrix,
    heisenberg_hamiltonian,
    ising_hamiltonian,
    max_cut_hamiltonian,
    pauli_hamiltonian,
    train_reinforce,
)
from qmlkit.algorithms.adapt import _commutator, pauli_rotation
from qmlkit.shadows import ClassicalShadow, shadow_shot_cost


# --------------------------------------------------------------------------- #
# Hamiltonians
# --------------------------------------------------------------------------- #
def test_hamiltonian_matrix_matches_the_pauli_algebra():
    """The dense oracle has to be right, since everything else is checked against it."""
    single = {
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    h = pauli_hamiltonian([("ZZ", (0, 1), 1.0), ("X", (0,), 0.5)])
    expected = np.kron(single["Z"], single["Z"]) + 0.5 * np.kron(single["X"], np.eye(2))
    assert hamiltonian_matrix(h, 2) == pytest.approx(expected, abs=1e-12)


def test_hamiltonians_are_hermitian():
    for h, n in (
        (ising_hamiltonian(3), 3),
        (heisenberg_hamiltonian(3), 3),
        (max_cut_hamiltonian([(0, 1), (1, 2)]), 3),
    ):
        m = hamiltonian_matrix(h, n)
        assert m == pytest.approx(m.conj().T, abs=1e-12)


def test_a_constant_term_is_kept_rather_than_dropped():
    """MaxCut's offset makes the reported energy the negated cut size directly."""
    h = max_cut_hamiltonian([(0, 1), (1, 2), (2, 0)])
    assert exact_ground_energy(h, 3) == pytest.approx(-2.0, abs=1e-9)  # a triangle cuts 2


def test_dense_matrix_refuses_a_size_it_cannot_hold():
    with pytest.raises(ValueError, match="verify small cases"):
        hamiltonian_matrix(ising_hamiltonian(20), 20)


# --------------------------------------------------------------------------- #
# VQE
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("optimizer", ["rotosolve", "gradient-descent"])
def test_vqe_approaches_the_exact_ground_energy(optimizer):
    h = ising_hamiltonian(4, j=1.0, h=0.5)
    kwargs = {"n_sweeps": 30} if optimizer == "rotosolve" else {"n_steps": 200, "lr": 0.1}
    result = VQE(h, ansatz=qk.hardware_efficient(4, 3), optimizer=optimizer).run(seed=0, **kwargs)
    assert result.exact == pytest.approx(exact_ground_energy(h, 4))
    assert result.error_vs_exact < 0.05
    assert result.energy >= result.exact - 1e-9, "variational energy cannot beat the true ground"


def test_vqe_never_reports_below_the_true_ground_state():
    """The variational principle is a hard floor; violating it means a broken energy."""
    h = heisenberg_hamiltonian(3, jx=1.0, jy=1.0, jz=1.0)
    result = VQE(h, ansatz=qk.hardware_efficient(3, 4)).run(seed=1, n_sweeps=40)
    assert result.energy >= result.exact - 1e-9


def test_vqe_rejects_an_ansatz_that_is_too_narrow():
    with pytest.raises(ValueError, match="acts on"):
        VQE(ising_hamiltonian(4), ansatz=qk.hardware_efficient(2, 1))


# --------------------------------------------------------------------------- #
# QAOA
# --------------------------------------------------------------------------- #
def test_qaoa_finds_the_maximum_cut_and_improves_with_depth():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    ratios = []
    for p in (1, 2, 3):
        solver = QAOA(edges, p=p)
        result = solver.run(seed=0, n_steps=400, lr=0.15)
        assert solver.cut_size(result.bitstring) == 4  # the true maximum for this graph
        ratios.append(result.approximation_ratio)
    assert ratios == sorted(ratios), f"more rounds should not do worse: {ratios}"


def test_qaoa_warns_that_rotosolve_is_invalid_here():
    """QAOA's cost angle drives one rz per edge, so E(gamma) is not a single sinusoid.

    Rotosolve would converge immediately on the wrong point and report it as a result,
    which is worse than failing.
    """
    from qmlkit.optim import supports_rotosolve

    solver = QAOA([(0, 1), (1, 2)], p=1, optimizer="rotosolve")
    assert not supports_rotosolve(solver.ansatz.build())
    with pytest.warns(UserWarning, match="single Pauli rotation"):
        solver.run(seed=0, n_sweeps=3)


def test_supports_rotosolve_separates_the_cases():
    assert qk.optim.supports_rotosolve(qk.hardware_efficient(4, 2).build())
    assert not qk.optim.supports_rotosolve(qk.qaoa_ansatz(4, p=2).build())
    # several occurrences on the same wire *do* compose into one rotation
    assert qk.optim.supports_rotosolve(qk.Ansatz(1, qk.share(3, qk.RotationLayer("ry"))).build())


# --------------------------------------------------------------------------- #
# ADAPT-VQE
# --------------------------------------------------------------------------- #
def test_pauli_rotation_reproduces_the_matrix_exponential():
    """Everything ADAPT does rests on this primitive being exactly right."""
    scipy_linalg = pytest.importorskip("scipy.linalg")
    for paulis in (((0, "Y"),), ((0, "Y"), (1, "Z")), ((0, "X"), (1, "Y"))):
        term = qk.PauliString(paulis, 1.0)
        angle = 0.7
        builder = qk.QCircuit(2)
        pauli_rotation(builder, term, angle)
        got = qk.statevector(builder.to_spec())

        matrix = hamiltonian_matrix(term, 2)
        want = scipy_linalg.expm(-1j * angle * matrix / 2) @ np.eye(4)[:, 0]
        overlap = np.vdot(want, got)
        phase = overlap / abs(overlap)
        assert got == pytest.approx(phase * want, abs=1e-12)


def test_the_commutator_is_hermitian_so_it_can_be_measured():
    """``[H,P]`` is anti-Hermitian; ``i[H,P]`` is the observable, and the sign matters."""
    h, p = ising_hamiltonian(3), qk.PauliString(((0, "Y"),), 1.0)
    matrix = hamiltonian_matrix(_commutator(h, p), 3)
    assert matrix == pytest.approx(matrix.conj().T, abs=1e-12)
    hm, pm = hamiltonian_matrix(h, 3), hamiltonian_matrix(p, 3)
    assert matrix == pytest.approx(1j * (hm @ pm - pm @ hm), abs=1e-12)


def test_adapt_grows_a_circuit_and_lowers_the_energy():
    h = ising_hamiltonian(4, j=1.0, h=0.5)
    result = AdaptVQE(h, 4).run(max_operators=8, n_steps=100, lr=0.25)
    assert result.n_operators > 0, "ADAPT selected no operators at all"
    assert result.theta.size == result.n_operators
    assert result.history[-1] < result.history[0]
    assert result.energy >= result.exact - 1e-9
    assert result.error_vs_exact < 0.2


def test_adapt_stops_when_no_operator_helps():
    """A high tolerance should end it immediately rather than padding the circuit."""
    result = AdaptVQE(ising_hamiltonian(3), 3).run(max_operators=6, gradient_tol=1e9)
    assert result.n_operators == 0


# --------------------------------------------------------------------------- #
# quantum autoencoder
# --------------------------------------------------------------------------- #
def _compressible(n: int = 5, seed: int = 0):
    """States that genuinely occupy only wires 0 and 1."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        builder = qk.QCircuit(4)
        a, b = rng.uniform(0, np.pi, 2)
        builder.ry(0, a).ry(1, b).cx(0, 1)
        out.append(builder.to_spec())
    return out


def test_autoencoder_compresses_a_compressible_family():
    states = _compressible()
    model = QuantumAutoencoder(4, n_latent=2, encoder=qk.hardware_efficient(4, 6))
    result = model.fit(states, seed=0, n_sweeps=60)
    assert result.trash_fidelity > 0.95


def test_trash_fidelity_equals_the_round_trip_fidelity():
    """The loss must be the thing it claims to be a proxy for, not merely correlated.

    Purity is *not* — an encoder can leave the trash pure but pointing away from
    ``|0>``, scoring purity 0.998 while the round trip returns only 0.21.
    """
    states = _compressible(n=3)
    model = QuantumAutoencoder(4, n_latent=2)
    theta = model.encoder.init("small", seed=0)
    assert model.trash_fidelity(theta, states) == pytest.approx(
        model.round_trip_fidelity(theta, states), abs=1e-9
    )


def test_autoencoder_rejects_an_impossible_latent_size():
    with pytest.raises(ValueError, match="n_latent"):
        QuantumAutoencoder(4, n_latent=4)


# --------------------------------------------------------------------------- #
# classical shadows
# --------------------------------------------------------------------------- #
def test_shadows_estimate_observables_they_never_measured_directly():
    ansatz = qk.hardware_efficient(3, 2)
    spec = ansatz.build(ansatz.init("uniform", seed=0))
    shadow = ClassicalShadow(spec, n_snapshots=6000, seed=0)
    for obs in (qk.Z(0), qk.X(1), qk.ZZ(0, 2), qk.Z(0) + 0.5 * qk.ZZ(0, 2)):
        assert shadow.expectation(obs) == pytest.approx(qk.expval(spec, obs), abs=0.15)


def test_shadow_accuracy_improves_with_more_snapshots():
    ansatz = qk.hardware_efficient(2, 2)
    spec = ansatz.build(ansatz.init("uniform", seed=0))
    exact = qk.expval(spec, qk.Z(0))

    def mean_error(n):
        return np.mean(
            [
                abs(ClassicalShadow(spec, n_snapshots=n, seed=s).expectation(qk.Z(0)) - exact)
                for s in range(4)
            ]
        )

    assert mean_error(4000) < mean_error(250)


def test_shadow_cost_is_logarithmic_in_the_observable_count():
    """The whole point: many observables cost barely more than a few.

    Compared against the alternative rather than an arbitrary bound — measuring each
    observable separately is *linear* in M, so a 1000x increase in M would cost 1000x.
    """
    few, many = shadow_shot_cost(2, 10), shadow_shot_cost(2, 10_000)
    assert many / few < 10, "1000x the observables must not cost anything like 1000x"
    # and the growth should track log(M) specifically
    expected = np.log(2 * 10_000) / np.log(2 * 10)
    assert many / few == pytest.approx(expected, rel=0.01)
    assert shadow_shot_cost(3, 100) > shadow_shot_cost(2, 100)  # exponential in locality


# --------------------------------------------------------------------------- #
# q-means and the policy
# --------------------------------------------------------------------------- #
def test_qmeans_recovers_known_clusters():
    X, y = qk.datasets.make_blobs(n_samples=40, n_features=2, centers=2, seed=0)
    X = qk.AngleScaler().fit(X).transform(X)
    result = QMeans(2, seed=0).fit(X)
    agreement = max((result.labels == y).mean(), (result.labels == 1 - y).mean())
    assert agreement > 0.9
    assert result.history == sorted(result.history, reverse=True), "inertia must not increase"


def test_qmeans_takes_any_feature_map():
    X, _ = qk.datasets.make_blobs(n_samples=20, n_features=2, centers=2, seed=0)
    X = qk.AngleScaler().fit(X).transform(X)
    for fmap in (qk.AngleFeatureMap(2), qk.ZZFeatureMap(2, reps=1)):
        assert QMeans(2, feature_map=fmap, seed=0).fit(X).labels.shape == (20,)


def test_policy_gradient_learns_a_bandit_with_a_known_optimum():
    policy = QuantumPolicy(2, 2)
    result = train_reinforce(policy, ContextualBandit(seed=0), n_episodes=300, lr=0.3, seed=0)
    early = float(np.mean(result.returns[:50]))
    assert result.mean_return() > early + 0.1, f"no learning: {early} -> {result.mean_return()}"


def test_policy_probabilities_are_a_distribution():
    policy = QuantumPolicy(2, 3)
    probs = policy.probabilities(np.array([0.3, -0.7]))
    assert probs.shape == (3,)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs > 0).all()


# --------------------------------------------------------------------------- #
# regressions: things that failed quietly or failed obscurely
# --------------------------------------------------------------------------- #
def test_a_policy_can_be_seeded_independently():
    """The seed was once a constant inside the constructor.

    Every QuantumPolicy then started from the same point, so "average over seeds" —
    the only honest way to report an RL result — could not be written at all.
    """
    first, second = QuantumPolicy(2, 2, seed=1), QuantumPolicy(2, 2, seed=2)
    assert not np.allclose(first.theta, second.theta)
    assert np.allclose(QuantumPolicy(2, 2, seed=1).theta, first.theta)


def test_qmeans_refuses_more_clusters_than_samples():
    """It used to surface numpy's message about sampling, which mentions no clusters."""
    with pytest.raises(ValueError, match="cannot form 5 clusters from 3 samples"):
        QMeans(5, seed=0).fit(np.zeros((3, 2)))


def test_qmeans_survives_duplicate_points_and_empty_clusters():
    result = QMeans(2, seed=0).fit(np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]))
    assert result.labels.shape == (3,)
    assert np.isfinite(result.inertia)


def test_qaoa_rejects_an_empty_problem_clearly():
    """It used to fail inside observable_support with an AttributeError about lists."""
    with pytest.raises(ValueError, match="empty edge list defines nothing"):
        QAOA([], p=1)


def test_qaoa_accepts_a_bare_cost_observable():
    """Not every problem is an edge list; MaxCut is only the demo."""
    solver = QAOA(max_cut_hamiltonian([(0, 1), (1, 2)]), p=1, n_qubits=3)
    result = solver.run(seed=0, n_steps=150, lr=0.2)
    assert result.energy >= result.exact - 1e-9
    with pytest.raises(ValueError, match="given as edges"):
        solver.cut_size(result.bitstring)


@pytest.mark.parametrize("optimizer", ["rotosolve", "gradient-descent", "spsa"])
def test_the_reported_energy_belongs_to_the_reported_parameters(optimizer):
    """A result whose energy and theta disagree is worse than no result."""
    kwargs = {
        "rotosolve": {"n_sweeps": 10},
        "gradient-descent": {"n_steps": 40, "lr": 0.1},
        "spsa": {"n_iterations": 60},
    }[optimizer]
    solver = VQE(ising_hamiltonian(3), ansatz=qk.hardware_efficient(3, 2), optimizer=optimizer)
    result = solver.run(seed=0, **kwargs)
    assert solver.energy(result.theta) == pytest.approx(result.energy, abs=1e-12)


def test_vqe_runs_on_shots_alone():
    """The hardware path: no statevector anywhere in the loop."""
    result = VQE(
        ising_hamiltonian(3), ansatz=qk.hardware_efficient(3, 2), optimizer="spsa", shots=2048
    ).run(seed=0, n_iterations=80)
    assert np.isfinite(result.energy)
    assert result.energy < 0  # it has to have moved off the |000> energy of +2


def test_autoencoder_honours_custom_trash_wires():
    rng = np.random.default_rng(0)
    states = []
    for _ in range(3):
        builder = qk.QCircuit(4)
        builder.ry(1, rng.uniform(0, np.pi)).ry(3, rng.uniform(0, np.pi))
        states.append(builder.to_spec())
    model = QuantumAutoencoder(4, n_latent=2, trash=[0, 2])
    result = model.fit(states, seed=0, n_sweeps=20)
    assert model.trash == [0, 2]
    assert result.trash_fidelity == pytest.approx(result.fidelity, abs=1e-9)


def test_adapt_accepts_a_custom_operator_pool():
    pool = [qk.PauliString(((0, "Y"),), 1.0), qk.PauliString(((1, "Y"),), 1.0)]
    result = AdaptVQE(ising_hamiltonian(3), 3, pool=pool).run(max_operators=4, n_steps=60, lr=0.2)
    assert 0 < result.n_operators <= 4
    assert all(op in pool for op in result.operators)


# --------------------------------------------------------------------------- #
# molecular Hamiltonians
# --------------------------------------------------------------------------- #
def test_h2_reproduces_the_published_ground_state_energy():
    """The whole chemistry pipeline is only trustworthy if this number is right.

    -1.1373 Ha at 0.735 A is the standard FCI/STO-3G result, and it is the one
    external fact this module depends on.
    """
    from qmlkit.algorithms import h2_hamiltonian

    hamiltonian, info = h2_hamiltonian(0.735)
    assert info["n_qubits"] == 4
    assert exact_ground_energy(hamiltonian, 4) == pytest.approx(-1.1373, abs=1e-4)


def test_h2_curve_binds_and_dissociates():
    """A potential energy curve has to have the right shape, not just one point."""
    from qmlkit.algorithms import h2_hamiltonian

    energies = {r: exact_ground_energy(h2_hamiltonian(r)[0], 4) for r in (0.4, 0.735, 1.4, 3.0)}
    assert energies[0.735] == min(energies.values()), "the minimum must be near equilibrium"
    assert energies[0.4] > energies[0.735], "repulsion must dominate at short range"
    assert energies[3.0] > energies[1.4], "the curve must flatten as the atoms separate"
    assert energies[3.0] == pytest.approx(-0.933, abs=0.01)  # two isolated H atoms


def test_the_pauli_decomposition_round_trips():
    """Projecting onto Paulis and rebuilding the matrix must be lossless."""
    from qmlkit.algorithms.chemistry import _matrix, h2_hamiltonian

    hamiltonian, _ = h2_hamiltonian(0.9)
    rebuilt = hamiltonian_matrix(hamiltonian, 4)
    direct, _ = _matrix(0.9)
    assert rebuilt == pytest.approx(direct, abs=1e-10)


def test_the_generic_adapt_pool_is_blind_to_a_molecular_hamiltonian():
    """Not a bug — physics, and worth pinning so nobody 'fixes' it later.

    A molecular Hamiltonian conserves particle number, so every generator that does
    not has exactly zero gradient at Hartree-Fock. ADAPT correctly grows nothing.
    """
    from qmlkit.algorithms import chemistry_operator_pool, default_operator_pool, h2_hamiltonian
    from qmlkit.algorithms.adapt import _commutator

    hamiltonian, _ = h2_hamiltonian(0.735)
    builder = qk.QCircuit(4)
    builder.x(0).x(1)
    hartree_fock = builder.to_spec()

    for operator in default_operator_pool(4):
        commutator = _commutator(hamiltonian, operator)
        if commutator.terms:
            assert abs(qk.expval(hartree_fock, commutator)) < 1e-12

    best = max(
        abs(qk.expval(hartree_fock, _commutator(hamiltonian, op)))
        for op in chemistry_operator_pool(4)
    )
    assert best > 0.1, "a particle-number-conserving pool must find something to do"


def test_adapt_solves_h2_exactly_with_one_operator():
    """The textbook result: a single double excitation is the whole ansatz."""
    from qmlkit.algorithms import CHEMICAL_ACCURACY, chemistry_operator_pool, h2_hamiltonian

    hamiltonian, _ = h2_hamiltonian(0.735)
    exact = exact_ground_energy(hamiltonian, 4)
    result = AdaptVQE(hamiltonian, 4, pool=chemistry_operator_pool(4), reference=[0, 1]).run(
        max_operators=3, gradient_tol=1e-4, n_steps=250, lr=0.3
    )

    assert result.n_operators == 1
    assert abs(result.energy - exact) < CHEMICAL_ACCURACY
    assert result.energy >= exact - 1e-9
