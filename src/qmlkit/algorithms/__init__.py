"""Algorithms built on the rest of the library.

Each one is a *loop* over machinery that already exists — ansatz, gradients,
optimisers, observables — so each is thin, and every structural choice it makes is
an argument rather than something baked in.
"""

from __future__ import annotations

from qmlkit.algorithms.adapt import (
    AdaptResult,
    AdaptVQE,
    chemistry_operator_pool,
    default_operator_pool,
    pauli_rotation,
)
from qmlkit.algorithms.autoencoder import AutoencoderResult, QuantumAutoencoder
from qmlkit.algorithms.chemistry import (
    CHEMICAL_ACCURACY,
    HARTREE_TO_KCAL,
    h2_curve,
    h2_hamiltonian,
)
from qmlkit.algorithms.clustering import QMeans, QMeansResult
from qmlkit.algorithms.hamiltonians import (
    exact_ground_energy,
    exact_ground_state,
    hamiltonian_matrix,
    heisenberg_hamiltonian,
    ising_hamiltonian,
    max_cut_hamiltonian,
    pauli_hamiltonian,
)
from qmlkit.algorithms.molecule import (
    MolecularInfo,
    Molecule,
    from_integrals,
    hydrogen_chain,
    hydrogen_ring,
    molecular_hamiltonian,
)
from qmlkit.algorithms.qaoa import QAOA, QAOAResult
from qmlkit.algorithms.rl import ContextualBandit, QuantumPolicy, ReinforceResult, train_reinforce
from qmlkit.algorithms.vqe import OPTIMIZERS, VQE, VQEResult

__all__ = [
    "VQE",
    "QAOA",
    "QAOAResult",
    "AdaptVQE",
    "AdaptResult",
    "pauli_rotation",
    "default_operator_pool",
    "chemistry_operator_pool",
    "QuantumAutoencoder",
    "AutoencoderResult",
    "QMeans",
    "h2_hamiltonian",
    "Molecule",
    "MolecularInfo",
    "molecular_hamiltonian",
    "from_integrals",
    "hydrogen_chain",
    "hydrogen_ring",
    "h2_curve",
    "CHEMICAL_ACCURACY",
    "HARTREE_TO_KCAL",
    "QMeansResult",
    "QuantumPolicy",
    "ContextualBandit",
    "train_reinforce",
    "ReinforceResult",
    "VQEResult",
    "OPTIMIZERS",
    "pauli_hamiltonian",
    "ising_hamiltonian",
    "heisenberg_hamiltonian",
    "max_cut_hamiltonian",
    "hamiltonian_matrix",
    "exact_ground_energy",
    "exact_ground_state",
]
