"""Every optimiser a solver advertises must actually run from that solver.

`OPTIMIZERS` is an open dict, and three algorithms take a name out of it. Two of
the four entries need the gradient injected by the caller, and the injection named
only one of them — so `optimizer="adam"` raised `TypeError: _adam() missing 1
required keyword-only argument: 'grad'` from VQE, QAOA and AdaptVQE alike, while
being listed as a supported value in all three docstrings.

That is a documented feature that never worked, which is the failure mode this file
exists to prevent. The parametrisation is over `OPTIMIZERS` itself rather than a
hand-written list, so a fifth optimiser cannot be added without being covered here.
"""

from __future__ import annotations

import pytest

from qmlkit.algorithms.adapt import AdaptVQE
from qmlkit.algorithms.hamiltonians import ising_hamiltonian
from qmlkit.algorithms.qaoa import QAOA
from qmlkit.algorithms.vqe import OPTIMIZERS, VQE

#: Each optimiser spells its iteration count differently, so a solver's `**kwargs`
#: passthrough needs the right one. Kept short: this asserts wiring, not convergence.
BUDGET = {
    "rotosolve": {"n_sweeps": 2},
    "spsa": {"n_iterations": 5},
    "gradient-descent": {"n_steps": 5},
    "adam": {"n_steps": 5},
}


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_optimizer_runs_from_vqe(name: str) -> None:
    result = VQE(ising_hamiltonian(3), n_qubits=3, optimizer=name).run(
        seed=0, compare_exact=False, **BUDGET[name]
    )
    assert isinstance(result.energy, float)
    assert len(result.history) >= 2


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_optimizer_runs_from_qaoa(name: str) -> None:
    result = QAOA([(0, 1), (1, 2), (0, 2)], p=1, optimizer=name).run(
        seed=0, compare_exact=False, **BUDGET[name]
    )
    assert result.bitstring
    assert isinstance(result.cut_value, int | float)


@pytest.mark.parametrize("name", sorted(OPTIMIZERS))
def test_every_optimizer_runs_from_adapt(name: str) -> None:
    result = AdaptVQE(ising_hamiltonian(3), n_qubits=3, optimizer=name).run(
        seed=0, max_operators=1, compare_exact=False, **BUDGET[name]
    )
    assert isinstance(result.energy, float)


def test_the_budget_table_covers_every_optimizer() -> None:
    """A new optimiser must be given a budget here, not silently skipped."""
    assert set(BUDGET) == set(OPTIMIZERS)


def test_gradient_optimizers_reach_the_known_minimum() -> None:
    """Wiring is not enough: the injected gradient has to be the right one.

    A `grad` that was injected but wrong would pass every test above. Given enough
    steps both gradient routes must land on the exact ground state.
    """
    exact = None
    for name in ("gradient-descent", "adam"):
        result = VQE(ising_hamiltonian(3), n_qubits=3, optimizer=name).run(
            seed=0, n_steps=400, lr=0.05
        )
        exact = result.exact if exact is None else exact
        assert result.exact == exact
        assert abs(result.energy - result.exact) < 5e-2, f"{name} did not converge"
