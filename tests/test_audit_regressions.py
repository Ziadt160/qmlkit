"""The five faults a full code-versus-docs audit turned up, each held fixed.

Every one of these shipped, and every one was reachable by following the documentation
exactly. They are collected in one file because they share a cause worth naming: a
behaviour nothing executed. The ADAPT default was never run end to end against the
recipe the guide gives; the autoencoder was left out of the regression test written for
precisely its bug; the re-uploading warning had no test asserting it stays *quiet*; and
two documented call paths were prose nothing imported.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import qmlkit as qk


# --------------------------------------------------------------------------- #
# ADAPT-VQE selected nothing and called it converged
# --------------------------------------------------------------------------- #
def test_adapt_refuses_to_pass_off_the_reference_energy_as_converged():
    """Zero operators is not convergence, it is a reference/pool mismatch.

    Following the documented chemistry recipe without ``reference=`` returned
    ``+0.720`` Ha against a true ``-1.137`` — an error of 1.86 Ha, about 1,165
    kcal/mol where chemical accuracy is 1 — with no warning of any kind.
    """
    from qmlkit.algorithms import AdaptVQE, chemistry_operator_pool, h2_hamiltonian

    hamiltonian, _ = h2_hamiltonian()
    with pytest.warns(UserWarning, match="ADAPT selected no operators"):
        result = AdaptVQE(hamiltonian, n_qubits=4, pool=chemistry_operator_pool(4)).run(seed=0)
    assert not result.operators, "the fixture only means anything if nothing was selected"


def test_adapt_is_silent_when_it_actually_works():
    """The false-positive guard: a run that finds the ground state must not warn."""
    from qmlkit.algorithms import (
        AdaptVQE,
        chemistry_operator_pool,
        exact_ground_energy,
        h2_hamiltonian,
    )

    hamiltonian, _ = h2_hamiltonian()
    exact = exact_ground_energy(hamiltonian, 4)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # any warning fails the test
        result = AdaptVQE(
            hamiltonian, n_qubits=4, pool=chemistry_operator_pool(4), reference=[0, 1]
        ).run(seed=0)
    assert result.operators
    assert result.energy == pytest.approx(exact, abs=1e-6)


# --------------------------------------------------------------------------- #
# the autoencoder advertised four optimisers and honoured two
# --------------------------------------------------------------------------- #
def _trash_states(n_qubits: int = 3, n: int = 4):
    rng = np.random.default_rng(0)
    specs = []
    for _ in range(n):
        qc = qk.QCircuit(n_qubits)
        for wire in range(n_qubits):
            qc.ry(wire, float(rng.uniform(0, np.pi)))
        for wire in range(n_qubits - 1):
            qc.cx(wire, wire + 1)
        specs.append(qc.to_spec())
    return specs


@pytest.mark.parametrize("optimizer", ["rotosolve", "spsa", "adam", "gradient-descent"])
def test_every_registered_optimizer_runs_the_autoencoder(optimizer):
    """`adam` and `gradient-descent` raised TypeError: the gradient was never injected.

    VQE, QAOA and ADAPT were all fixed for this and covered by
    `tests/test_optimizer_wiring.py`; the autoencoder was left out of both.
    """
    from qmlkit.algorithms import QuantumAutoencoder

    result = QuantumAutoencoder(3, 1, optimizer=optimizer).fit(_trash_states(), seed=0)
    assert 0.0 <= result.trash_fidelity <= 1.0


def test_the_autoencoder_gradient_is_the_real_one():
    """Injecting *a* gradient is not enough; it has to be the loss's own."""
    from qmlkit.algorithms import QuantumAutoencoder

    states = _trash_states()
    model = QuantumAutoencoder(3, 1)
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, model.encoder.n_params)
    analytic = model.gradient_of_loss(theta, states)

    eps, basis = 1e-6, np.eye(theta.size)
    numeric = np.array(
        [
            (
                model.loss(theta + eps * basis[k], states)
                - model.loss(theta - eps * basis[k], states)
            )
            / (2 * eps)
            for k in range(theta.size)
        ]
    )
    assert np.allclose(analytic, numeric, atol=1e-6)


# --------------------------------------------------------------------------- #
# the re-uploading warning fired on a circuit that was fine
# --------------------------------------------------------------------------- #
def test_the_collapse_warning_stays_quiet_when_an_entangler_breaks_the_collapse():
    """`Ry(x) Ry(t) Ry(x)` only composes on one wire with nothing in between.

    The entangler this class applies by default breaks that identity, so the model
    reaches the full spectrum — but the warning fired anyway, telling the caller their
    default-constructed model was broken. A false positive here is worse than a false
    negative: it teaches people to ignore the tool.
    """
    from qmlkit.encoding.hamiltonian import DataReuploadEncoder

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        DataReuploadEncoder(
            n_features=2, n_uploads=3, rotations=("ry",), encoding_rotation="ry"
        )  # entanglement="chain" by default


def test_the_collapse_warning_still_fires_when_the_collapse_is_real():
    """With no entangler the identity does hold, and the warning must survive."""
    from qmlkit.encoding.hamiltonian import DataReuploadEncoder

    with pytest.warns(UserWarning, match="collapse"):
        DataReuploadEncoder(
            n_features=2,
            n_uploads=3,
            rotations=("ry",),
            encoding_rotation="ry",
            entanglement=None,
        )


def test_the_warning_matches_the_spectrum_it_claims():
    """Assert the finding is *true*, not that it fired: count the frequencies."""
    from qmlkit.core.execute import expectation
    from qmlkit.encoding.hamiltonian import DataReuploadEncoder
    from qmlkit.fourier import spectrum

    def frequencies(entanglement):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            enc = DataReuploadEncoder(
                n_features=2,
                n_uploads=3,
                rotations=("ry",),
                encoding_rotation="ry",
                entanglement=entanglement,
            )
        weights = np.random.default_rng(0).uniform(-np.pi, np.pi, enc.n_weights)
        found = spectrum(
            lambda t: float(expectation(enc.build(np.array([t, 0.3])).bind(weights), qk.Z(0))),
            degree=6,
        )
        return sorted(k for k, amplitude in found.items() if abs(amplitude) > 1e-3)

    assert frequencies(None) == [3], "no entangler: the collapse is real"
    assert frequencies("chain") == [0, 1, 2, 3], "entangled: the full spectrum is reached"


# --------------------------------------------------------------------------- #
# two documented call paths raised AttributeError
# --------------------------------------------------------------------------- #
def test_the_logging_helpers_are_importable_the_way_the_docs_now_say():
    """`qmlkit.progress.log` cannot work: the name is the context manager.

    Importing `progress` into the top-level namespace rebinds the package attribute to
    the function, so the dotted path the guide and the generated HTML report both
    printed raised AttributeError for anyone who typed it.
    """
    from qmlkit.progress import log, task

    assert callable(log) and callable(task)
    assert callable(qk.progress), "the top-level name is still the context manager"


def test_the_generated_report_names_a_path_that_works():
    with qk.progress("regression") as run:
        pass
    html = run.html()
    assert "from qmlkit.progress import log" in html
    assert "<code>qmlkit.progress.log(" not in html


# --------------------------------------------------------------------------- #
# the cost estimate understated by the observable count
# --------------------------------------------------------------------------- #
def test_resources_counts_every_observable():
    """`forward_batch` and `backward_batch` both loop per observable, with no sharing.

    Measured on a default 4-qubit VQC: 164 real circuit evaluations under
    parameter-shift against the 41 `resources()` used to report. An estimate that
    exists "so nobody discovers it an hour in" was wrong by 4x for the default
    configuration.

    The assertion that matters is the *scaling*, which is readable straight off the two
    `for j, obs in enumerate(self.observables)` loops. The per-observable constants are
    deliberately not re-derived here: only the missing observable factor was the bug.
    """
    pytest.importorskip("torch")  # the layer is a torch Module; core CI has no torch
    from qmlkit.nn.models import VQC

    four = VQC(n_features=4, n_qubits=4, seed=0).quantum.resources()
    one = VQC(n_features=4, n_qubits=4, observables=[qk.Z(0)], seed=0).quantum.resources()

    assert four["n_observables"] == 4
    assert one["n_observables"] == 1
    assert four["circuits_per_sample_parameter_shift"] == 164
    assert one["circuits_per_sample_parameter_shift"] == 41

    for key in ("circuits_per_sample_parameter_shift", "passes_per_sample_adjoint"):
        assert four[key] == 4 * one[key], f"{key} must scale with the observable count"


# --------------------------------------------------------------------------- #
# and the one that was a documentation defect rather than a behaviour one
# --------------------------------------------------------------------------- #
def test_qsvc_has_a_docstring():
    """It sat below `_estimator_type`, so Python never assigned it to `__doc__`."""
    assert qk.QSVC.__doc__ is not None
    assert "support vector classifier" in qk.QSVC.__doc__
    assert qk.QSVC._estimator_type == "classifier"


# --------------------------------------------------------------------------- #
# the second pass: errors that named the wrong thing, and counts that lied
# --------------------------------------------------------------------------- #
def test_qaoa_validates_the_ansatz_width_like_vqe_does():
    """It failed several frames later with `IndexError: tuple index out of range`."""
    from qmlkit.algorithms import QAOA

    with pytest.raises(ValueError, match="the ansatz has 2 qubits but the problem acts on 4"):
        QAOA([(0, 1), (1, 2), (2, 3), (3, 0)], p=1, ansatz=qk.hardware_efficient(2, 1))


@pytest.mark.parametrize(
    "edges",
    [[(0, 1), (1, 2), (2, 0)], [[0, 1], [1, 2], [2, 0]]],
    ids=["tuples", "lists"],
)
def test_qaoa_takes_an_edge_list_however_it_is_spelled(edges):
    """`isinstance(problem[0], tuple)` sent a list-of-lists down the observable branch,
    where it died on `'list' object has no attribute 'support'`."""
    from qmlkit.algorithms import QAOA

    problem = QAOA(edges, p=1)
    assert problem.n_qubits == 3
    assert problem.edges == [(0, 1), (1, 2), (2, 0)]


def test_an_observable_still_routes_to_the_observable_branch():
    """The guard for the fix above: widening edge detection must not swallow a cost."""
    from qmlkit.algorithms import QAOA

    problem = QAOA(qk.ZZ(0, 1) + qk.ZZ(1, 2), p=1)
    assert problem.edges is None


def test_input_slots_nothing_can_read_are_refused():
    """`n_inputs` on a block with no EncodingLayer made every slot inert: `bind(x, w)`
    took any `x` at all and returned the same circuit, losing the data silently."""
    with pytest.raises(ValueError, match="no EncodingLayer"):
        qk.Ansatz(2, qk.RotationLayer("ry"), n_inputs=5)

    qk.Ansatz(2, qk.RotationLayer("ry"), n_inputs=0)  # zero is not a claim, and is fine


def test_spsa_cost_follows_the_averaging_count():
    """Hardcoded at 2, so tutorial 3's own `n_avg=50` example was understated 50x."""
    spec = qk.hardware_efficient(3, 2).build()
    assert qk.gradient_cost(spec, "spsa") == 2
    assert qk.gradient_cost(spec, "spsa", n_avg=50) == 100
    assert qk.gradient_cost(spec, "adjoint") == 1  # unchanged


def test_n_evaluations_counts_the_gradient_circuits_too():
    """It counted `energy()` calls only, so a gradient-based run reported the loss
    count and hid every circuit the optimiser actually spent."""
    from qmlkit.algorithms import VQE, ising_hamiltonian

    hamiltonian = ising_hamiltonian(3)
    cheap = VQE(hamiltonian, n_qubits=3, optimizer="gradient-descent", gradient="adjoint").run(
        seed=0, n_steps=10, lr=0.1
    )
    dear = VQE(
        hamiltonian, n_qubits=3, optimizer="gradient-descent", gradient="parameter-shift"
    ).run(seed=0, n_steps=10, lr=0.1)

    assert cheap.n_evaluations == 21, "11 losses + 10 adjoint gradients at 1 circuit each"
    assert dear.n_evaluations > cheap.n_evaluations, "parameter-shift costs far more per gradient"


def test_the_feature_map_contract_names_the_methods_that_are_actually_abstract():
    """The docstring said `build`, which is concrete; no shipped subclass overrides it."""
    import inspect

    from qmlkit.encoding.feature_maps import FeatureMap, PauliFeatureMap

    doc = inspect.getdoc(FeatureMap)
    assert "is **not** one of them" in doc
    assert "build" not in PauliFeatureMap.__dict__, "still true: subclasses do not override build"
    for required in ("angles", "n_angles", "_emit"):
        assert required in PauliFeatureMap.__dict__
