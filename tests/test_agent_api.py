"""The API as a coding agent meets it.

Most code written against this library is now written by a language model, and a
model does not read the documentation site before it types. It guesses a name from
what it has read elsewhere, runs it, reads the traceback, and tries again. That
loop is the real interface, and these tests hold it to three promises:

1. A wrong *name* is answered with the right one — from the table of what PennyLane
   and Qiskit call the same thing, or failing that a near-match over the exports.
2. That table cannot rot. Every target it names must still exist, or this file
   fails; a translation pointing at a deleted function is worse than none.
3. Registry lookups say what is valid, not merely that the input was not.
"""

from __future__ import annotations

import doctest
import re

import numpy as np
import pytest

import qmlkit as qk
from qmlkit import _aliases, diagnostics
from qmlkit.ansatz import EncodingLayer, repeat
from qmlkit.utils import errors
from qmlkit.utils.errors import did_you_mean, unknown, wrong_size


def test_the_examples_in_these_modules_run() -> None:
    """Their docstrings show output, so the output has to be real."""
    for module in (errors, _aliases):
        result = doctest.testmod(module, verbose=False)
        assert result.attempted > 0, f"{module.__name__} has no runnable examples"
        assert result.failed == 0, f"{module.__name__} has failing examples"


# --------------------------------------------------------------------------- #
# near matching
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("got", "valid", "expected"),
    [
        # separator drift, which is what half-remembering another library looks like
        ("parameter_shift", ["parameter-shift", "adjoint"], "parameter-shift"),
        ("hardware-efficient", ["hardware_efficient", "two_local"], "hardware_efficient"),
        # case drift
        ("Parameter-Shift", ["parameter-shift", "adjoint"], "parameter-shift"),
        ("NumPy", ["numpy", "qiskit"], "numpy"),
        # an ordinary typo
        ("adjiont", ["adjoint", "spsa"], "adjoint"),
        ("zeroes", ["small", "uniform", "zeros"], "zeros"),
    ],
)
def test_drifted_spellings_resolve_to_one_suggestion(
    got: str, valid: list[str], expected: str
) -> None:
    assert did_you_mean(got, valid)[0] == expected


def test_nothing_is_suggested_when_nothing_is_close() -> None:
    """A confident wrong suggestion is worse than none: it costs another round trip."""
    assert did_you_mean("wildly-unrelated", ["adjoint", "spsa"]) == ()


# --------------------------------------------------------------------------- #
# the message itself
# --------------------------------------------------------------------------- #
def test_an_unknown_name_reports_the_guess_and_the_whole_valid_set() -> None:
    text = str(unknown("gradient method", "parameter_shift", ["adjoint", "parameter-shift"]))
    assert "unknown gradient method 'parameter_shift'." in text
    assert "Did you mean 'parameter-shift'?" in text
    assert "Valid: adjoint, parameter-shift." in text


def test_the_exception_type_is_the_callers_to_choose() -> None:
    assert isinstance(unknown("gate", "zz", ["cz"]), ValueError)
    assert isinstance(unknown("gate", "zz", ["cz"], error=KeyError), KeyError)


def test_a_size_error_names_the_edit_that_fixes_it() -> None:
    text = str(
        wrong_size(
            "ZZFeatureMap(4)", 4, 8, unit="feature", hint="Use ZZFeatureMap(8), or PCAReducer(4)."
        )
    )
    assert "expects 4 features, got 8." in text
    assert "PCAReducer(4)" in text


# --------------------------------------------------------------------------- #
# the live registries
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("lookup", "typo", "correction"),
    [
        (qk.get_gate, "cnott", "cnot"),
        (qk.get_backend, "NumPy", "numpy"),
        (qk.get_ansatz, "hardware-efficient", "hardware_efficient"),
    ],
)
def test_registry_lookups_suggest_the_registered_name(lookup, typo: str, correction: str) -> None:
    with pytest.raises((KeyError, ValueError)) as caught:
        lookup(typo)
    assert f"Did you mean '{correction}'" in str(caught.value)


def test_an_unknown_gradient_method_lists_the_ones_that_exist() -> None:
    ansatz = qk.hardware_efficient(2, 1)
    with pytest.raises(KeyError) as caught:
        qk.grad(ansatz.build(), ansatz.init(seed=0), qk.Z(0), method="parameter_shift")
    text = str(caught.value)
    assert "Did you mean 'parameter-shift'" in text
    assert "register_gradient" in text


def test_a_missing_backend_names_the_install_command() -> None:
    """Unavailable is a different answer from unknown, and stays a different answer."""
    if qk.is_available("spinqit"):
        pytest.skip("SpinQit is installed in this interpreter")
    with pytest.raises(qk.BackendNotAvailable, match=re.escape("pip install")):
        qk.get_backend("spinqit")


# --------------------------------------------------------------------------- #
# the vocabulary of the other libraries
# --------------------------------------------------------------------------- #
def test_every_translation_target_still_exists() -> None:
    """The table names real attributes, or this test is the thing that says so."""
    missing = []
    for name, (_source, target, _note) in _aliases.ELSEWHERE.items():
        if target is None:
            continue
        try:
            getattr(qk, target)
        except ImportError:
            continue  # a torch export, absent only because torch is
        except AttributeError:
            missing.append(f"{name} -> {target}")
    assert not missing, f"translations pointing at names that no longer exist: {missing}"


def test_no_translation_shadows_a_real_export() -> None:
    """An entry for a name that already resolves is dead code, and misleading."""
    shadowed = sorted(n for n in _aliases.ELSEWHERE if n in dir(qk))
    assert not shadowed, f"these resolve for real, so their table entries never fire: {shadowed}"


@pytest.mark.parametrize(
    ("foreign", "mentions"),
    [
        ("AngleEmbedding", "AngleFeatureMap"),
        ("AmplitudeEmbedding", "amplitude_encode"),
        ("StronglyEntanglingLayers", "strongly_entangling"),
        ("PauliZ", "qmlkit.Z"),
        ("Hamiltonian", "PauliSum"),
        ("QuantumCircuit", "QCircuit"),
        ("SparsePauliOp", "PauliSum"),
        ("EfficientSU2", "hardware_efficient"),
        ("TorchConnector", "QuantumLayer"),
        ("EstimatorQNN", "QuantumLayer"),
        ("expectation_value", "expectation"),
        ("gradient", "grad"),
        ("transpile", "no transpiler"),
        ("Observable", "PauliSum"),
    ],
)
def test_a_name_from_another_library_is_answered_with_the_qmlkit_one(
    foreign: str, mentions: str
) -> None:
    with pytest.raises(AttributeError, match=re.escape(mentions)):
        getattr(qk, foreign)


def test_the_source_library_is_named_so_the_confusion_is_visible() -> None:
    with pytest.raises(AttributeError, match="PennyLane"):
        _ = qk.AngleEmbedding
    with pytest.raises(AttributeError, match="Qiskit"):
        _ = qk.QuantumCircuit


def test_a_plain_typo_falls_back_to_a_near_match() -> None:
    with pytest.raises(AttributeError, match="hardware_efficient"):
        _ = qk.hardware_eficient


def test_a_name_nobody_could_have_meant_still_raises_plainly() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = qk.totally_made_up_xyz


def test_dir_lists_the_lazily_served_torch_exports() -> None:
    """Introspection is how an agent surveys a module, so it has to be complete."""
    listed = dir(qk)
    assert "VQC" in listed
    assert "QuantumLayer" in listed
    assert "expectation" in listed


# --------------------------------------------------------------------------- #
# diagnose: the failures that return a number instead of raising
# --------------------------------------------------------------------------- #
def _hand_composed_reupload(rotation: str, block_gate: str, layers: int = 3) -> qk.Ansatz:
    """A re-uploading model built out of blocks, which is the case reupload() cannot warn about."""
    fmap = qk.AngleFeatureMap(1, rotation=rotation)
    body = repeat(layers, EncodingLayer(fmap) + qk.RotationLayer(block_gate))
    return qk.Ansatz(1, body, n_inputs=1)


def test_the_diagnostics_examples_run() -> None:
    result = doctest.testmod(diagnostics, verbose=False)
    assert result.attempted > 0
    assert result.failed == 0


def test_a_healthy_ansatz_reports_nothing() -> None:
    report = qk.diagnose(qk.hardware_efficient(3, 2))
    assert not report
    assert report.codes == ()
    assert "nothing to report" in str(report)


def test_the_commuting_encoding_trap_is_caught_when_composed_by_hand() -> None:
    """reupload() warns at construction; blocks composed directly do not, and this is why."""
    report = qk.diagnose(_hand_composed_reupload("ry", "ry"))
    assert "ENCODING_COMMUTES" in report.codes
    assert report.errors[0].fix


def test_a_non_commuting_block_is_left_alone() -> None:
    assert "ENCODING_COMMUTES" not in qk.diagnose(_hand_composed_reupload("ry", "rx")).codes


def test_uploads_are_counted_through_repeat_not_by_tree_node() -> None:
    """``repeat(3, ...)`` is one node and three uploads; counting nodes hides the collapse."""
    report = qk.diagnose(_hand_composed_reupload("ry", "ry", layers=3))
    finding = next(f for f in report if f.code == "ENCODING_COMMUTES")
    assert finding.value == 3.0
    assert "3 uploads" in finding.message


def test_a_weight_that_cannot_move_the_state_is_reported() -> None:
    """Rz on |0> is a global phase, so its angle is optimised over and cannot matter."""
    report = qk.diagnose(qk.Ansatz(2, qk.RotationLayer("rz")))
    assert "DEAD_WEIGHTS" in report.codes


def test_a_live_weight_is_not_called_dead() -> None:
    assert "DEAD_WEIGHTS" not in qk.diagnose(qk.hardware_efficient(3, 2)).codes


def test_an_input_the_model_cannot_feel_is_an_error() -> None:
    report = qk.diagnose(_hand_composed_reupload("rz", "rz", layers=1))
    assert "INPUTS_UNUSED" in report.codes
    assert report.findings[0].severity == "error"


def test_a_circuit_that_only_makes_product_states_is_reported() -> None:
    report = qk.diagnose(qk.Ansatz(3, qk.RotationLayer("ry")))
    assert "NO_ENTANGLEMENT" in report.codes


def test_findings_come_back_worst_first() -> None:
    report = qk.diagnose(_hand_composed_reupload("rz", "rz", layers=1))
    ranks = [{"error": 0, "warning": 1, "info": 2}[f.severity] for f in report]
    assert ranks == sorted(ranks)


def test_a_deep_wide_ansatz_is_priced_in_shots() -> None:
    """Exact gradients hide what the same model costs on a sampling device."""
    report = qk.diagnose(qk.hardware_efficient(10, 20), n_samples=40)
    finding = next(f for f in report if f.code == "FLAT_GRADIENTS")
    assert "shots" in finding.message
    assert finding.value is not None and finding.value < 1e-3


# --------------------------------------------------------------------------- #
# diagnose: Gram matrices
# --------------------------------------------------------------------------- #
def _rbf_gram(n: int = 10, seed: int = 1) -> np.ndarray:
    X = np.random.default_rng(seed).normal(size=(n, 2))
    return np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))


def test_a_healthy_gram_matrix_reports_nothing() -> None:
    assert not qk.diagnose(_rbf_gram())


def test_a_concentrated_gram_matrix_is_an_error() -> None:
    flat = np.full((8, 8), 0.5)
    np.fill_diagonal(flat, 1.0)
    report = qk.diagnose(flat)
    assert "KERNEL_CONCENTRATED" in report.codes
    assert report.errors


def test_a_non_psd_gram_matrix_names_the_repair() -> None:
    report = qk.diagnose(np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]))
    finding = next(f for f in report if f.code == "KERNEL_NOT_PSD")
    assert "closest_psd_matrix" in finding.fix


def test_a_signal_under_the_shot_noise_is_reported() -> None:
    rng = np.random.default_rng(3)
    gram = np.full((10, 10), 0.5) + rng.normal(0, 0.01, (10, 10))
    gram = (gram + gram.T) / 2
    np.fill_diagonal(gram, 1.0)
    assert "KERNEL_UNRESOLVABLE" in qk.diagnose(gram, shots=100).codes
    assert "KERNEL_UNRESOLVABLE" not in qk.diagnose(gram, shots=10_000_000).codes


def test_a_gram_matrix_has_to_be_square() -> None:
    with pytest.raises(ValueError, match="must be square"):
        qk.diagnose(np.zeros((3, 5)))


# --------------------------------------------------------------------------- #
# diagnose: what it accepts
# --------------------------------------------------------------------------- #
def test_diagnose_finds_the_ansatz_inside_a_torch_model() -> None:
    """A model is whatever holds an ansatz, however deeply nested."""
    torch = pytest.importorskip("torch")
    layer = qk.QuantumLayer(qk.ZZFeatureMap(4), qk.hardware_efficient(4, 2), [qk.Z(0)])
    net = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Tanh(), layer)
    assert "hardware_efficient" in str(qk.diagnose(net))
    assert "hardware_efficient" in str(qk.diagnose(qk.VQC(n_features=4, n_classes=2)))


def test_diagnose_says_what_it_takes_when_given_something_else() -> None:
    with pytest.raises(TypeError) as caught:
        qk.diagnose("an ansatz, surely")
    text = str(caught.value)
    assert "cannot inspect a str" in text
    assert "Ansatz" in text and "Gram matrix" in text


def test_a_report_behaves_like_the_sequence_it_is() -> None:
    report = qk.diagnose(qk.Ansatz(2, qk.RotationLayer("rz")))
    assert bool(report) is True
    assert len(report) == len(report.codes) == len(list(report))
    assert report[0] is report.findings[0]
    assert set(report.errors) | set(report.warnings) <= set(report.findings)
