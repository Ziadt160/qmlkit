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

import pytest

import qmlkit as qk
from qmlkit import _aliases
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
        getattr(qk, "AngleEmbedding")
    with pytest.raises(AttributeError, match="Qiskit"):
        getattr(qk, "QuantumCircuit")


def test_a_plain_typo_falls_back_to_a_near_match() -> None:
    with pytest.raises(AttributeError, match="hardware_efficient"):
        getattr(qk, "hardware_eficient")


def test_a_name_nobody_could_have_meant_still_raises_plainly() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        getattr(qk, "totally_made_up_xyz")


def test_dir_lists_the_lazily_served_torch_exports() -> None:
    """Introspection is how an agent surveys a module, so it has to be complete."""
    listed = dir(qk)
    assert "VQC" in listed
    assert "QuantumLayer" in listed
    assert "expectation" in listed
