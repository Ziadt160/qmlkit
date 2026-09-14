"""The data-loading cost model, and the QRAM comparison it exists to make.

The estimate above 512 amplitudes is extrapolated from measured constants, so the
tests that matter are the ones checking the extrapolation against a circuit actually
built and counted.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.encoding import loading_cost, qram_cost
from qmlkit.encoding.amplitude import amplitude_encode


def test_amplitude_encoding_compresses_into_log_qubits():
    assert loading_cost(1024, "amplitude").n_qubits == 10
    assert loading_cost(1000, "amplitude").n_qubits == 10  # padded up to 1024


def test_angle_encoding_costs_a_qubit_per_feature_and_no_entanglers():
    cost = loading_cost(64, "angle")
    assert cost.n_qubits == 64
    assert cost.two_qubit_gates == 0
    assert cost.depth == 1


def _generic(n: int) -> np.ndarray:
    """A vector with signs. np.ones costs half, because it needs no phase preparation."""
    return np.random.default_rng(1).normal(size=n)


def test_counted_costs_match_the_circuit_they_describe():
    """Below the extrapolation cutoff the model builds the circuit, so it cannot drift."""
    for n in (16, 64, 256):
        cost = loading_cost(n, "amplitude")
        assert cost.exact
        built = amplitude_encode(_generic(n)).resources()
        # The gate count is structural and identical for any generic vector; depth
        # shifts a percent or two with the particular angles, so it is approximate.
        assert cost.two_qubit_gates == built["n_2q"]
        assert cost.depth == pytest.approx(built["depth"], rel=0.05)


def test_the_model_prices_a_generic_vector_not_the_cheapest_one():
    """A real non-negative vector needs no phases and costs half. Quoting that as the
    price would understate every budget built on it."""
    cheap = amplitude_encode(np.ones(256)).resources()["n_2q"]
    generic = amplitude_encode(_generic(256)).resources()["n_2q"]
    assert generic == 2 * cheap
    assert loading_cost(256, "amplitude").two_qubit_gates == generic


def test_the_extrapolation_is_within_a_few_percent_of_a_real_circuit():
    """1024 amplitudes is past the cutoff, so the model estimates. Check it against
    the circuit anyway -- measured 4,052 CNOTs and depth 6,027."""
    cost = loading_cost(1024, "amplitude")
    assert not cost.exact
    built = amplitude_encode(_generic(1024)).resources()
    assert cost.two_qubit_gates == pytest.approx(built["n_2q"], rel=0.05)
    assert cost.depth == pytest.approx(built["depth"], rel=0.05)


def test_the_estimate_errs_high_rather_than_low():
    """An underestimate of a cost is the dangerous direction for a budget."""
    built = amplitude_encode(_generic(1024)).resources()
    assert loading_cost(1024, "amplitude").two_qubit_gates >= built["n_2q"]


def test_amplitude_loading_is_linear_in_the_features():
    """The claim the module is built on: doubling the data doubles the gates."""
    small = loading_cost(128, "amplitude").two_qubit_gates
    large = loading_cost(256, "amplitude").two_qubit_gates
    assert large == pytest.approx(2 * small, rel=0.15)


def test_qram_needs_a_qubit_per_stored_value():
    """The log-depth claim is about depth; the memory is still Theta(N)."""
    cost = qram_cost(1024)
    assert cost.address_qubits == 10
    assert cost.memory_qubits == 1024
    assert cost.ideal_depth == 10
    assert cost.total_qubits > 2 * cost.n_addresses - 1


def test_qram_report_names_the_encoding_alternative():
    text = str(qram_cost(256))
    assert "no such device exists" in text
    assert str(qram_cost(256).encoding.n_qubits) in text


def test_an_unknown_method_says_what_is_allowed():
    with pytest.raises(Exception, match="amplitude"):
        loading_cost(16, "qram")


@pytest.mark.parametrize("bad", [0, -1])
def test_a_nonpositive_size_is_refused(bad):
    with pytest.raises(ValueError):
        loading_cost(bad)
    with pytest.raises(ValueError):
        qram_cost(bad)


def test_exports_are_reachable_from_the_top_level():
    assert qk.loading_cost(64, "angle").n_qubits == 64
    assert qk.qram_cost(64).memory_qubits == 64
