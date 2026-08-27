"""The reproducibility record, and the four-routes-agree self-check.

The self-check is the interesting one. Its claim is that four independent exact
gradient methods agreeing to machine precision is evidence of correctness, and that
disagreement is caught rather than returned. Both halves are tested: a correct
circuit produces an empty report, and a gate whose declared generator frequency is
wrong produces an error finding — which is exactly the failure mode that returns a
plausible number instead of raising.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.provenance import fingerprint, selfcheck


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_records_the_stack_that_decides_the_number():
    got = fingerprint(seed=7)
    assert got.qmlkit == qk.__version__
    assert got.numpy == np.__version__
    assert got.seed == 7
    assert got.default_backend in qk.list_backends()
    assert set(got.backends) == {"qiskit", "cirq", "spinqit"}


def test_fingerprint_is_json_serialisable():
    import json

    payload = json.dumps(fingerprint(seed=1, shots=1024).as_dict())
    assert "qmlkit" in json.loads(payload)


def test_extras_are_carried_verbatim_and_printed():
    got = fingerprint(shots=2048, ansatz="hardware_efficient")
    assert got.extra["shots"] == 2048
    assert got.as_dict()["ansatz"] == "hardware_efficient"
    text = str(got)
    assert "2048" in text and "hardware_efficient" in text


def test_missing_optional_packages_are_recorded_as_absent_not_omitted():
    got = fingerprint()
    # every probed name has an entry, whether or not it is installed
    assert set(got.optional) == {"torch", "sklearn", "matplotlib"}
    assert all(v is None or isinstance(v, str) for v in got.optional.values())


# --------------------------------------------------------------------------- #
# selfcheck
# --------------------------------------------------------------------------- #
def test_a_correct_circuit_reports_nothing():
    ansatz = qk.hardware_efficient(3, 2)
    report = selfcheck(ansatz.build(), np.full(ansatz.n_params, 0.3), qk.Z(0))
    assert not report
    assert report.codes == ()


@pytest.mark.parametrize("seed", range(4))
def test_random_circuits_agree_across_every_route(seed):
    rng = np.random.default_rng(seed)
    ansatz = qk.strongly_entangling(3, 2)
    theta = rng.uniform(-np.pi, np.pi, ansatz.n_params)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
    assert not selfcheck(ansatz.build(), theta, obs)


def test_a_wrong_generator_frequency_is_caught_rather_than_returned():
    """The exact failure the module exists for: a plausible number, no exception."""

    def matrix(t):
        return np.array(
            [[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex
        )

    def dmatrix(t):
        return 0.5 * np.array(
            [[-np.sin(t / 2), -np.cos(t / 2)], [np.cos(t / 2), -np.sin(t / 2)]], dtype=complex
        )

    # the true frequency is 0.5; declaring 2.0 gives parameter-shift the wrong rule
    # while adjoint, which differentiates the matrix, stays correct
    qk.register_gate(
        qk.GateDef(
            "wrong_freq_ry",
            n_qubits=1,
            n_params=1,
            matrix=matrix,
            frequencies=(2.0,),
            dmatrix=dmatrix,
        )
    )
    qc = qk.QCircuit(1)
    qc.apply("wrong_freq_ry", 0, qc.param())

    report = selfcheck(qc.to_spec(), [0.4], qk.Z(0), cross_backend=False)
    assert "selfcheck.gradient-disagreement" in report.codes
    finding = next(f for f in report if f.code == "selfcheck.gradient-disagreement")
    assert finding.severity == "error"
    assert finding.value is not None and finding.value > 1e-3
    assert "frequencies" in finding.fix


def test_cross_backend_agreement_runs_over_every_installed_sdk():
    installed = [n for n in qk.available_backends() if n != "numpy"]
    if not installed:
        pytest.skip("no second backend installed")
    ansatz = qk.hardware_efficient(2, 2)
    report = selfcheck(ansatz.build(), np.full(ansatz.n_params, 0.4), qk.Z(0))
    assert "selfcheck.backend-disagreement" not in report.codes


def test_cross_backend_can_be_skipped():
    ansatz = qk.hardware_efficient(2, 1)
    report = selfcheck(
        ansatz.build(), np.full(ansatz.n_params, 0.4), qk.Z(0), cross_backend=False
    )
    assert "selfcheck.backend-disagreement" not in report.codes


def test_a_parameterless_circuit_says_it_checked_nothing():
    spec = qk.angle_encode([0.3, 0.8])
    report = selfcheck(spec, [], qk.Z(0), cross_backend=False)
    assert "selfcheck.one-route" in report.codes
    assert next(iter(report)).severity == "info"  # not an error: nothing is wrong
