"""SpinQit-specific behaviour.

Kept out of ``test_cross_backend.py`` because that module parametrises every test
over all backends; these target one. Skipped wherever SpinQit is not importable,
which is every interpreter above Python 3.10.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.registry import is_available

pytestmark = [
    pytest.mark.spinqit,
    pytest.mark.skipif(not is_available("spinqit"), reason="spinqit not installed"),
]


def _layered_spec() -> qk.CircuitSpec:
    qc = qk.QCircuit(3)
    qc.rotation_layer(("ry", "rz")).entangle("ring").rotation_layer(("ry",))
    spec = qc.to_spec()
    rng = np.random.default_rng(17)
    return spec.bind(rng.uniform(-np.pi, np.pi, spec.n_params))


def test_spinqit_convention_self_check_passes():
    """The backend's own guard against a SpinQit release changing a convention."""
    results = qk.get_backend("spinqit").verify_conventions()
    failed = [name for name, ok in results.items() if not ok]
    assert not failed, f"SpinQit conventions changed: {failed}"
    assert len(results) >= 5


def test_spinqit_native_sampling_agrees_with_shared_sampling():
    """native_sampling=True hands shot-drawing to SpinQit; it is unseeded but valid."""
    spec = _layered_spec()
    native = qk.get_backend("spinqit", native_sampling=True)
    counts = native.counts(spec, shots=20_000)
    assert sum(counts.values()) == 20_000
    assert all(len(k) == spec.n_qubits for k in counts)

    exact = qk.get_backend("numpy").probabilities(spec)
    empirical = np.zeros_like(exact)
    for bits, n in counts.items():
        empirical[int(bits, 2)] = n / 20_000
    assert np.abs(empirical - exact).max() < 0.02


def test_spinqit_repr_reports_its_configuration():
    be = qk.get_backend("spinqit", native_sampling=True)
    assert "native_sampling=True" in repr(be)
    assert "compiler='native'" in repr(be)


def test_spinqit_rejects_an_unmapped_gate():
    qk.register_gate(qk.GateDef("exotic", 1, 0, lambda: np.eye(2, dtype=complex)))
    spec = qk.CircuitSpec(1, (qk.Op("exotic", (0,)),))
    with pytest.raises(NotImplementedError, match="no SpinQit mapping"):
        qk.get_backend("spinqit").statevector(spec)
