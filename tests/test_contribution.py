"""Did the quantum layer earn its place?

The complaint this answers is the most common one in the field and no library
answers it: "it trains, and it works just as well with the quantum part removed."

The discipline here is the one the earlier diagnostics got wrong. Each test asserts
the finding is *true* -- the probe really does score worse on what the layer returned
than on what it received -- not merely that the code path fired. A false positive in
this layer teaches people to ignore it, which is worse than staying quiet.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import qmlkit as qk  # noqa: E402
from qmlkit.diagnostics import _probe_score  # noqa: E402
from qmlkit.nn import QuantumLayer  # noqa: E402

pytestmark = pytest.mark.torch

N_QUBITS = 4


def _data(n: int = 120, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Feature 0 decides the label; features 1..3 are noise.

    Angles live in ``[0, pi]`` so that ``<Z> = cos(x)`` is monotonic over the range
    and a single qubit read-out *can* separate the classes. Sampling symmetrically
    about zero would make ``cos`` even, and then no read-out could -- which would
    make the healthy case fail for a reason that has nothing to do with the model.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, np.pi, size=(n, N_QUBITS))
    y = (X[:, 0] > np.pi / 2).astype(int)
    return X, y


def _layer(read_qubit: int) -> QuantumLayer:
    """A layer with no entangler, so qubit ``i`` carries feature ``i`` and nothing else.

    That isolation is what makes the test's claim provable rather than probable:
    reading qubit 3 cannot pick up feature 0 through an entangling gate, because
    there is no entangling gate.
    """
    feature_map = qk.AngleFeatureMap(N_QUBITS, rotation="ry")
    ansatz = qk.Ansatz(N_QUBITS, qk.RotationLayer("ry"))
    return QuantumLayer(feature_map, ansatz, [qk.Z(read_qubit)])


def test_reading_a_noise_qubit_is_caught_and_the_finding_is_true() -> None:
    """The layer discards the only informative feature, and the report says so."""
    X, y = _data()
    model = _layer(read_qubit=3)  # feature 3 is noise

    report = qk.diagnose(model, X, y)
    assert "QUANTUM_LAYER_BYPASSED" in report.codes

    # ...and the claim it makes is independently true: probe the activations directly.
    with torch.no_grad():
        returned = model(torch.as_tensor(X)).numpy()
    on_input = _probe_score(X, y, 0)
    on_output = _probe_score(returned, y, 0)
    assert on_input is not None and on_output is not None
    assert on_output[0] < on_input[0], "the finding fired without the drop being real"


def test_reading_the_informative_qubit_is_left_alone() -> None:
    """A layer that keeps the signal must not be accused of losing it."""
    X, y = _data()
    model = _layer(read_qubit=0)  # feature 0 is the label

    report = qk.diagnose(model, X, y)
    assert "QUANTUM_LAYER_BYPASSED" not in report.codes

    with torch.no_grad():
        returned = model(torch.as_tensor(X)).numpy()
    on_output = _probe_score(returned, y, 0)
    assert on_output is not None
    assert on_output[0] > 0.8, "the healthy case is only meaningful if the signal survived"


def test_silent_without_data() -> None:
    """The structural checks still run alone; this one needs X and y to say anything."""
    report = qk.diagnose(_layer(read_qubit=3))
    assert "QUANTUM_LAYER_BYPASSED" not in report.codes


def test_silent_on_a_continuous_target() -> None:
    """The probe is a classifier, so it declines a regression target rather than guessing."""
    X, _ = _data()
    rng = np.random.default_rng(1)
    y_continuous = rng.normal(size=len(X))
    report = qk.diagnose(_layer(read_qubit=3), X, y_continuous)
    assert "QUANTUM_LAYER_BYPASSED" not in report.codes


def test_probe_is_deterministic_for_a_seed() -> None:
    """Two runs of the same diagnosis must agree, or the verdict is not reportable."""
    X, y = _data()
    model = _layer(read_qubit=3)
    assert qk.diagnose(model, X, y).codes == qk.diagnose(model, X, y).codes
