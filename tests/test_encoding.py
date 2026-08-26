"""Encoding tests.

The feature-map tests check against the *analytic kernel each map induces* rather
than against a golden circuit. That is the property that actually matters for
learning, and it catches convention errors a gate-count assertion would sail past.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.encoding.amplitude import (
    pad_to_power_of_two,
    state_preparation_angles,
    uniformly_controlled_rotation,
)
from qmlkit.encoding.feature_maps import basis_change, default_data_map, pauli_terms


def fidelity(spec_a, spec_b) -> float:
    """|<psi_a|psi_b>|^2 — the quantity a fidelity kernel estimates."""
    a = qk.statevector(spec_a)
    b = qk.statevector(spec_b)
    return float(abs(np.vdot(a, b)) ** 2)


# --------------------------------------------------------------------------- #
# amplitude encoding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "vec",
    [
        [1, 2, 3, 4],  # the Lecture 3 example
        [1, 0, 0, 0],
        [0, 1],
        [1, -2, 3, -4],  # signed
        [-1, -1, -1, -1],
        [1, 0, 0, 0, 0, 0, 0, 1],  # sparse
        [5.0],  # single element
        [1, 2, 3],  # padded to 4
    ],
)
def test_amplitude_encoding_prepares_the_target_state(vec):
    target = pad_to_power_of_two(vec)
    target = target / np.linalg.norm(target)
    got = qk.statevector(qk.amplitude_encode(vec))
    assert abs(np.vdot(target, got)) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_amplitude_encoding_handles_complex_amplitudes(n):
    rng = np.random.default_rng(n)
    vec = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    target = vec / np.linalg.norm(vec)
    got = qk.statevector(qk.amplitude_encode(vec))
    assert abs(np.vdot(target, got)) == pytest.approx(1.0, abs=1e-10)


def test_amplitude_encoding_uses_only_portable_gates():
    """No backend-specific state-prep primitive: it must run anywhere."""
    counts = qk.amplitude_encode(np.arange(1, 9, dtype=float)).gate_counts()
    assert set(counts) <= {"ry", "rz", "cx"}


def test_amplitude_encoding_check_flag_validates_the_result():
    qk.amplitude_encode([1, 2, 3, 4], check=True)  # must not raise


def test_amplitude_encoding_rejects_degenerate_input():
    with pytest.raises(ValueError, match="zero vector"):
        qk.amplitude_encode([0, 0, 0, 0])
    with pytest.raises(ValueError, match="empty vector"):
        qk.amplitude_encode([])
    with pytest.raises(ValueError, match="not a power of two"):
        qk.amplitude_encode([1, 2, 3], pad=False)
    with pytest.raises(ValueError, match="norm"):
        qk.amplitude_encode([1, 2, 3, 4], normalize=False)


def test_amplitude_encoding_cost_grows_exponentially():
    """Loading 2**n numbers costs O(2**n) gates - the price should be visible."""
    sizes = [len(qk.amplitude_encode(np.arange(1, 2**n + 1, dtype=float)).ops) for n in (2, 3, 4)]
    assert sizes == sorted(sizes)
    assert sizes[-1] > 2 * sizes[0]


def test_uniformly_controlled_rotation_applies_the_right_angle_per_branch():
    """The recursion is the load-bearing part; check each control branch directly."""
    angles = np.array([0.3, 1.1, -0.7, 2.0])
    for k, expected in enumerate(angles):
        qc = qk.QCircuit(3)
        for bit, q in zip(format(k, "02b"), (0, 1), strict=True):
            if bit == "1":
                qc.x(q)
        uniformly_controlled_rotation(qc, "ry", angles, [0, 1], 2)
        psi = qk.statevector(qc.to_spec()).reshape(2, 2, 2)
        branch = psi[int(format(k, "02b")[0]), int(format(k, "02b")[1])]
        assert branch[0] == pytest.approx(np.cos(expected / 2), abs=1e-10)
        assert branch[1] == pytest.approx(np.sin(expected / 2), abs=1e-10)


def test_state_preparation_angles_skip_the_phase_cascade_for_real_data():
    _, rz = state_preparation_angles(np.array([0.5, 0.5, 0.5, 0.5], dtype=complex))
    assert rz == []


# --------------------------------------------------------------------------- #
# the helpers Lecture 3 references but never defines
# --------------------------------------------------------------------------- #
def test_default_data_map_matches_the_standard_convention():
    x = np.array([0.3, 1.2, 2.0])
    assert default_data_map(x, (1,)) == pytest.approx(1.2)
    assert default_data_map(x, (0, 1)) == pytest.approx((np.pi - 0.3) * (np.pi - 1.2))
    assert default_data_map(x, (0, 1, 2)) == pytest.approx(
        (np.pi - 0.3) * (np.pi - 1.2) * (np.pi - 2.0)
    )


def test_basis_change_gates():
    assert basis_change("Z") == ((), ())
    assert basis_change("X") == (("h",), ("h",))
    assert basis_change("Y") == (("sdg", "h"), ("h", "s"))
    with pytest.raises(ValueError, match="unknown Pauli"):
        basis_change("Q")


def test_basis_change_actually_diagonalises_the_pauli():
    """W P W^dagger = Z, checked by preparing a P eigenstate and reading Z."""
    for pauli, prep in (("X", "ry"), ("Y", "rx")):
        qc = qk.QCircuit(1)
        qc.apply(prep, 0, np.pi / 2 if pauli == "X" else -np.pi / 2)
        before = qk.expectation(qc.to_spec(), getattr(qk, pauli)(0))
        for gate in basis_change(pauli)[0]:
            qc.apply(gate, 0)
        after = qk.expectation(qc.to_spec(), qk.Z(0))
        assert after == pytest.approx(before, abs=1e-10)


def test_pauli_terms_expansion():
    assert pauli_terms(("Z",), 3) == [((0,), "Z"), ((1,), "Z"), ((2,), "Z")]
    zz = pauli_terms(("ZZ",), 3, "linear")
    assert [idx for idx, _ in zz] == [(0, 1), (1, 2)]
    ring = pauli_terms(("ZZ",), 3, "ring")
    assert [idx for idx, _ in ring] == [(0, 1), (1, 2), (2, 0)]
    assert [idx for idx, _ in pauli_terms(("ZZZ",), 3)] == [(0, 1, 2)]
    with pytest.raises(ValueError, match="empty Pauli"):
        pauli_terms(("",), 2)


def test_pauli_feature_map_runs_at_all():
    """Lecture 3's version cannot: it calls _basis and _phi, which do not exist."""
    fm = qk.PauliFeatureMap(3, paulis=("Z", "ZZ"), reps=2)
    spec = fm.build([0.3, 0.8, 1.4])
    assert spec.n_qubits == 3
    assert qk.statevector(spec).shape == (8,)


# --------------------------------------------------------------------------- #
# feature maps, checked against the kernels they induce
# --------------------------------------------------------------------------- #
def test_angle_map_kernel_is_cos_squared_half_delta():
    """The one-qubit analytic result the whole of Lecture 4 is built on."""
    fm = qk.AngleFeatureMap(1, entangle=False)
    for x, xp in [(0.0, 0.0), (0.0, np.pi / 2), (0.0, np.pi), (0.4, 1.9)]:
        k = fidelity(fm.build([x]), fm.build([xp]))
        assert k == pytest.approx(np.cos((x - xp) / 2) ** 2, abs=1e-10)


def test_unentangled_angle_map_kernel_factorises():
    """No entangler means the kernel is a product over features."""
    fm = qk.AngleFeatureMap(3, entangle=False)
    x = np.array([0.2, 1.1, 2.4])
    xp = np.array([0.9, 0.3, 1.7])
    expected = np.prod(np.cos((x - xp) / 2) ** 2)
    assert fidelity(fm.build(x), fm.build(xp)) == pytest.approx(expected, abs=1e-10)


def test_z_feature_map_kernel_factorises_too():
    """The Z map has no entanglers, so its kernel is a product of cos^2 terms."""
    fm = qk.ZFeatureMap(3, reps=1)
    x = np.array([0.2, 1.1, 2.4])
    xp = np.array([0.9, 0.3, 1.7])
    expected = np.prod(np.cos(x - xp) ** 2)
    assert fidelity(fm.build(x), fm.build(xp)) == pytest.approx(expected, abs=1e-10)


def test_zz_map_kernel_does_not_factorise():
    """The entangling terms are the point: the kernel must stop being a product."""
    x = np.array([0.4, 1.3])
    xp = np.array([1.9, 0.6])
    z_k = fidelity(qk.ZFeatureMap(2, reps=1).build(x), qk.ZFeatureMap(2, reps=1).build(xp))
    zz = qk.ZZFeatureMap(2, reps=1)
    zz_k = fidelity(zz.build(x), zz.build(xp))
    assert abs(zz_k - z_k) > 1e-3


@pytest.mark.parametrize(
    "fm",
    [
        qk.ZFeatureMap(2),
        qk.ZZFeatureMap(2),
        qk.AngleFeatureMap(2),
        qk.PauliFeatureMap(2, paulis=("Z", "X", "ZZ")),
        qk.PauliFeatureMap(3, paulis=("Y", "ZZ"), entanglement="ring"),
    ],
)
def test_every_feature_map_has_unit_self_kernel(fm):
    """k(x, x) = 1 for any fidelity kernel — a cheap check that adjoint() is right."""
    rng = np.random.default_rng(4)
    x = rng.uniform(0, np.pi, fm.n_features)
    assert fidelity(fm.build(x), fm.build(x)) == pytest.approx(1.0, abs=1e-10)
    combined = fm.build(x).compose(fm.adjoint(x))
    assert abs(qk.statevector(combined)[0]) == pytest.approx(1.0, abs=1e-10)


def test_feature_map_validates_its_input_width():
    fm = qk.ZZFeatureMap(3)
    with pytest.raises(ValueError, match="expects 3 features"):
        fm.build([0.1, 0.2])


def test_feature_map_rejects_degenerate_configuration():
    with pytest.raises(ValueError, match="n_features must be at least 1"):
        qk.ZZFeatureMap(0)
    with pytest.raises(ValueError, match="reps must be at least 1"):
        qk.PauliFeatureMap(2, reps=0)


def test_reps_increase_depth_but_not_width():
    one = qk.ZZFeatureMap(3, reps=1).resources()
    two = qk.ZZFeatureMap(3, reps=2).resources()
    assert two["n_qubits"] == one["n_qubits"]
    assert two["n_ops"] == 2 * one["n_ops"]


# --------------------------------------------------------------------------- #
# Hamiltonian encoding
# --------------------------------------------------------------------------- #
def test_trotter_angles_match_the_lecture_formulas():
    assert qk.encoding.trotter_rz_angle(0.7, 4.0, 8) == pytest.approx(2 * 0.7 * 4.0 / 8)
    assert qk.encoding.trotter_zz_angle(0.7, 1.3, 4.0, 8) == pytest.approx(2 * 0.7 * 1.3 * 4.0 / 8)


def test_hamiltonian_encoding_is_exact_at_any_step_count():
    """Every term is diagonal, so they commute and the Trotter split is exact.

    More steps buy depth and nothing else — worth pinning so nobody tunes `steps`
    hoping for accuracy.
    """
    x = [0.7, 1.3, 0.4]
    ref = qk.statevector(qk.hamiltonian_encode(x, t=1.5, steps=1))
    for steps in (2, 3, 7):
        got = qk.statevector(qk.hamiltonian_encode(x, t=1.5, steps=steps))
        assert abs(np.vdot(ref, got)) == pytest.approx(1.0, abs=1e-10)


def test_hamiltonian_encoding_without_hadamards_only_adds_a_global_phase():
    """A Z-diagonal evolution on |0...0> is unobservable — hence the H layer."""
    probs = qk.probabilities(qk.hamiltonian_encode([0.7, 1.3], initial_hadamard=False))
    assert probs[0] == pytest.approx(1.0, abs=1e-12)


def test_hamiltonian_encoding_depends_on_the_data():
    a = qk.statevector(qk.hamiltonian_encode([0.7, 1.3]))
    b = qk.statevector(qk.hamiltonian_encode([1.1, 0.2]))
    assert abs(np.vdot(a, b)) < 0.999


def test_hamiltonian_encoding_validates_arguments():
    with pytest.raises(ValueError, match="at least one feature"):
        qk.hamiltonian_encode([])
    with pytest.raises(ValueError, match="steps must be at least 1"):
        qk.hamiltonian_encode([0.5], steps=0)


# --------------------------------------------------------------------------- #
# data re-uploading
# --------------------------------------------------------------------------- #
def test_reupload_parameter_shape_and_frequency_count():
    enc = qk.DataReuploadEncoder(2, n_uploads=3, rotations=("rz", "ry", "rz"))
    assert enc.weight_shape == (3, 2, 3)
    assert enc.n_weights == 18
    assert enc.n_params == 18
    assert enc.n_frequencies == 4  # frequencies 0..3


def test_reupload_widens_the_reachable_spectrum():
    """L uploads reach frequencies 0..L. Measured by FFT of <Z> over x.

    The trainable block must NOT commute with the encoding rotation. With
    ``rotations=("ry",)`` against an ``ry`` encoding the uploads merge into a single
    rotation, so this would measure the collapsed case and pass for the wrong reason.
    """
    seen = []
    for uploads in (1, 2, 3):
        enc = qk.DataReuploadEncoder(
            1, n_uploads=uploads, rotations=("rz", "ry", "rz"), entanglement=None
        )
        rng = np.random.default_rng(0)
        theta = rng.uniform(-np.pi, np.pi, enc.n_weights)
        xs = np.linspace(0, 2 * np.pi, 128, endpoint=False)
        vals = [qk.expectation(enc.build([x]), qk.Z(0), theta=theta) for x in xs]
        amps = np.abs(np.fft.rfft(vals)) / len(xs)
        highest = int(np.max(np.nonzero(amps > 1e-6)))
        seen.append(highest)
    assert seen == sorted(seen)
    assert seen[-1] > seen[0], f"more uploads must reach higher frequencies, got {seen}"


def test_reupload_trainable_input_exposes_input_gradients():
    """trainable_input=True is what gives a classical pre-net something to learn from."""
    enc = qk.DataReuploadEncoder(1, n_uploads=2, rotations=("rz", "ry", "rz"), trainable_input=True)
    spec = enc.build()
    assert spec.n_params == enc.n_weights + 1
    theta = np.concatenate([np.zeros(enc.n_weights), [0.4]])
    grad = qk.param_shift_grad_circuit(spec, theta, qk.Z(0))
    assert grad[-1] != pytest.approx(0.0, abs=1e-9)  # df/dx exists


def test_reupload_requires_x_unless_input_is_trainable():
    enc = qk.DataReuploadEncoder(2, n_uploads=1)
    with pytest.raises(ValueError, match="x is required"):
        enc.build()
    with pytest.raises(ValueError, match="expected 2 features"):
        enc.build([0.1])


def test_reupload_rejects_degenerate_configuration():
    with pytest.raises(ValueError, match="n_uploads must be at least 1"):
        qk.DataReuploadEncoder(2, n_uploads=0)


# --------------------------------------------------------------------------- #
# scaling and dimensionality reduction
# --------------------------------------------------------------------------- #
def test_angle_scaler_reuses_the_training_range():
    """Rescaling test data to its own extremes silently breaks comparability."""
    train = np.array([[0.0], [10.0]])
    test = np.array([[5.0]])
    scaler = qk.AngleScaler(0.0, 2 * np.pi).fit(train)
    assert scaler.transform(test)[0, 0] == pytest.approx(np.pi)
    # ...whereas a fresh fit on the test point alone would put it mid-range
    assert qk.to_angle_range(test, 0.0, 2 * np.pi)[0, 0] == pytest.approx(np.pi)


def test_angle_scaler_must_be_fitted():
    with pytest.raises(ValueError, match="must be fitted"):
        qk.AngleScaler().transform(np.array([[1.0]]))


def test_pca_reducer_keeps_the_leading_variance():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(60, 2))
    x = np.column_stack([base, base @ rng.normal(size=(2, 5))])  # rank 2
    reducer = qk.PCAReducer(2).fit(x)
    assert reducer.explained_variance_ratio_.sum() == pytest.approx(1.0, abs=1e-8)
    assert reducer.transform(x).shape == (60, 2)


def test_pca_reducer_must_be_fitted_and_cannot_grow_dimensions():
    with pytest.raises(ValueError, match="must be fitted"):
        qk.PCAReducer(2).transform(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="cannot reduce"):
        qk.PCAReducer(9).fit(np.zeros((3, 4)))


def test_reduce_to_qubits_produces_encodable_angles():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(40, 12)) * 100
    reduced = qk.reduce_to_qubits(x, 3)
    assert reduced.shape == (40, 3)
    assert reduced.min() >= 0.0 and reduced.max() <= 2 * np.pi
    spec = qk.angle_encode(reduced[0])
    assert spec.n_qubits == 3


def test_reduce_to_qubits_truncate_mode_and_validation():
    x = np.arange(12, dtype=float).reshape(3, 4)
    assert qk.reduce_to_qubits(x, 2, method="truncate", to_angles=False).shape == (3, 2)
    with pytest.raises(ValueError, match="unknown method"):
        qk.reduce_to_qubits(x, 2, method="magic")
    with pytest.raises(ValueError, match="n_qubits must be at least 1"):
        qk.reduce_to_qubits(x, 0)
    with pytest.raises(ValueError, match="only 4 features"):
        qk.reduce_to_qubits(x, 9, method="truncate")
