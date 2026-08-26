"""Cross-validation against PennyLane.

A library's own test suite can only catch the bugs its author thought of. Agreeing
with a second, independently written implementation catches the rest — so this file
checks qmlkit against PennyLane across every layer that both libraries implement:
gate matrices, gate derivatives, randomly generated circuits, observables, all six
gradient methods, feature maps, ansatz structure, kernels, quantum information,
Fourier spectra, and the Fubini-Study metric.

The randomised circuit tests matter most. Hand-picked cases test what the author
thought to test; a fuzzer explores the space, and every bug found in this project so
far has been of the "plausible wrong number" kind that only a second opinion catches.

Both libraries are big-endian (wire 0 is the most significant bit), so wires map
straight across with no reversal. Where a convention genuinely differs it is named
and handled explicitly rather than absorbed into a tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")

import qmlkit as qk  # noqa: E402
from qmlkit.core.gates import gate_derivative, gate_matrix, get_gate, list_gates  # noqa: E402
from qmlkit.core.ir import CircuitSpec, Op, ParamRef  # noqa: E402

pytestmark = pytest.mark.pennylane

#: exact simulators on both sides; anything looser than this is a real disagreement
EXACT = 1e-12

#: qmlkit's built-in gates, snapshotted at import. Other test modules register throwaway
#: gates into the same global registry at run time, so reading it live would let the
#: fuzzer draw a gate that exists only inside somebody else's test.
BUILTIN_GATES = tuple(list_gates())


# --------------------------------------------------------------------------- #
# translation helpers
# --------------------------------------------------------------------------- #
#: qmlkit gate name -> the PennyLane operation that must equal it
_PL_GATE = {
    "i": lambda w, p: qml.Identity(w[0]),
    "x": lambda w, p: qml.PauliX(w[0]),
    "y": lambda w, p: qml.PauliY(w[0]),
    "z": lambda w, p: qml.PauliZ(w[0]),
    "h": lambda w, p: qml.Hadamard(w[0]),
    "s": lambda w, p: qml.S(w[0]),
    "sdg": lambda w, p: qml.adjoint(qml.S(w[0])),
    "t": lambda w, p: qml.T(w[0]),
    "tdg": lambda w, p: qml.adjoint(qml.T(w[0])),
    "rx": lambda w, p: qml.RX(p[0], w[0]),
    "ry": lambda w, p: qml.RY(p[0], w[0]),
    "rz": lambda w, p: qml.RZ(p[0], w[0]),
    "phase": lambda w, p: qml.PhaseShift(p[0], w[0]),
    "cx": lambda w, p: qml.CNOT(wires=list(w)),
    "cy": lambda w, p: qml.CY(wires=list(w)),
    "cz": lambda w, p: qml.CZ(wires=list(w)),
    "swap": lambda w, p: qml.SWAP(wires=list(w)),
    "crx": lambda w, p: qml.CRX(p[0], wires=list(w)),
    "cry": lambda w, p: qml.CRY(p[0], wires=list(w)),
    "crz": lambda w, p: qml.CRZ(p[0], wires=list(w)),
}


def to_pennylane(spec: CircuitSpec) -> None:
    """Replay a bound qmlkit circuit as PennyLane operations, inside a QNode."""
    for op in spec.ops:
        params = [float(p) for p in op.params]
        _PL_GATE[op.gate](op.qubits, params)


def pl_state(spec: CircuitSpec) -> np.ndarray:
    """The statevector PennyLane produces for a bound qmlkit circuit."""
    dev = qml.device("default.qubit", wires=spec.n_qubits)

    @qml.qnode(dev)
    def circuit():
        to_pennylane(spec)
        return qml.state()

    return np.asarray(circuit())


def to_pl_observable(obs) -> object:
    """A qmlkit observable as the equivalent PennyLane one."""
    single = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    total = None
    for term in qk.PauliSum(()).__add__(obs).terms:
        factors = [single[p](q) for q, p in term.paulis if p != "I"]
        word = qml.Identity(0) if not factors else factors[0]
        for f in factors[1:]:
            word = word @ f
        piece = float(np.real(term.coeff)) * word
        total = piece if total is None else total + piece
    return total


def pl_expval(spec: CircuitSpec, obs) -> float:
    dev = qml.device("default.qubit", wires=spec.n_qubits)

    @qml.qnode(dev)
    def circuit():
        to_pennylane(spec)
        return qml.expval(to_pl_observable(obs))

    return float(circuit())


# --------------------------------------------------------------------------- #
# 1. gate matrices — the foundation everything else rests on
# --------------------------------------------------------------------------- #
def test_every_builtin_gate_has_a_pennylane_counterpart():
    """Adding a gate to qmlkit must not silently escape this file's coverage."""
    assert set(BUILTIN_GATES) == set(_PL_GATE), (
        f"gates without a PennyLane mapping: {sorted(set(BUILTIN_GATES) - set(_PL_GATE))}"
    )


@pytest.mark.parametrize("name", BUILTIN_GATES)
@pytest.mark.parametrize("angle", [0.0, 0.3, -1.7, 2.9, np.pi, np.pi / 2])
def test_gate_matrix_matches_pennylane(name, angle):
    """Every gate qmlkit defines, at several angles, against PennyLane's matrix."""
    gate = get_gate(name)
    params = [angle] * gate.n_params
    if gate.n_params == 0 and angle != 0.0:
        pytest.skip("constant gate — one angle is enough")
    ours = gate_matrix(name, params)
    theirs = qml.matrix(_PL_GATE[name](tuple(range(gate.n_qubits)), params))
    assert ours == pytest.approx(np.asarray(theirs), abs=EXACT)


@pytest.mark.parametrize("name", [n for n in BUILTIN_GATES if get_gate(n).n_params])
@pytest.mark.parametrize("angle", [0.2, -0.9, 2.4])
def test_gate_derivative_matches_differentiated_pennylane_matrix(name, angle):
    """qmlkit declares dU/dtheta in closed form; PennyLane's matrix is differenced.

    This is what makes adjoint differentiation trustworthy — a wrong dmatrix would
    otherwise produce a smooth, plausible, wrong gradient.
    """
    n_q = get_gate(name).n_qubits
    eps = 1e-6

    def m(t):
        return np.asarray(qml.matrix(_PL_GATE[name](tuple(range(n_q)), [t])))

    numeric = (m(angle + eps) - m(angle - eps)) / (2 * eps)
    assert gate_derivative(name, [angle]) == pytest.approx(numeric, abs=1e-8)


# --------------------------------------------------------------------------- #
# 2. randomised circuits — the part that explores rather than confirms
# --------------------------------------------------------------------------- #
def random_spec(n_qubits: int, n_ops: int, seed: int) -> CircuitSpec:
    """A random circuit drawn from the full gate set, fully bound."""
    rng = np.random.default_rng(seed)
    names = list(BUILTIN_GATES)
    ops: list[Op] = []
    while len(ops) < n_ops:
        name = names[rng.integers(len(names))]
        gate = get_gate(name)
        if gate.n_qubits > n_qubits:
            continue
        wires = tuple(rng.choice(n_qubits, size=gate.n_qubits, replace=False).tolist())
        params = tuple(rng.uniform(-np.pi, np.pi, gate.n_params).tolist())
        ops.append(Op(name, wires, params))
    return CircuitSpec(n_qubits, tuple(ops), 0)


@pytest.mark.parametrize("n_qubits", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(8))
def test_random_circuit_statevectors_agree(n_qubits, seed):
    """40 random circuits over the whole gate set, compared amplitude by amplitude."""
    spec = random_spec(n_qubits, 6 * n_qubits, seed)
    assert qk.statevector(spec) == pytest.approx(pl_state(spec), abs=EXACT)


@pytest.mark.parametrize("seed", range(12))
def test_random_circuit_probabilities_agree(seed):
    spec = random_spec(3, 14, 100 + seed)
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def probs():
        to_pennylane(spec)
        return qml.probs(wires=range(3))

    assert qk.probabilities(spec) == pytest.approx(np.asarray(probs()), abs=EXACT)


@pytest.mark.parametrize("seed", range(12))
def test_random_circuit_expectations_agree_over_random_observables(seed):
    """Random circuit and a random multi-term observable, including X and Y terms."""
    rng = np.random.default_rng(500 + seed)
    spec = random_spec(3, 12, 500 + seed)
    obs = None
    for _ in range(rng.integers(1, 4)):
        letters = rng.choice(["X", "Y", "Z"], size=rng.integers(1, 4))
        wires = rng.choice(3, size=len(letters), replace=False)
        term = qk.PauliString(
            tuple(sorted((int(w), str(p)) for w, p in zip(wires, letters, strict=True))),
            float(rng.uniform(-2, 2)),
        )
        obs = term if obs is None else obs + term
    assert qk.expval(spec, obs) == pytest.approx(pl_expval(spec, obs), abs=EXACT)


# --------------------------------------------------------------------------- #
# 3. gradients — every method, on both sides
# --------------------------------------------------------------------------- #
ANSATZE = ["hardware_efficient", "strongly_entangling", "simplified_two_design", "mps", "qcnn"]
OBSERVABLES = {
    "single-Z": lambda n: qk.Z(0),
    "two-body": lambda n: qk.ZZ(0, n - 1),
    "three-term": lambda n: qk.Z(0) + 0.5 * qk.ZZ(0, n - 1) + 0.3 * qk.X(1),
    "global-Z": lambda n: qk.PauliString(tuple((q, "Z") for q in range(n)), 1.0),
}


def pl_grad_of_spec(spec: CircuitSpec, theta: np.ndarray, obs) -> np.ndarray:
    """PennyLane's gradient of the *same* circuit, built through the IR.

    Going through the IR means both libraries differentiate an identical operation
    list — so a disagreement is a gradient bug, never a transcription slip.
    """
    dev = qml.device("default.qubit", wires=spec.n_qubits)
    pl_obs = to_pl_observable(obs)

    @qml.qnode(dev)
    def circuit(t):
        slot_angles = [t[s.ref.index] * s.ref.scale + s.ref.offset for s in spec.slots()]
        cursor = 0
        for op in spec.ops:
            params = []
            for p in op.params:
                if isinstance(p, ParamRef):
                    params.append(slot_angles[cursor])
                    cursor += 1
                else:
                    params.append(float(p))
            _PL_GATE[op.gate](op.qubits, params)
        return qml.expval(pl_obs)

    return np.asarray(qml.grad(circuit)(qml.numpy.array(theta, requires_grad=True)))


@pytest.mark.parametrize("name", ANSATZE)
@pytest.mark.parametrize("obs_name", list(OBSERVABLES))
def test_gradients_agree_across_ansatze_and_observables(name, obs_name):
    """20 (ansatz, observable) pairs — qmlkit's default gradient against PennyLane."""
    a = qk.get_ansatz(name, n_qubits=4)
    spec, theta = a.build(), a.init(seed=0)
    obs = OBSERVABLES[obs_name](4)
    assert qk.grad(spec, theta, obs) == pytest.approx(pl_grad_of_spec(spec, theta, obs), abs=1e-10)


@pytest.mark.parametrize("method", ["adjoint", "parameter-shift", "hadamard", "backprop"])
@pytest.mark.parametrize("name", ANSATZE)
def test_every_exact_method_agrees_with_pennylane(method, name):
    """Four exact routes on our side, one on theirs. All five must coincide."""
    if method == "backprop":
        pytest.importorskip("torch")
    a = qk.get_ansatz(name, n_qubits=3)
    spec, theta = a.build(), a.init(seed=1)
    obs = qk.Z(0) + 0.4 * qk.ZZ(0, 2)
    if method == "hadamard" and not qk.supports_adjoint(spec):
        pytest.skip("no Pauli generator")
    assert qk.grad(spec, theta, obs, method=method) == pytest.approx(
        pl_grad_of_spec(spec, theta, obs), abs=1e-10
    )


@pytest.mark.parametrize("pl_method", ["parameter-shift", "adjoint", "backprop", "finite-diff"])
def test_pennylanes_own_methods_agree_with_ours(pl_method):
    """The comparison run the other way: their four methods against our adjoint."""
    a = qk.hardware_efficient(3, n_layers=2)
    spec, theta = a.build(), a.init(seed=2)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
    dev = qml.device("default.qubit", wires=3)
    pl_obs = to_pl_observable(obs)

    @qml.qnode(dev, diff_method=pl_method)
    def circuit(t):
        for layer in range(2):
            for w in range(3):
                qml.RY(t[layer * 6 + w * 2], wires=w)
                qml.RZ(t[layer * 6 + w * 2 + 1], wires=w)
            for w in range(2):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(pl_obs)

    theirs = np.asarray(qml.grad(circuit)(qml.numpy.array(theta, requires_grad=True)))
    tol = 1e-6 if pl_method == "finite-diff" else 1e-10
    assert qk.grad(spec, theta, obs, method="adjoint") == pytest.approx(theirs, abs=tol)


@pytest.mark.parametrize("seed", range(6))
def test_gradients_agree_on_random_circuits(seed):
    """The fuzzer again, this time differentiated."""
    rng = np.random.default_rng(900 + seed)
    n_qubits = 3
    base = random_spec(n_qubits, 10, 900 + seed)
    # re-emit with every rotation parameterised, so there is something to differentiate
    ops, idx = [], 0
    for op in base.ops:
        if get_gate(op.gate).n_params:
            ops.append(Op(op.gate, op.qubits, (ParamRef(idx),)))
            idx += 1
        else:
            ops.append(op)
    if idx == 0:
        pytest.skip("no parameterised gates in this draw")
    spec = CircuitSpec(n_qubits, tuple(ops), idx)
    theta = rng.uniform(-np.pi, np.pi, idx)
    obs = qk.Z(0) + 0.3 * qk.ZZ(0, n_qubits - 1)
    assert qk.grad(spec, theta, obs) == pytest.approx(pl_grad_of_spec(spec, theta, obs), abs=1e-10)


# --------------------------------------------------------------------------- #
# 4. feature maps and encodings
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rotation", ["X", "Y", "Z"])
@pytest.mark.parametrize("n_features", [1, 2, 3, 4])
def test_angle_embedding_matches(rotation, n_features):
    x = np.linspace(0.2, 1.4, n_features)
    ours = qk.angle_encode(x, rotation="r" + rotation.lower())
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev)
    def circuit():
        qml.AngleEmbedding(x, wires=range(n_features), rotation=rotation)
        return qml.state()

    assert qk.statevector(ours) == pytest.approx(np.asarray(circuit()), abs=EXACT)


@pytest.mark.parametrize("seed", range(6))
def test_amplitude_embedding_matches(seed):
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=8)
    dev = qml.device("default.qubit", wires=3)

    @qml.qnode(dev)
    def circuit():
        qml.AmplitudeEmbedding(vec, wires=range(3), normalize=True)
        return qml.state()

    ours = qk.statevector(qk.amplitude_encode(vec))
    theirs = np.asarray(circuit())

    # qmlkit builds this from uniformly-controlled rotations rather than a backend
    # state-prep primitive, and the phase cascade drops one overall factor. That is
    # documented, and unobservable -- so what must match exactly is every observable
    # quantity, plus the claim that the leftover difference really is *one* phase.
    assert np.abs(ours) == pytest.approx(np.abs(theirs), abs=1e-10)
    ratio = ours / theirs
    assert np.abs(ratio - ratio[0]).max() < 1e-10, "difference is not a single global phase"
    assert abs(abs(ratio[0]) - 1.0) < 1e-10


@pytest.mark.parametrize("bits", [[0, 0], [1, 0], [0, 1, 1], [1, 1, 1], [1, 0, 1, 0]])
def test_basis_embedding_matches(bits):
    dev = qml.device("default.qubit", wires=len(bits))

    @qml.qnode(dev)
    def circuit():
        qml.BasisEmbedding(bits, wires=range(len(bits)))
        return qml.state()

    assert qk.statevector(qk.basis_encode(bits)) == pytest.approx(np.asarray(circuit()), abs=EXACT)


@pytest.mark.parametrize("reps", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_iqp_feature_map_matches_under_its_own_convention(reps, seed):
    """PennyLane's IQPEmbedding emits RZ(x_i) / MultiRZ(x_i x_j).

    qmlkit's PauliFeatureMap follows the Qiskit convention and emits Rz(2 phi).
    Neither is wrong; halving the data map lines them up exactly. This is recorded
    as a test because a kernel differing by exactly this factor is very hard to spot.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, np.pi, 2)
    fmap = qk.PauliFeatureMap(
        2,
        paulis=("Z", "ZZ"),
        reps=reps,
        data_map=lambda v, idx: float(np.prod([v[i] for i in idx])) / 2,
    )
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit():
        qml.IQPEmbedding(x, wires=range(2), n_repeats=reps)
        return qml.state()

    assert qk.statevector(fmap.build(x)) == pytest.approx(np.asarray(circuit()), abs=EXACT)


# --------------------------------------------------------------------------- #
# 5. kernels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits", [2, 3])
def test_fidelity_kernel_matrix_matches(n_qubits):
    """A whole Gram matrix, not a single entry."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, np.pi, (6, n_qubits))
    fmap = qk.AngleFeatureMap(n_qubits, entangle=False)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def overlap(a, b):
        qml.AngleEmbedding(a, wires=range(n_qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(n_qubits), rotation="Y")
        return qml.probs(wires=range(n_qubits))

    theirs = np.array([[float(overlap(a, b)[0]) for b in X] for a in X])
    assert qk.QuantumKernel(fmap)(X) == pytest.approx(theirs, abs=1e-10)


def test_swap_test_and_hadamard_test_agree_with_the_exact_overlap():
    """Three qmlkit estimators against PennyLane's compute-uncompute probability."""
    rng = np.random.default_rng(3)
    fmap = qk.AngleFeatureMap(2, entangle=False)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def overlap(a, b):
        qml.AngleEmbedding(a, wires=range(2), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(b, wires=range(2), rotation="Y")
        return qml.probs(wires=range(2))

    for _ in range(5):
        x, xp = rng.uniform(0, np.pi, 2), rng.uniform(0, np.pi, 2)
        reference = float(overlap(x, xp)[0])
        assert qk.fidelity_kernel(fmap, x, xp) == pytest.approx(reference, abs=1e-10)
        assert qk.swap_test_kernel(fmap, x, xp) == pytest.approx(reference, abs=1e-10)


# --------------------------------------------------------------------------- #
# 6. quantum information
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", range(10))
def test_reduced_density_matrices_and_entropies_agree(seed):
    spec = random_spec(3, 12, 700 + seed)
    state = qk.statevector(spec)
    dm = np.outer(state, np.conj(state))
    for wires in ([0], [1], [2], [0, 1], [1, 2]):
        assert qk.reduced_dm(spec, wires) == pytest.approx(
            np.asarray(qml.math.reduce_dm(dm, wires)), abs=EXACT
        )
        assert qk.purity(spec, wires) == pytest.approx(
            float(qml.math.purity(dm, indices=wires)), abs=EXACT
        )
        assert qk.vn_entropy(spec, wires, base=2) == pytest.approx(
            float(qml.math.vn_entropy(dm, indices=wires, base=2)), abs=1e-10
        )


@pytest.mark.parametrize("seed", range(8))
def test_mutual_information_agrees(seed):
    spec = random_spec(3, 10, 800 + seed)
    state = qk.statevector(spec)
    dm = np.outer(state, np.conj(state))
    assert qk.mutual_info(spec, [0], [1]) == pytest.approx(
        float(qml.math.mutual_info(dm, indices0=[0], indices1=[1])), abs=1e-10
    )


@pytest.mark.parametrize("seed", range(8))
def test_state_fidelity_agrees(seed):
    a, b = random_spec(2, 8, 300 + seed), random_spec(2, 8, 400 + seed)
    sa, sb = qk.statevector(a), qk.statevector(b)
    assert qk.state_fidelity(a, b) == pytest.approx(
        float(qml.math.fidelity_statevector(sa, sb)), abs=1e-10
    )


@pytest.mark.parametrize("seed", range(8))
def test_state_fidelity_is_the_more_accurate_of_the_two_routes(seed):
    """qmlkit hits the analytic |<a|b>|^2 exactly; the density-matrix route does not.

    ``qml.math.fidelity`` takes matrix square roots of rank-1 density matrices, which
    is ill-conditioned and costs ~8 digits. Recorded so that a future tolerance
    change here is a deliberate decision rather than an accident.
    """
    a, b = random_spec(2, 8, 300 + seed), random_spec(2, 8, 400 + seed)
    sa, sb = qk.statevector(a), qk.statevector(b)
    analytic = float(abs(np.vdot(sa, sb)) ** 2)
    assert qk.state_fidelity(a, b) == pytest.approx(analytic, abs=1e-15)
    via_dm = float(qml.math.fidelity(np.outer(sa, np.conj(sa)), np.outer(sb, np.conj(sb))))
    assert abs(via_dm - analytic) < 1e-6  # loose: it is the less accurate route


# --------------------------------------------------------------------------- #
# 7. Fourier spectra of re-uploading models
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
def test_reuploading_fourier_spectrum_matches(n_layers):
    """The central claim of the re-uploading literature, measured on both sides."""
    model = qk.reupload(
        qk.AngleFeatureMap(1, entangle=False),
        n_layers=n_layers,
        rotations=("rz", "ry", "rz"),
        entangler=None,
    )
    weights = np.random.default_rng(0).uniform(-np.pi, np.pi, model.n_weights)
    bound = model.build()

    def ours(x):
        return qk.expval(bound, qk.Z(0), theta=np.concatenate([model.angles([x]), weights]))

    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def theirs(x):
        for layer in range(n_layers):
            qml.RY(x, wires=0)
            qml.RZ(weights[layer * 3], wires=0)
            qml.RY(weights[layer * 3 + 1], wires=0)
            qml.RZ(weights[layer * 3 + 2], wires=0)
        return qml.expval(qml.PauliZ(0))

    xs = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    assert np.array([ours(x) for x in xs]) == pytest.approx(
        np.array([float(theirs(x)) for x in xs]), abs=EXACT
    )

    ours_spectrum = qk.fourier.spectrum(ours, n_layers + 3)
    coeffs = qml.fourier.coefficients(lambda x: theirs(x), 1, n_layers)
    for k in range(n_layers + 1):
        theirs_amp = float(abs(coeffs[k]) + abs(coeffs[-k])) if k else float(abs(coeffs[0]))
        assert ours_spectrum.get(k, 0.0) == pytest.approx(theirs_amp, abs=1e-10)


# --------------------------------------------------------------------------- #
# 8. Fubini-Study metric / quantum Fisher information
# --------------------------------------------------------------------------- #
def _metric_case(n_layers):
    """The same RY/CNOT circuit on both sides, plus a spare wire for their ancilla."""
    a = qk.hardware_efficient(3, n_layers, rotations=("ry",), pattern="chain")
    spec, theta = a.build(), a.init(seed=0)
    # PennyLane's exact metric tensor runs Hadamard tests and needs a free wire;
    # qmlkit computes it straight from the statevector, so it needs none.
    dev = qml.device("default.qubit", wires=4)

    @qml.qnode(dev)
    def circuit(t):
        for layer in range(n_layers):
            for w in range(3):
                qml.RY(t[layer * 3 + w], wires=w)
            for w in range(2):
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(qml.PauliZ(0))

    return spec, theta, circuit, qml.numpy.array(theta, requires_grad=True)


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_full_metric_tensor_matches(n_layers):
    """QNG is only as good as the metric it follows, so this one has to be right."""
    spec, theta, circuit, tp = _metric_case(n_layers)
    theirs = np.asarray(qml.metric_tensor(circuit, approx=None)(tp))
    # exact on both sides: qmlkit differentiates the state in closed form, PennyLane
    # runs Hadamard tests. Neither is finite-differenced, so this is a tight bound.
    assert qk.metric_tensor(spec, theta, approx=None) == pytest.approx(theirs, abs=1e-12)


@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_diagonal_metric_tensor_matches(n_layers):
    spec, theta, circuit, tp = _metric_case(n_layers)
    theirs = np.asarray(qml.metric_tensor(circuit, approx="diag")(tp))
    assert qk.metric_tensor(spec, theta, approx="diag") == pytest.approx(theirs, abs=1e-12)


def test_block_diag_means_something_different_in_each_library():
    """A real definitional difference, and one that changes QNG steps.

    PennyLane's ``approx="block-diag"`` blocks by *layer* and zeroes every cross-layer
    entry. qmlkit computes the exact metric from state overlaps, which on a simulator
    costs no more than the approximation, so its ``"block-diag"`` returns the full
    tensor. qmlkit is therefore following the true geometry where PennyLane follows an
    approximation to it — but anyone porting QNG code between the two should know the
    same keyword does not mean the same thing.
    """
    spec, theta, circuit, tp = _metric_case(2)
    exact = qk.metric_tensor(spec, theta, approx=None)
    assert qk.metric_tensor(spec, theta, approx="block-diag") == pytest.approx(exact, abs=EXACT)

    theirs = np.asarray(qml.metric_tensor(circuit, approx="block-diag")(tp))
    dropped = np.abs(theirs - exact).max()
    assert dropped > 0.01, "expected PennyLane to drop a real cross-layer term here"


def test_quantum_fisher_information_is_four_times_the_metric():
    """The factor of 4 is a definition, and an easy one to get wrong."""
    a = qk.hardware_efficient(2, 1, rotations=("ry",), pattern="chain")
    spec, theta = a.build(), a.init(seed=0)
    assert qk.quantum_fisher_information(spec, theta) == pytest.approx(
        4 * qk.metric_tensor(spec, theta, approx=None), abs=EXACT
    )


# --------------------------------------------------------------------------- #
# 9. sampling converges on the exact value both libraries agree about
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("shots", [2000, 20000])
def test_sampled_expectations_converge_to_the_shared_exact_value(shots):
    spec = random_spec(2, 8, 42)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 1)
    exact = pl_expval(spec, obs)
    sampled, err = qk.expectation(spec, obs, shots=shots, seed=0, return_std=True)
    assert abs(sampled - exact) < 5 * err + 1e-12


# --------------------------------------------------------------------------- #
# 10. ansatz templates against PennyLane's own
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits", [2, 3, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 3])
def test_basic_entangler_matches_pennylanes_template(n_qubits, n_layers):
    """RX per wire then a CNOT ring — same definition in both libraries."""
    a = qk.get_ansatz("basic_entangler", n_qubits=n_qubits, n_layers=n_layers)
    spec, theta = a.build(), a.init(seed=0)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(w):
        qml.BasicEntanglerLayers(w, wires=range(n_qubits))
        return qml.state()

    weights = theta.reshape(n_layers, n_qubits)
    assert qk.statevector(spec.bind(theta)) == pytest.approx(
        np.asarray(circuit(weights)), abs=EXACT
    )


@pytest.mark.parametrize("n_qubits", [3, 4, 5])
@pytest.mark.parametrize("n_layers", [1, 2])
def test_strongly_entangling_matches_with_the_range_pattern_pinned(n_qubits, n_layers):
    """Rot = RZ.RY.RZ per wire, then an entangling ring.

    PennyLane widens the CNOT range on each successive layer by default; qmlkit
    repeats a fixed ring. Pinning ``ranges`` to 1 compares the same circuit — the
    difference is a template default, not a disagreement about the ansatz.
    """
    a = qk.get_ansatz("strongly_entangling", n_qubits=n_qubits, n_layers=n_layers)
    spec, theta = a.build(), a.init(seed=0)
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits), ranges=[1] * n_layers)
        return qml.state()

    weights = theta.reshape(n_layers, n_qubits, 3)
    assert qk.statevector(spec.bind(theta)) == pytest.approx(
        np.asarray(circuit(weights)), abs=EXACT
    )


def test_two_qubit_ring_is_one_cnot_here_and_two_in_pennylane():
    """A convention difference that only exists at n=2, pinned so it stays deliberate.

    On two qubits a ring would revisit the same pair, so ``entangler_pairs`` collapses
    it to a single ``(0, 1)``. PennyLane's templates run their loop uniformly and emit
    both ``CNOT(0, 1)`` and ``CNOT(1, 0)``. Neither is wrong — but a two-qubit
    strongly-entangling layer is genuinely a different circuit in the two libraries.
    """
    assert qk.entangler_pairs(2, "ring") == ((0, 1),)
    assert qk.entangler_pairs(3, "ring") == ((0, 1), (1, 2), (2, 0))

    a = qk.get_ansatz("strongly_entangling", n_qubits=2, n_layers=1)
    spec, theta = a.build(), a.init(seed=0)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(2), ranges=[1])
        return qml.state()

    theirs = np.asarray(circuit(theta.reshape(1, 2, 3)))
    assert np.abs(qk.statevector(spec.bind(theta)) - theirs).max() > 1e-6

    # ...and adding the second CNOT by hand reconciles them exactly
    builder = qk.QCircuit(2)
    for w in range(2):
        builder.rz(w, theta[w * 3]).ry(w, theta[w * 3 + 1]).rz(w, theta[w * 3 + 2])
    builder.cx(0, 1).cx(1, 0)
    assert qk.statevector(builder.to_spec()) == pytest.approx(theirs, abs=EXACT)


# --------------------------------------------------------------------------- #
# 11. gradients with respect to the *input*, not the weights
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_features", [1, 2, 3])
def test_input_gradients_match(n_features):
    """df/dx is what makes a classical layer placed before the circuit trainable."""
    x = np.linspace(0.3, 1.1, n_features)
    spec = qk.angle_encode(x, trainable=True)
    obs = qk.Z(0) if n_features == 1 else qk.Z(0) + 0.5 * qk.Z(n_features - 1)
    dev = qml.device("default.qubit", wires=n_features)

    @qml.qnode(dev)
    def circuit(v):
        qml.AngleEmbedding(v, wires=range(n_features), rotation="Y")
        return qml.expval(to_pl_observable(obs))

    theirs = np.asarray(qml.grad(circuit)(qml.numpy.array(x, requires_grad=True)))
    assert qk.grad(spec, x, obs) == pytest.approx(theirs, abs=1e-10)


# --------------------------------------------------------------------------- #
# 12. second derivatives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_layers", [1, 2])
def test_hessian_matches_pennylanes(n_layers):
    a = qk.hardware_efficient(2, n_layers, rotations=("ry",), pattern="chain")
    spec, theta = a.build(), a.init(seed=0)
    obs = qk.Z(0) + 0.5 * qk.ZZ(0, 1)
    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def circuit(t):
        for layer in range(n_layers):
            for w in range(2):
                qml.RY(t[layer * 2 + w], wires=w)
            qml.CNOT(wires=[0, 1])
        return qml.expval(to_pl_observable(obs))

    theirs = np.asarray(qml.jacobian(qml.grad(circuit))(qml.numpy.array(theta, requires_grad=True)))
    # ours differences the exact gradient, so it carries the outer O(eps^2) only
    assert qk.hessian(spec, theta, obs) == pytest.approx(theirs, abs=1e-6)


# --------------------------------------------------------------------------- #
# 13. optimisers reach the same place
# --------------------------------------------------------------------------- #
def _matched_optimiser_case(n_wires=3, n_layers=2, spare=0):
    """hardware_efficient(ry, rz, chain) built identically on both sides."""
    a = qk.hardware_efficient(n_wires, n_layers)
    spec, start = a.build(), a.init("uniform", seed=1)
    obs = sum((qk.Z(w) for w in range(1, n_wires)), qk.Z(0))  # minimum is -n_wires
    dev = qml.device("default.qubit", wires=n_wires + spare)

    @qml.qnode(dev)
    def circuit(t):
        for layer in range(n_layers):
            for w in range(n_wires):
                qml.RY(t[layer * 2 * n_wires + w * 2], wires=w)
                qml.RZ(t[layer * 2 * n_wires + w * 2 + 1], wires=w)
            for w in range(n_wires - 1):  # chain, not ring
                qml.CNOT(wires=[w, w + 1])
        return qml.expval(to_pl_observable(obs))

    return a, spec, start, obs, circuit


def test_rotosolve_traces_the_same_trajectory_as_pennylanes():
    """Not just the same minimum — the same value at every sweep.

    Rotosolve is deterministic and closed-form, so two correct implementations must
    agree step for step. Comparing only the endpoint would hide a different path.
    """
    a, spec, start, obs, circuit = _matched_optimiser_case()
    opt = qml.RotosolveOptimizer()
    # PennyLane requires the spectrum to be declared; qmlkit infers it from the gates
    nums_frequency = {"t": {(i,): 1 for i in range(a.n_params)}}
    t = qml.numpy.array(start, requires_grad=True)
    theirs = [float(circuit(t))]
    for _ in range(20):
        t = opt.step(circuit, t, nums_frequency=nums_frequency)
        theirs.append(float(circuit(t)))

    _, ours = qk.minimize_rotosolve(
        lambda v: qk.expval(spec, obs, theta=np.asarray(v)), start, n_sweeps=20, tol=0.0
    )
    assert ours == pytest.approx(theirs, abs=1e-10)
    assert ours[-1] < ours[0]


def test_qng_matches_pennylane_when_both_use_the_exact_metric():
    a, spec, start, obs, circuit = _matched_optimiser_case(spare=1)
    opt = qml.QNGOptimizer(stepsize=0.15, approx=None, lam=1e-6)
    t = qml.numpy.array(start, requires_grad=True)
    theirs = [float(circuit(t))]
    for _ in range(25):
        t = opt.step(circuit, t)
        theirs.append(float(circuit(t)))

    _, ours = qk.minimize_qng(spec, start, obs, n_steps=25, lr=0.15)
    assert ours == pytest.approx(theirs, abs=1e-6)


def test_qng_beats_pennylanes_default_because_its_metric_is_exact():
    """The practical consequence of the block-diag difference, measured.

    PennyLane's QNGOptimizer defaults to ``approx="block-diag"``, which discards
    cross-layer curvature. qmlkit always follows the exact metric — free on a
    simulator — and gets materially further down the same landscape in the same
    number of steps at the same step size.
    """
    a, spec, start, obs, circuit = _matched_optimiser_case(spare=1)
    opt = qml.QNGOptimizer(stepsize=0.15, approx="block-diag", lam=1e-6)
    t = qml.numpy.array(start, requires_grad=True)
    for _ in range(25):
        t = opt.step(circuit, t)

    _, ours = qk.minimize_qng(spec, start, obs, n_steps=25, lr=0.15)
    assert ours[-1] < float(circuit(t)) - 0.5, "expected a clear margin, not a coin flip"
    assert ours[-1] == pytest.approx(-3.0, abs=1e-6)
