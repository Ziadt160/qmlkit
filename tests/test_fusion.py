"""Gate fusion must not change a single amplitude.

Fusion rewrites the circuit before it runs: adjacent gates are composed into one
wider matrix and applied once. It is worth 5.6x at 18 qubits and it is also the most
dangerous optimisation in the library, because a mistake in it does not raise. The
circuit still runs, the state is still normalised, the expectation is still in range,
and the number is wrong.

So every test here compares against the unfused path on the same circuit, and the
randomised ones are the ones that matter — the hand-picked cases confirm what the
author already believed.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.core.backends.numpy_backend import NumpyBackend
from qmlkit.core.ir import CircuitSpec, Op

#: Gates drawn from at random. A snapshot, not the live registry - other test modules
#: register throwaway gates at run time, which is how a fuzzer comes to pass alone and
#: fail in a full run.
ONE_QUBIT = ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
TWO_QUBIT = ["cx", "cy", "cz", "crx", "cry", "crz"]
PARAMETRIC = {"rx", "ry", "rz", "crx", "cry", "crz"}


def _unfused(backend: NumpyBackend, spec: CircuitSpec) -> np.ndarray:
    """The same backend with fusion switched off."""
    keep = backend.fuse_min_qubits
    backend.fuse_min_qubits = 10**6
    try:
        return backend.statevector(spec)
    finally:
        backend.fuse_min_qubits = keep


def _random_spec(n_qubits: int, n_ops: int, rng: np.random.Generator) -> CircuitSpec:
    ops = []
    for _ in range(n_ops):
        if n_qubits > 1 and rng.random() < 0.4:
            gate = str(rng.choice(TWO_QUBIT))
            a, b = rng.choice(n_qubits, size=2, replace=False)
            qubits = (int(a), int(b))
        else:
            gate = str(rng.choice(ONE_QUBIT))
            qubits = (int(rng.integers(n_qubits)),)
        params = (float(rng.uniform(-np.pi, np.pi)),) if gate in PARAMETRIC else ()
        ops.append(Op(gate, qubits, params))
    return CircuitSpec(n_qubits=n_qubits, ops=tuple(ops))


# --------------------------------------------------------------------------- #
# the property that matters
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_qubits", [14, 15, 16])
@pytest.mark.parametrize("seed", range(6))
def test_fused_equals_unfused_on_random_circuits(n_qubits: int, seed: int) -> None:
    """Randomised, because a fuzzer explores what hand-picked cases confirm.

    Descending and non-adjacent wires are what a fusion pass gets wrong - a block
    reindexes its gates onto local positions, and getting that map backwards is
    silent.
    """
    rng = np.random.default_rng(seed)
    spec = _random_spec(n_qubits, n_ops=40, rng=rng)
    backend = NumpyBackend()

    fused = backend.statevector(spec)
    plain = _unfused(backend, spec)

    assert np.allclose(fused, plain, atol=1e-12), f"fusion changed the state at {n_qubits} qubits"


def test_fusion_is_actually_on_at_these_widths() -> None:
    """A test that compares two identical code paths proves nothing.

    If `fuse_min_qubits` ever rises above the widths parametrised above, every
    assertion here would pass by running the same loop twice.
    """
    backend = NumpyBackend()
    assert backend.fuse_min_qubits <= 14
    spec = _random_spec(14, n_ops=40, rng=np.random.default_rng(0))
    assert len(backend._fused(spec)) < len(spec.ops), "nothing was actually fused"


def test_fusion_is_off_where_it_loses() -> None:
    """Below the threshold the blocks cost more than the applies they replace."""
    backend = NumpyBackend()
    spec = _random_spec(6, n_ops=40, rng=np.random.default_rng(0))
    assert backend.fuse_min_qubits > 6
    assert np.allclose(backend.statevector(spec), _unfused(backend, spec), atol=1e-12)


# --------------------------------------------------------------------------- #
# it has to survive everything else the library does
# --------------------------------------------------------------------------- #
def test_expectation_and_gradient_agree_through_fusion() -> None:
    """Fusion sits under every consumer, so the consumers are what must agree."""
    ansatz = qk.hardware_efficient(14, 2)
    spec, theta = ansatz.build(), ansatz.init(seed=0)
    obs = qk.Z(0) + 0.5 * qk.ZZ(3, 11)
    backend = NumpyBackend()

    bound = spec.bind(theta)
    assert np.allclose(backend.statevector(bound), _unfused(backend, bound), atol=1e-12)

    fused_grad = qk.grad(spec, theta, obs, method="adjoint")
    assert np.isfinite(fused_grad).all()
    assert fused_grad.shape == (spec.n_params,)


def test_a_registered_gate_survives_fusion() -> None:
    """A gate the library has never seen has to fuse like any other."""
    from qmlkit.core.gates import _REGISTRY

    def xy(t: float) -> np.ndarray:
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.array(
            [[1, 0, 0, 0], [0, c, -1j * s, 0], [0, -1j * s, c, 0], [0, 0, 0, 1]], dtype=complex
        )

    qk.register_gate(qk.GateDef("xy_fuse", 2, 1, xy, frequencies=(1.0,)))
    try:
        rng = np.random.default_rng(3)
        ops = [Op("h", (q,)) for q in range(14)]
        for _ in range(12):
            a, b = rng.choice(14, size=2, replace=False)
            ops.append(Op("xy_fuse", (int(a), int(b)), (float(rng.uniform(-np.pi, np.pi)),)))
        spec = CircuitSpec(n_qubits=14, ops=tuple(ops))
        backend = NumpyBackend()
        assert np.allclose(backend.statevector(spec), _unfused(backend, spec), atol=1e-12)
    finally:
        _REGISTRY.pop("xy_fuse", None)


# --------------------------------------------------------------------------- #
# the recommender has to agree with the thresholds it describes
# --------------------------------------------------------------------------- #
def test_recommend_matches_the_backend_it_describes() -> None:
    """A recommender that drifts from the code is worse than none.

    Both the fusion flag and the batching flag are read off `NumpyBackend`, so this
    fails the moment a threshold moves without the advice moving with it.
    """
    backend = NumpyBackend()
    for n in (4, 10, 12, 14, 16):
        rec = qk.recommend(qk.hardware_efficient(n, 2))
        assert rec.n_qubits == n
        assert rec.statevector_bytes == 2**n * 16
        if rec.backend == "numpy":
            assert rec.fusion == (n >= backend.fuse_min_qubits)
            assert rec.batching == (n <= backend.batch_max_qubits)


def test_recommend_always_names_something_runnable() -> None:
    """The NumPy reference is always installed, so there is always an answer."""
    rec = qk.recommend(qk.hardware_efficient(20, 2))
    assert rec.backend in set(qk.list_backends())
    assert rec.reason
    assert rec.notes


def test_recommend_refuses_what_it_cannot_read() -> None:
    with pytest.raises(TypeError, match="recommend"):
        qk.recommend("not a circuit")
