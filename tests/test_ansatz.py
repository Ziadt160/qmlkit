"""Ansatz vocabulary, the built-in zoo, and the registry.

The point of this module is that a new ansatz costs a line and inherits every
capability. These tests assert exactly that: parameter counts are inferred, weight
tying is real, and a custom ansatz gets correct gradients without opting in.
"""

from __future__ import annotations

import numpy as np
import pytest

import qmlkit as qk
from qmlkit.ansatz import (
    Ansatz,
    BuildContext,
    Custom,
    EntanglerLayer,
    ParametricEntangler,
    PoolLayer,
    RotationLayer,
    conv_block,
    get_ansatz,
    list_ansatze,
    register_ansatz,
    repeat,
    share,
)

ZOO = list(list_ansatze())


# --------------------------------------------------------------------------- #
# the vocabulary
# --------------------------------------------------------------------------- #
def test_rotation_layer_allocates_one_parameter_per_gate_per_wire():
    a = Ansatz(3, RotationLayer(("ry", "rz")))
    assert a.n_params == 6
    assert a.build().gate_counts() == {"ry": 3, "rz": 3}


def test_entangler_layer_follows_its_pattern():
    for pattern, expected in [("chain", 2), ("ring", 3), ("full", 3)]:
        a = Ansatz(3, EntanglerLayer("cx", pattern))
        assert a.build().gate_counts().get("cx", 0) == expected


def test_blocks_compose_with_plus():
    block = RotationLayer("ry") + EntanglerLayer("cx") + RotationLayer("rz")
    a = Ansatz(3, block)
    assert a.n_params == 6
    assert "cx" in a.build().gate_counts()


def test_repeat_allocates_fresh_parameters_each_time():
    a = Ansatz(2, repeat(3, RotationLayer("ry")))
    assert a.n_params == 6
    assert all(len(a.build().occurrences_of(i)) == 1 for i in range(6))


def test_share_reuses_one_parameter_set():
    """Weight tying: one parameter, several occurrences."""
    a = Ansatz(2, share(3, RotationLayer("ry")))
    assert a.n_params == 2
    assert [len(a.build().occurrences_of(i)) for i in range(2)] == [3, 3]


def test_pool_layer_halves_the_active_register():
    a = Ansatz(4, RotationLayer("ry") + PoolLayer("odd") + RotationLayer("ry"))
    assert a.n_params == 4 + 2  # four wires, then two


def test_parametric_entangler_uses_the_four_term_rule():
    a = Ansatz(3, ParametricEntangler("crz", "ring"))
    assert a.n_params == 3
    assert a.resources()["grad_circuits"] == 3 * 4  # not 3 * 2


def test_layers_reject_the_wrong_kind_of_gate():
    with pytest.raises(ValueError, match="takes no parameters"):
        RotationLayer("cx")
    with pytest.raises(ValueError, match="is parameterised"):
        EntanglerLayer("crz")
    with pytest.raises(ValueError, match="takes no parameters"):
        ParametricEntangler("cx")


def test_repeat_and_share_reject_degenerate_counts():
    with pytest.raises(ValueError, match="times must be at least 1"):
        repeat(0, RotationLayer("ry"))
    with pytest.raises(ValueError, match="times must be at least 1"):
        share(0, RotationLayer("ry"))


def test_pool_layer_validates_its_argument():
    with pytest.raises(ValueError, match="must be 'even' or 'odd'"):
        PoolLayer("middle")


def test_custom_block_is_the_escape_hatch():
    """Anything the vocabulary cannot express still composes with everything else."""

    def weird(qc, ctx):
        for q in ctx.active:
            qc.apply("rx", q, ctx.new_param(scale=2.0))

    a = Ansatz(3, Custom(weird, "weird") + EntanglerLayer("cz", "ring"))
    assert a.n_params == 3
    # the scale=2 chain rule must survive into the gradient
    g = qk.grad(a.build(), np.array([0.3, 0.0, 0.0]), qk.Z(0), method="parameter-shift")
    assert g[0] == pytest.approx(-2.0 * np.sin(2 * 0.3), abs=1e-9)


def test_build_context_replay_is_scoped():
    ctx = BuildContext(2)
    first = ctx.new_param().index
    with ctx.replaying([first]):
        assert ctx.new_param().index == first
    assert ctx.new_param().index == first + 1


# --------------------------------------------------------------------------- #
# the zoo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ZOO)
def test_every_zoo_ansatz_builds_runs_and_differentiates(name):
    a = get_ansatz(name, n_qubits=4)
    assert a.n_params > 0
    theta = a.init(seed=0)
    assert theta.shape == a.param_shape
    spec = a.build(theta)
    value = qk.expectation(spec, qk.Z(0))
    assert -1.0 - 1e-9 <= value <= 1.0 + 1e-9
    # gradients work without the ansatz knowing anything about gradients
    exact = qk.grad(a.build(), theta, qk.Z(0), method="adjoint")
    shift = qk.grad(a.build(), theta, qk.Z(0), method="parameter-shift")
    assert exact == pytest.approx(shift, abs=1e-9)


@pytest.mark.parametrize("name", ZOO)
def test_every_zoo_ansatz_reports_resources(name):
    r = get_ansatz(name, n_qubits=4).resources()
    for key in ("n_qubits", "n_params", "depth", "n_2q", "grad_circuits"):
        assert key in r


def test_qaoa_has_two_parameters_per_round_whatever_the_width():
    for n in (3, 6, 10):
        for p in (1, 2, 3):
            assert qk.qaoa_ansatz(n, p=p).n_params == 2 * p


def test_qcnn_weight_tying_cuts_parameters_not_gradient_cost():
    """The real convolutional tradeoff: fewer weights, same measurement cost."""
    tied = qk.qcnn_ansatz(8, tie_weights=True)
    free = qk.qcnn_ansatz(8, tie_weights=False)
    assert tied.n_params < free.n_params
    assert tied.resources()["grad_circuits"] == free.resources()["grad_circuits"]
    # the tied filter really does slide across every pair
    assert len(tied.build().occurrences_of(0)) > 1


def test_tree_tensor_network_is_log_depth():
    deep = qk.hardware_efficient(8, 3).resources()["depth"]
    assert qk.tree_tensor_network(8).resources()["depth"] < deep


def test_init_methods():
    a = qk.hardware_efficient(3, 2)
    assert np.allclose(a.init("zeros"), 0.0)
    small = a.init("small", seed=0)
    wide = a.init("uniform", seed=0)
    assert np.abs(small).max() < np.abs(wide).max()
    with pytest.raises(ValueError, match="unknown init method"):
        a.init("magic")


def test_ansatz_validates_width():
    with pytest.raises(ValueError, match="n_qubits must be at least 1"):
        Ansatz(0, RotationLayer("ry"))


# --------------------------------------------------------------------------- #
# the registry — how a researcher adds their own
# --------------------------------------------------------------------------- #
def test_a_custom_ansatz_is_one_line_and_inherits_everything():
    @register_ansatz("brick_wall_test")
    def brick_wall(n_qubits: int, n_layers: int = 2) -> Ansatz:
        return Ansatz(
            n_qubits,
            repeat(n_layers, RotationLayer("ry") + EntanglerLayer("cz", "alternating")),
            "brick_wall",
        )

    assert "brick_wall_test" in list_ansatze()
    a = get_ansatz("brick_wall_test", n_qubits=4, n_layers=2)

    assert a.n_params == 8  # inferred, never declared
    theta = a.init(seed=0)
    assert a.resources()["grad_circuits"] == 16
    exact = qk.grad(a.build(), theta, qk.Z(0), method="adjoint")
    shift = qk.grad(a.build(), theta, qk.Z(0), method="parameter-shift")
    assert exact == pytest.approx(shift, abs=1e-9)


def test_registry_rejects_duplicates_and_unknowns():
    with pytest.raises(ValueError, match="already registered"):
        register_ansatz(
            "hardware_efficient", lambda n_qubits: Ansatz(n_qubits, RotationLayer("ry"))
        )
    with pytest.raises(KeyError, match="unknown ansatz"):
        get_ansatz("no_such_ansatz", n_qubits=2)


def test_conv_block_tied_and_free():
    tied = Ansatz(4, conv_block(tied=True))
    free = Ansatz(4, conv_block(tied=False))
    assert tied.n_params == 2
    assert free.n_params == 6  # three chain pairs, two angles each
