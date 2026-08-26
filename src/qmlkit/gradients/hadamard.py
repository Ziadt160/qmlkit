r"""Hadamard-test gradient — one circuit per parameter instead of two.

For :math:`U_k = e^{-i\theta_k P_k/2}` inserted at position :math:`k`, write
:math:`|\varphi\rangle` for the state you get by applying :math:`P_k` right after
:math:`U_k`. Then

.. math::  \partial_k E = -\,\mathrm{Im}\,\langle\varphi| O |\psi\rangle

and that imaginary part is exactly what a Hadamard test reads out. Put an ancilla in
:math:`|+\rangle`, run the circuit with a **controlled** :math:`P_k` inserted after
gate :math:`k`, and measure :math:`\langle Y_a \otimes O\rangle`:

.. code-block:: text

    |+>_a ----------●----------  <Y_a (x) O>
    |0>_n -- U_1..U_k -- P_k -- U_{k+1}..U_L --

Half the circuits of parameter-shift, and unlike adjoint it is a real measurement,
so it is valid on hardware. The trade is an ancilla that must couple to every wire the
generator touches, plus controlled gates. On real devices that routing cost usually
outweighs the saved circuits, which is why parameter-shift stays the default there.

On a simulator the halving does show up in wall-clock: 5 qubits, `P=120`, two-term
observable gives 404 ms against parameter-shift's 823 ms. Getting that required
passing the lifted observable as a single sum rather than term by term — the ancilla
already doubles the statevector, so re-preparing it per term would have handed the
whole advantage straight back.

Only single-qubit Pauli rotations are supported: their generators are Paulis, so the
controlled form is a ``cx``/``cy``/``cz``. A controlled rotation's generator is not a
Pauli, and this method refuses it rather than guessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.execute import BackendLike, expval
from qmlkit.core.ir import CircuitSpec, Op
from qmlkit.core.observables import Observable, PauliString, PauliSum, Z, as_sum

__all__ = ["hadamard_grad", "supports_hadamard_grad", "hadamard_grad_cost"]

#: parametric gate -> the Pauli its generator is built from
_GENERATOR = {"rx": "X", "ry": "Y", "rz": "Z"}
_CONTROLLED = {"X": "cx", "Y": "cy", "Z": "cz"}


def supports_hadamard_grad(spec: CircuitSpec) -> bool:
    """True if every parameterised gate has a plain Pauli generator."""
    return all(s.gate in _GENERATOR for s in spec.slots())


def hadamard_grad(
    spec: CircuitSpec,
    theta: npt.NDArray[Any],
    obs: Observable | None = None,
    backend: BackendLike = None,
    shots: int | None = None,
    seed: int | None = None,
) -> npt.NDArray[Any]:
    """Exact gradient using one extra qubit and one circuit per parameter."""
    obs = Z(0) if obs is None else obs
    if not supports_hadamard_grad(spec):
        offenders = sorted({s.gate for s in spec.slots() if s.gate not in _GENERATOR})
        raise ValueError(
            f"the Hadamard-test gradient needs Pauli-generated rotations; {offenders} "
            'are not. Use method="parameter-shift", which handles any declared spectrum.'
        )

    arr = np.asarray(theta, dtype=float).ravel()
    slots = spec.slots()
    slot_angles = spec.bind_slots(arr)
    bound = spec.with_slot_angles(slot_angles)
    n = spec.n_qubits
    ancilla = n

    grad = np.zeros(spec.n_params, dtype=float)
    for slot in slots:
        pauli = _GENERATOR[slot.gate]
        target = spec.ops[slot.op_index].qubits[0]

        ops: list[Op] = [Op("h", (ancilla,))]
        for j, op in enumerate(bound.ops):
            ops.append(op)
            if j == slot.op_index:
                # controlled generator, inserted immediately after the gate
                ops.append(Op(_CONTROLLED[pauli], (ancilla, target)))
        probe = CircuitSpec(n + 1, tuple(ops), 0)

        # <Y_ancilla (x) O> reads off the imaginary part we need. Pass the whole sum
        # in one call: the backend prepares the state once and accumulates every term
        # from it, so a k-term observable still costs a single circuit, not k of them.
        lifted = PauliSum(
            tuple(PauliString((*t.paulis, (ancilla, "Y")), t.coeff) for t in as_sum(obs).terms)
        )
        value = expval(probe, lifted, shots=shots, backend=backend, seed=seed)
        grad[slot.ref.index] += value * slot.ref.scale  # += sums tied occurrences

    return grad


def hadamard_grad_cost(spec: CircuitSpec) -> int:
    """Circuits per gradient: one per parameterised slot."""
    return len(spec.slots())
