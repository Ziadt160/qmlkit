"""Exact statevector simulator — the reference implementation.

This backend is the yardstick every other backend is measured against, and it is
what lets the library (and CI) run on Python 3.11+, where SpinQit has no wheel.
It is deliberately double-precision throughout: ``torch.autograd.gradcheck``
requires float64, and so does any honest claim that parameter-shift is *exact*.

Sampling, basis rotation and expectation semantics all live in
:class:`~qmlkit.core.backends.base.Backend`; this class supplies the statevector
and nothing else.

Convention: qubit 0 is the **most significant** bit of a bitstring, so
``basis_encode([1, 0, 1])`` yields ``'101'`` and reads in the order it was written.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.core.backends.base import Backend
from qmlkit.core.gates import gate_matrix
from qmlkit.core.ir import CircuitSpec, Op, ParamRef


def _apply(
    state: npt.NDArray[Any], matrix: npt.NDArray[Any], qubits: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Apply a k-qubit gate to the tensor-shaped state."""
    k = len(qubits)
    op = matrix.reshape((2,) * (2 * k))
    # contract the gate's input legs with the state's qubit axes
    state = np.tensordot(op, state, axes=(list(range(k, 2 * k)), list(qubits)))
    # tensordot puts the gate's output legs first; move them back into place
    return np.moveaxis(state, list(range(k)), list(qubits))


def _apply_batch(
    state: npt.NDArray[Any], matrices: npt.NDArray[Any], qubits: tuple[int, ...]
) -> npt.NDArray[Any]:
    """Apply a k-qubit gate to a whole stack of states at once.

    ``state`` is ``(batch,) + (2,)*n``, so axis 0 is the batch and qubit ``q`` lives on
    axis ``q+1``. ``matrices`` is either one ``(d, d)`` matrix used for every sample —
    a gate with literal angles — or ``(batch, d, d)`` when the angle varies across the
    batch, which is the case for every encoding gate. Both shapes take the same route,
    because ``@`` broadcasts the first against the batch.

    This used to be an ``einsum`` on generated subscripts, chosen because bringing the
    qubit axes to the front and reshaping copies the stack twice per gate and that
    copying measured more expensive than the arithmetic. The copy is real; the
    conclusion was wrong, and it was wrong for a reason worth writing down.
    ``np.einsum`` without ``optimize=`` runs NumPy's own C kernel, which **never calls
    BLAS**. Reshaping to ``(batch, 2**k, 2**(n-k))`` turns the contraction into a
    batched ``gemm``, and BLAS repays the copy several times over. Microseconds per
    gate on a 32-row stack at 10 qubits:

    | gate | einsum | this |
    |---|---|---|
    | 1-qubit, wire 0 | 190.4 | **82.5** |
    | 1-qubit, wire 5 | 366.0 | **90.9** |
    | 1-qubit, wire 9 | 282.2 | **70.3** |
    | 2-qubit, wires 0,1 | 297.9 | **89.9** |
    | 2-qubit, wires 6,7 | 755.2 | **352.7** |
    | 2-qubit, wires 8,9 | 353.7 | **90.8** |

    The einsum cost also swung by 2.5x with *where the gate sat*, which is the tell: a
    naive nested loop is at the mercy of the stride pattern, and a gemm on a packed
    block is not. The first benchmark that compared the two only tried wires ``(0, 1)``
    and missed it.

    Dropping einsum also drops a 26-letter subscript alphabet, which had capped the
    batched path at 23 qubits independently of ``max_qubits``.
    """
    n = state.ndim - 1
    k = len(qubits)
    batch = state.shape[0]
    axes = [q + 1 for q in qubits]
    front = list(range(1, k + 1))
    # moveaxis leaves the untouched axes in their original relative order and the
    # inverse moveaxis at the end puts them back, which is what keeps this correct for
    # descending and non-adjacent wires -- the case a reindexing bug survives silently.
    stack = np.moveaxis(state, axes, front).reshape(batch, 2**k, -1)
    product = (matrices.reshape(-1, 2**k, 2**k) @ stack).reshape((batch,) + (2,) * n)
    out: npt.NDArray[Any] = np.moveaxis(product, front, axes)
    return out


def _rx_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    c, s = np.cos(a / 2), np.sin(a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = c
    m[:, 0, 1] = m[:, 1, 0] = -1j * s
    return m


def _ry_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    c, s = np.cos(a / 2), np.sin(a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = m[:, 1, 1] = c
    m[:, 0, 1] = -s
    m[:, 1, 0] = s
    return m


def _rz_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    e = np.exp(-1j * a / 2)
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = e
    m[:, 1, 1] = np.conj(e)
    return m


def _phase_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 0, 0] = 1.0
    m[:, 1, 1] = np.exp(1j * a)
    return m


def _controlled_batch(sub: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.tile(np.eye(4, dtype=complex), (sub.shape[0], 1, 1))
    m[:, 2:, 2:] = sub
    return m


#: Closed-form batched builders for the rotations that dominate every ansatz. Each is
#: asserted equal to the scalar ``gate_matrix`` for that gate in ``tests/test_batch.py``
#: — a vectorised matrix that disagrees with the reference by a sign is exactly the
#: plausible-wrong-number bug this library exists to catch.
_VECTORISED: dict[str, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]] = {
    "rx": _rx_batch,
    "ry": _ry_batch,
    "rz": _rz_batch,
    "phase": _phase_batch,
    "crx": lambda a: _controlled_batch(_rx_batch(a)),
    "cry": lambda a: _controlled_batch(_ry_batch(a)),
    "crz": lambda a: _controlled_batch(_rz_batch(a)),
}


_X2 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z2 = np.array([[1, 0], [0, -1]], dtype=complex)


def _d_rotation(pauli: npt.NDArray[Any], builder: Any) -> Any:
    """``dU/dtheta = -i/2 P U`` for a Pauli rotation, for a whole batch at once.

    Derived from the same identity the scalar derivatives use, so it cannot drift
    from them by construction — and it is asserted equal to ``gate_derivative``
    anyway, because "cannot drift" has been wrong before.
    """
    return lambda a: -0.5j * (pauli @ builder(a))


def _d_phase_batch(a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    m = np.zeros((a.size, 2, 2), dtype=complex)
    m[:, 1, 1] = 1j * np.exp(1j * a)
    return m


def _d_controlled_batch(sub: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Only the control-1 block varies, so the control-0 block differentiates to 0."""
    m = np.zeros((sub.shape[0], 4, 4), dtype=complex)
    m[:, 2:, 2:] = sub
    return m


#: Batched gate derivatives, mirroring :data:`_VECTORISED`. Building these one row at
#: a time cost four times the contraction they feed (profiled on a 4-qubit, 4-layer
#: ansatz at batch 128), which made it the bottleneck of the batched adjoint.
_VECTORISED_D: dict[str, Callable[[npt.NDArray[Any]], npt.NDArray[Any]]] = {
    "rx": _d_rotation(_X2, _rx_batch),
    "ry": _d_rotation(_Y2, _ry_batch),
    "rz": _d_rotation(_Z2, _rz_batch),
    "phase": _d_phase_batch,
    "crx": lambda a: _d_controlled_batch(_d_rotation(_X2, _rx_batch)(a)),
    "cry": lambda a: _d_controlled_batch(_d_rotation(_Y2, _ry_batch)(a)),
    "crz": lambda a: _d_controlled_batch(_d_rotation(_Z2, _rz_batch)(a)),
}


def _batched_matrices(op: Op, columns: npt.NDArray[Any] | None) -> npt.NDArray[Any]:
    """One matrix for the whole batch, or one per row when this gate is parameterised.

    ``columns`` holds this op's slot angles, ``(batch, n_params_of_gate)``, or is
    ``None``/empty when the op has no parameterised slot.
    """
    if columns is None or columns.shape[1] == 0:
        literals = tuple(float(p) for p in op.params if not isinstance(p, ParamRef))
        return gate_matrix(op.gate, literals)
    builder = _VECTORISED.get(op.gate)
    if builder is not None and columns.shape[1] == 1:
        return builder(columns[:, 0])
    # a registered custom gate: correctness first, one build per row
    return np.stack([gate_matrix(op.gate, tuple(row)) for row in columns])


class NumpyBackend(Backend):
    """Exact statevector simulation in NumPy."""

    name = "numpy"
    supports_statevector = True
    supports_exact = True

    #: Above this width, :meth:`statevector_batch` falls back to simulating one
    #: sample at a time. Batching trades per-sample Python overhead for worse memory
    #: locality, so it wins while the overhead dominates and loses once ``2**n`` does.
    #:
    #: This was 10, measured against the one-at-a-time loop when the batched kernel was
    #: an ``einsum``. Routing that kernel through BLAS instead (see :func:`_apply_batch`)
    #: made the batched path ~5x faster and moved the boundary with it. Re-measured on a
    #: hardware-efficient ansatz, batched over loop:
    #:
    #:     qubits      9     10     11     12     13     14
    #:     batch 8    5.1x   4.0x   3.1x   0.9x   0.8x   0.6x
    #:     batch 32   8.8x   2.3x   1.8x   1.2x   0.9x   0.6x
    #:     batch 128  5.3x   3.1x   1.9x   1.2x   0.8x   0.5x
    #:
    #: 11 is the last width that wins at *every* batch size. At 12 it depends on the
    #: batch — a win at 32 rows and a loss at 8 — so the boundary does now move with
    #: batch size, which the previous note said it did not. Taking the safe side of a
    #: width that is only sometimes faster is the difference between a default and a
    #: gamble; raise it if you measure otherwise on your own hardware.
    batch_max_qubits = 11

    #: Below this width, fusing adjacent gates into one wider matrix *loses*. The
    #: block matrix is ``2**(2k)`` and the state is ``2**n``, so fusion only pays once
    #: the state dominates: at 6 qubits a ``k=4`` block is 256 complex numbers against
    #: a 64-element state, and **84% of the fused runtime is building blocks**.
    #: Measured against the unfused loop on a hardware-efficient ansatz: 0.68x at 6
    #: qubits, 0.91x at 12, 1.77x at 15, 3.28x at 18, 4.07x at 20.
    fuse_min_qubits = 14

    #: Widest block fusion will build. Also grows with ``n`` — see :meth:`_fuse_width`.
    fuse_max_width = 6

    def __init__(self, seed: int | None = None, max_qubits: int = 24) -> None:
        super().__init__(seed)
        self.max_qubits = max_qubits

    def _fuse_width(self, n_qubits: int) -> int:
        """How wide a fused block to build on ``n`` qubits.

        The best width grows with the register, because what fusion trades is a
        ``2**(2k)`` build against ``2**n`` applies. Measured optimum: ``k=3`` at 12
        qubits, 5 at 15, 6 at 18 and 20 — close enough to ``n // 3`` to use it.
        """
        return max(2, min(self.fuse_max_width, n_qubits // 3))

    def statevector(self, spec: CircuitSpec) -> npt.NDArray[Any]:
        self._check_bound(spec)
        if spec.n_qubits > self.max_qubits:
            raise ValueError(
                f"{spec.n_qubits} qubits exceeds max_qubits={self.max_qubits}; "
                "raise it explicitly if you really mean to allocate that much memory"
            )
        state = np.zeros((2,) * spec.n_qubits, dtype=complex)
        state[(0,) * spec.n_qubits] = 1.0
        if spec.n_qubits >= self.fuse_min_qubits:
            for wires, matrix in self._fused(spec):
                state = _apply(state, matrix, wires)
            return state.reshape(-1)
        for op in spec.ops:
            state = _apply(state, self._matrix(op), op.qubits)
        return state.reshape(-1)

    def _fused(self, spec: CircuitSpec) -> list[tuple[tuple[int, ...], npt.NDArray[Any]]]:
        """Consecutive gates merged into blocks of at most :meth:`_fuse_width` qubits.

        Greedy and order-preserving, which is what makes it obviously correct: a gate
        joins the current block whenever the union of their wires still fits, and
        otherwise starts a new one. No reordering, so no commutation argument is
        needed — and a fusion bug would be a *silent wrong number*, which is the one
        class of defect this library cannot ship.

        Each block is composed by applying its gates to the output legs of an
        identity, using the same contraction the state path uses. A convention error
        here would have to be a convention error there too.
        """
        width = self._fuse_width(spec.n_qubits)
        out: list[tuple[tuple[int, ...], npt.NDArray[Any]]] = []
        current: list[Op] = []
        wires: set[int] = set()

        def flush() -> None:
            if not current:
                return
            order = tuple(sorted(wires))
            if len(current) == 1:
                out.append((current[0].qubits, self._matrix(current[0])))
            else:
                k = len(order)
                index = {q: i for i, q in enumerate(order)}
                block = np.eye(2**k, dtype=complex).reshape((2,) * (2 * k))
                for held in current:
                    block = _apply(block, self._matrix(held), tuple(index[q] for q in held.qubits))
                out.append((order, block.reshape(2**k, 2**k)))

        for op in spec.ops:
            candidate = wires | set(op.qubits)
            if current and len(candidate) > width:
                flush()
                current, wires = [op], set(op.qubits)
            else:
                current.append(op)
                wires = candidate
        flush()
        return out

    def statevector_batch_slots(
        self, spec: CircuitSpec, slot_angles: npt.NDArray[Any]
    ) -> npt.NDArray[Any]:
        """Every row's state in one pass, batch carried as a leading axis.

        The loop is over *gates*, not over rows: each gate is applied to the whole
        stack at once. A gate with literal angles contributes one matrix; a
        parameterised one contributes a stack of them, read straight off the slot
        columns and built in closed form for the common rotations.
        """
        rows = np.atleast_2d(np.asarray(slot_angles, dtype=float))
        n_slots = len(spec.slots())
        if rows.shape[1] != n_slots:
            raise ValueError(
                f"circuit has {n_slots} slot(s); got vectors of length {rows.shape[1]}"
            )
        if spec.n_qubits > self.max_qubits:
            raise ValueError(
                f"{spec.n_qubits} qubits exceeds max_qubits={self.max_qubits}; "
                "raise it explicitly if you really mean to allocate that much memory"
            )
        if spec.n_qubits > self.batch_max_qubits:
            # wider than the crossover: the one-at-a-time loop is genuinely faster
            return super().statevector_batch_slots(spec, rows)

        batch = rows.shape[0]
        state = np.zeros((batch,) + (2,) * spec.n_qubits, dtype=complex)
        state[(slice(None),) + (0,) * spec.n_qubits] = 1.0

        # slots are ordered by (op_index, param_pos), so walking them in step with the
        # ops keeps the cursor aligned without a lookup per gate
        slots = spec.slots()
        cursor = 0
        for op_index, op in enumerate(spec.ops):
            n_here = 0
            while cursor + n_here < len(slots) and slots[cursor + n_here].op_index == op_index:
                n_here += 1
            columns = rows[:, cursor : cursor + n_here] if n_here else None
            cursor += n_here
            state = _apply_batch(state, _batched_matrices(op, columns), op.qubits)
        return state.reshape(batch, -1)

    @staticmethod
    def _matrix(op: Op) -> npt.NDArray[Any]:
        angles: list[float] = []
        for p in op.params:
            if isinstance(p, ParamRef):  # pragma: no cover - is_bound rules this out
                raise ValueError(f"unbound parameter reached the backend in {op.gate!r}")
            angles.append(float(p))
        return gate_matrix(op.gate, tuple(angles))
