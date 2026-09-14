r"""What it costs to get the data in — and where the exponential speedup went.

Amplitude encoding puts ``N`` numbers into ``log2(N)`` qubits, and that compression is
real: 1024 features in 10 qubits. It is also the most quoted and least examined number
in quantum machine learning, because the qubits are not what you pay in.

**Measured, in this library:** preparing a 1024-amplitude state costs 4,052 CNOTs and
a depth of 6,027 — 3.96 two-qubit gates per feature, converging on ``4N`` as ``N``
grows. Loading the data is linear in the data. An algorithm that then runs in
``O(polylog N)`` has not beaten a classical one that reads the same vector: it has
moved the linear cost from the algorithm into the state preparation and stopped
counting it.

QRAM, and what it would have to be
----------------------------------
The device that would fix this is QRAM: bucket-brigade memory that returns
``sum_i psi_i |i>|x_i>`` in ``O(log N)`` *depth* (Giovannetti, Lloyd and Maccone 2008).
Every exponential-speedup claim built on amplitude-encoded classical data — HHL,
qPCA, quantum recommendation — assumes one.

No such device exists beyond toy demonstrations, and the accounting is worse than
"not yet built":

* It is ``O(log N)`` in **depth**, never in total resources. Holding ``N`` values needs
  ``Θ(N)`` memory cells and ``Θ(N)`` routing components, all coherent at once. There is
  no regime in which ``N`` numbers live in fewer than ``N`` places.
* The depth advantage assumes every one of those ``Θ(N)`` components acts in parallel
  and none of them decoheres.
* Under error correction the per-cell overhead multiplies that ``Θ(N)``.
* And where the access model was the whole source of the advantage, the advantage was
  not quantum: given the classical analogue of QRAM access — length-squared sampling —
  classical algorithms match the quantum ones to polynomial factors for recommendation
  systems, PCA and low-rank linear systems (Tang 2019, and the dequantisation results
  that followed).

So this module ships a cost model rather than a QRAM. :func:`loading_cost` prices the
encoders qmlkit actually has, from their measured gate counts; :func:`qram_cost` prices
the device you would need instead, so the comparison is in front of you rather than
in a footnote. Neither is discouraging by itself — amplitude encoding is the right
choice for a state you *compute* rather than load, and for that case the ``Θ(N)``
never appears.

What to use instead, when the data is classical
-----------------------------------------------
Angle encoding costs one rotation per feature and no two-qubit gates at all, at one
qubit per feature. That is the trade: amplitude encoding is exponential in space and
linear in time, angle encoding is linear in space and constant in depth. For the
feature counts a simulator can reach, and for every NISQ device, the second is usually
the honest choice — which is why :func:`~qmlkit.encoding.angle.angle_encode` and
:class:`~qmlkit.encoding.feature_maps.AngleFeatureMap` are the library's defaults and
amplitude encoding is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from qmlkit.utils.errors import unknown

__all__ = ["LoadingCost", "loading_cost", "QramCost", "qram_cost"]

#: Two-qubit gates per feature for amplitude encoding, measured on this library's
#: `amplitude_encode` over N = 2..1024: the ratio climbs 1.00, 2.00, 2.75, ... 3.96 and
#: converges on 4. Below N = 64 the constant is smaller and this overestimates, which
#: is the direction an estimate should err in.
_CNOT_PER_AMPLITUDE = 4.0

#: Total operations per feature, measured the same way (6,044 ops at N = 1024).
_OPS_PER_AMPLITUDE = 6.0


@dataclass(frozen=True)
class LoadingCost:
    """Qubits, gates and depth to get one sample into a circuit.

    ``exact`` distinguishes a counted circuit from a fitted estimate: below a few
    hundred features :func:`loading_cost` builds the circuit and counts it, and above
    that it extrapolates from the measured constants rather than allocating a
    ``2^n``-amplitude statevector to find out.

    ``two_qubit_gates`` is structural — every generic vector of the same length gives
    the same count. ``depth`` moves by a percent or two with the particular angles
    (measured: 1,458 to 1,484 over six random 256-vectors), so read it as a scale
    rather than a guarantee.
    """

    method: str
    n_features: int
    n_qubits: int
    two_qubit_gates: int
    depth: int
    exact: bool

    @property
    def gates_per_feature(self) -> float:
        return self.two_qubit_gates / max(self.n_features, 1)

    def __str__(self) -> str:
        how = "counted" if self.exact else "estimated from measured scaling"
        return (
            f"{self.method} encoding of {self.n_features:,} features ({how})\n"
            f"  qubits              {self.n_qubits:,}\n"
            f"  two-qubit gates     {self.two_qubit_gates:,}"
            f"   ({self.gates_per_feature:.2f} per feature)\n"
            f"  depth               {self.depth:,}"
        )


def loading_cost(n_features: int, method: str = "amplitude") -> LoadingCost:
    """Price one sample's encoding, by counting the circuit or extrapolating from it.

    The three methods trade the same budget in different directions:

    =========== ================= =================== =========================
    Method      Qubits            Two-qubit gates     Depth
    =========== ================= =================== =========================
    ``angle``   ``n``             0                   1
    ``basis``   ``n``             0                   1
    ``amplitude`` ``ceil(log2 n)`` ``~4n``            ``~6n``
    =========== ================= =================== =========================

    Amplitude encoding is the only one that compresses, and the only one whose cost
    grows with the data. That is not an implementation detail to be optimised away:
    ``Θ(n)`` two-qubit gates is a lower bound for preparing an arbitrary state
    (Plesch and Brukner 2011), so no better circuit exists to find.

    The amplitude figure is for a **generic** vector. A real, non-negative one — which
    many feature vectors are — needs no phase preparation and costs exactly half:
    measured, 2,026 CNOTs against 4,052 at 1024 features. This reports the generic
    number, so treat it as an upper bound and halve it if your data has no signs.

    Examples
    --------
    >>> import qmlkit as qk
    >>> qk.loading_cost(1024, "amplitude").n_qubits
    10
    >>> qk.loading_cost(1024, "angle").two_qubit_gates
    0
    """
    if n_features <= 0:
        raise ValueError(f"n_features must be positive, got {n_features}")

    if method in ("angle", "basis"):
        return LoadingCost(method, n_features, n_features, 0, 1, exact=True)

    if method != "amplitude":
        raise unknown(
            "encoding method",
            method,
            ("amplitude", "angle", "basis"),
            hint=" These are the methods with a cost model; the feature maps in "
            "qmlkit.encoding.feature_maps are priced by building them and reading "
            "spec.resources().",
        )

    n_qubits = max(int(math.ceil(math.log2(n_features))), 1)
    padded = 2**n_qubits
    # Under ~512 amplitudes, building the circuit and counting it is cheap and exact;
    # past that the statevector itself is the expensive part, so use the constants.
    if padded <= 512:
        import numpy as np

        from qmlkit.encoding.amplitude import amplitude_encode

        # A *generic* vector, not np.ones. The count is structural and identical for
        # any generic input, but a real non-negative one needs no phase preparation
        # and costs exactly half -- 2,026 CNOTs against 4,052 at N = 1024. Counting
        # the easy case and reporting it as the price is how a cost model flatters.
        probe = np.random.default_rng(0).normal(size=padded)
        resources = amplitude_encode(probe).resources()
        two_qubit, depth = resources["n_2q"], resources["depth"]
        # resources() is typed dict[str, object]; these two entries are always counts.
        assert isinstance(two_qubit, int) and isinstance(depth, int)
        return LoadingCost(method, n_features, n_qubits, two_qubit, depth, exact=True)
    return LoadingCost(
        method,
        n_features,
        n_qubits,
        int(_CNOT_PER_AMPLITUDE * padded),
        int(_OPS_PER_AMPLITUDE * padded),
        exact=False,
    )


@dataclass(frozen=True)
class QramCost:
    """What a bucket-brigade QRAM would need, against what encoding needs instead."""

    n_addresses: int
    address_qubits: int
    memory_qubits: int
    routing_qubits: int
    ideal_depth: int
    encoding: LoadingCost

    @property
    def total_qubits(self) -> int:
        return self.address_qubits + self.memory_qubits + self.routing_qubits

    def __str__(self) -> str:
        return (
            f"bucket-brigade QRAM over {self.n_addresses:,} addresses\n"
            f"  address qubits      {self.address_qubits:,}\n"
            f"  memory cells        {self.memory_qubits:,}\n"
            f"  routing components  {self.routing_qubits:,}\n"
            f"  total qubits        {self.total_qubits:,}"
            "   <- the Theta(N) the log-depth claim does not mention\n"
            f"  ideal depth         {self.ideal_depth:,}"
            "   <- only if all of the above act in parallel, coherently\n"
            "\n"
            "  no such device exists; without one, the same data is loaded by state\n"
            "  preparation, which costs:\n"
            f"    {self.encoding.two_qubit_gates:,} two-qubit gates, depth "
            f"{self.encoding.depth:,}, on {self.encoding.n_qubits} qubits\n"
            "\n"
            "  Either way you pay Theta(N) somewhere -- "
            f"{self.total_qubits:,} qubits with the QRAM,\n"
            f"  depth {self.encoding.depth:,} without it. The exponential compression is in\n"
            "  space alone, and only for state preparation."
        )


def qram_cost(n_addresses: int) -> QramCost:
    """Resources a bucket-brigade QRAM would need, beside what encoding needs instead.

    The point of the comparison is the qubit count. ``O(log N)`` is a claim about
    depth, and it holds only on hardware holding ``Θ(N)`` coherent memory cells and
    ``Θ(N)`` routing components — so the exponential saving never existed in total
    resources, only in how many of them run at once.

    This prices the architecture. It does not build one, and qmlkit has no QRAM
    backend: a simulated QRAM is a ``Θ(N)``-qubit circuit whose only honest use is to
    demonstrate that cost.

    Examples
    --------
    >>> import qmlkit as qk
    >>> cost = qk.qram_cost(1024)
    >>> cost.address_qubits
    10
    >>> cost.memory_qubits
    1024
    """
    if n_addresses <= 0:
        raise ValueError(f"n_addresses must be positive, got {n_addresses}")
    address = max(int(math.ceil(math.log2(n_addresses))), 1)
    # A bucket-brigade tree has one routing node per internal vertex: N - 1 of them.
    return QramCost(
        n_addresses=n_addresses,
        address_qubits=address,
        memory_qubits=n_addresses,
        routing_qubits=max(n_addresses - 1, 0),
        ideal_depth=address,
        encoding=loading_cost(n_addresses, "amplitude"),
    )
