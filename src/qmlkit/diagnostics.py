"""Whether the model you just built is quietly broken.

In most libraries a mistake raises. In quantum machine learning it returns a
number — right shape, right range, entirely plausible, and wrong:

* A re-uploading model whose trainable block commutes with its encoding trains,
  converges, and reaches one Fourier frequency instead of the ``L`` it was
  designed for. Every weight in it is a phase shift.
* An ansatz with a parameter the circuit cannot feel fits exactly as well without
  it. The optimiser reports no difficulty, because there is none.
* A kernel that has concentrated gives every pair of inputs the same similarity,
  and still produces a Gram matrix, an SVM, and an accuracy.

None of these are exceptions to catch. They are properties to measure, and this
module measures them.

The checks are deliberately *decisive* rather than exhaustive. Each one has a
threshold that separates "wrong" from "unusual" with room to spare, because a
diagnostic that cries wolf is one nobody runs twice. Where the check is exact —
a parameter that cannot change the state is dead, full stop — the threshold is
machine epsilon. Where it is statistical, the finding says what was measured, so
the number can be argued with.

    >>> import qmlkit as qk
    >>> report = qk.diagnose(qk.hardware_efficient(3, 2))
    >>> bool(report)
    False

A report is falsy when it found nothing, so ``if qk.diagnose(model): ...`` reads
the way it should. Findings carry a stable ``code`` to branch on, a ``message``
saying what is wrong, and a ``fix`` naming the edit that resolves it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qmlkit.ansatz.blocks import Block, EncodingLayer, RotationLayer
from qmlkit.ansatz.library import Ansatz
from qmlkit.core.execute import BackendLike, statevector
from qmlkit.core.observables import Observable, Z
from qmlkit.info import state_fidelity
from qmlkit.kernels.matrix import is_psd, kernel_spread, min_eigenvalue, shots_to_resolve
from qmlkit.metrics import entangling_capability, gradient_variance
from qmlkit.utils.shots import shots_for_precision

__all__ = ["Finding", "Report", "diagnose"]

#: A parameter shifted by this much either moves the state or does not exist.
#: 0.7 rad is far from every period in the gate set, so no gate returns to itself.
_PROBE_SHIFT = 0.7

#: Fidelity is exact on a statevector simulator, so "no change" means no change.
_EXACT = 1e-12

#: Below this, every pair of inputs has the same similarity to three decimals and
#: no model built on the matrix can separate them.
_CONCENTRATED = 1e-3

#: Gradient variance below this costs more than ~1000 shots per gradient entry to
#: resolve, which is where shot cost starts to dominate the training budget.
#: Calibrated against hardware-efficient ansaetze from 2 to 10 qubits and 2 to 20
#: layers, whose measured variance runs from 3e-1 down to 5e-4: this fires on the
#: deep-and-wide end and on global observables, and not on the rest.
_FLAT = 1e-3

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    """One thing that is wrong, why it matters, and the edit that fixes it."""

    code: str
    severity: str
    message: str
    fix: str = ""
    value: float | None = None

    def __str__(self) -> str:
        tail = f"  Fix: {self.fix}" if self.fix else ""
        return f"[{self.severity}] {self.code}: {self.message}{tail}"


@dataclass(frozen=True)
class Report:
    """Everything :func:`diagnose` found, worst first.

    Falsy when empty, so it can be tested directly. Iterating yields
    :class:`Finding` objects; ``codes`` is the flat list to assert against.
    """

    subject: str
    findings: tuple[Finding, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.findings)

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    def __getitem__(self, index: int) -> Finding:
        return self.findings[index]

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(f.code for f in self.findings)

    @property
    def errors(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "error")

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.severity == "warning")

    def __str__(self) -> str:
        if not self.findings:
            return f"{self.subject}: nothing to report."
        head = f"{self.subject}: {len(self.findings)} finding(s)"
        return "\n".join([head, *(f"  {f}" for f in self.findings)])


# --------------------------------------------------------------------------- #
# walking the block tree
# --------------------------------------------------------------------------- #
def _walk(block: Block, times: int = 1) -> Iterator[tuple[Block, int]]:
    """Every block in the tree, with how many times the circuit actually emits it.

    ``Sequential`` holds ``blocks``; ``Repeat`` and ``Share`` hold one ``block`` and
    emit it ``times`` times. Counting tree nodes would therefore understate the
    circuit — ``repeat(3, EncodingLayer(fmap) + ...)`` is one node and three
    uploads — and undercounting uploads is exactly how the collapse below hides.
    """
    yield block, times
    factor = times * int(getattr(block, "times", 1))
    for attr in ("blocks", "block"):
        child = getattr(block, attr, None)
        if isinstance(child, Block):
            yield from _walk(child, factor)
        elif isinstance(child, tuple):
            for item in child:
                if isinstance(item, Block):
                    yield from _walk(item, factor)


# --------------------------------------------------------------------------- #
# the checks
# --------------------------------------------------------------------------- #
def _dead_parameters(
    ansatz: Ansatz, *, probes: int, seed: int | None, backend: BackendLike
) -> npt.NDArray[Any]:
    """Indices whose value cannot change the state, so cannot change any result.

    Exact rather than statistical: shift one parameter and compare states. Fidelity
    ignores global phase, which is the right blindness — a parameter that only
    moves the global phase is unobservable, and so is genuinely dead.

    A parameter proven live is never probed again, so a healthy circuit costs one
    round of ``n_params`` statevectors rather than ``probes`` rounds of them.
    """
    rng = np.random.default_rng(seed)
    spec = ansatz.build()
    n = ansatz.n_params
    alive = np.zeros(n, dtype=bool)
    for _ in range(probes):
        theta = rng.uniform(-np.pi, np.pi, n)
        base = statevector(spec, theta, backend=backend)
        for i in np.flatnonzero(~alive):
            shifted = theta.copy()
            shifted[i] += _PROBE_SHIFT
            moved = statevector(spec, shifted, backend=backend)
            if abs(1.0 - state_fidelity(base, moved)) > _EXACT:
                alive[i] = True
        if alive.all():
            break
    return np.flatnonzero(~alive)


def _encoding_collapses(ansatz: Ansatz) -> tuple[int, str] | None:
    """``(n_uploads, rotation)`` when every trainable rotation shares the encoding's.

    ``Ry(x) Ry(t1) Ry(x) Ry(t2) = Ry(2x + t1 + t2)``: the uploads merge into one
    rotation, so the model reaches a single frequency and the weights become a
    phase. :func:`~qmlkit.ansatz.reupload.reupload` warns about this at
    construction; a model composed by hand out of blocks does not, and that is the
    case this catches.
    """
    encodings = [(b, n) for b, n in _walk(ansatz.block) if isinstance(b, EncodingLayer)]
    uploads = sum(n for _, n in encodings)
    if uploads < 2:
        return None
    generators = {getattr(b.feature_map, "rotation", None) for b, _ in encodings}
    if len(generators) != 1 or None in generators:
        return None  # a multi-gate map (ZZ, Pauli) never fully commutes
    rotations = {
        g for b, _ in _walk(ansatz.block) if isinstance(b, RotationLayer) for g in b.gates
    }
    encoding = str(generators.pop())
    return (uploads, encoding) if rotations and rotations <= {encoding} else None


def _diagnose_ansatz(
    ansatz: Ansatz,
    *,
    obs: Observable | None,
    n_samples: int,
    probes: int,
    seed: int | None,
    backend: BackendLike,
) -> list[Finding]:
    found: list[Finding] = []

    if ansatz.n_weights == 0:
        found.append(
            Finding(
                "NO_TRAINABLE_PARAMETERS",
                "error",
                f"{ansatz.name!r} has no weights: every parameter is a reserved input slot.",
                fix="Compose a trainable block into it, e.g. + RotationLayer('ry').",
                value=0.0,
            )
        )

    collapse = _encoding_collapses(ansatz)
    if collapse is not None:
        uploads, rotation = collapse
        found.append(
            Finding(
                "ENCODING_COMMUTES",
                "error",
                f"{uploads} uploads, but every trainable rotation is {rotation!r}, the same "
                f"generator the encoding uses. {rotation.upper()}(x) {rotation.upper()}(t) "
                "composes into one rotation, so the model reaches 1 frequency rather than "
                f"0..{uploads}, and its weights do nothing beyond a phase.",
                fix="Use a non-commuting block, e.g. RotationLayer(('rz', 'ry', 'rz')).",
                value=float(uploads),
            )
        )

    dead = _dead_parameters(ansatz, probes=probes, seed=seed, backend=backend)
    dead_inputs = [int(i) for i in dead if i < ansatz.n_inputs]
    dead_weights = [int(i) - ansatz.n_inputs for i in dead if i >= ansatz.n_inputs]

    if dead_inputs:
        found.append(
            Finding(
                "INPUTS_UNUSED",
                "error",
                f"input slot(s) {dead_inputs} cannot change the state, so those features "
                "reach the circuit and are then discarded. The model is blind to them.",
                fix="Check the feature map's width against n_inputs, and that its angles "
                "drive gates on wires the observable can see.",
                value=float(len(dead_inputs)),
            )
        )

    if dead_weights:
        found.append(
            Finding(
                "DEAD_WEIGHTS",
                "warning",
                f"{len(dead_weights)} of {ansatz.n_weights} weights cannot change the state "
                f"(indices {dead_weights[:8]}{'...' if len(dead_weights) > 8 else ''}). They "
                "are optimised over and cannot affect any result.",
                fix="Usually a rotation whose generator already fixes the state it acts on "
                "(Rz on |0>), or a layer past the last gate that reaches a measured wire.",
                value=float(len(dead_weights)),
            )
        )

    if ansatz.n_qubits > 1:
        q = entangling_capability(
            ansatz, n_samples=max(20, n_samples // 4), seed=seed, backend=backend
        )
        if q < _EXACT:
            found.append(
                Finding(
                    "NO_ENTANGLEMENT",
                    "warning",
                    f"{ansatz.n_qubits} qubits, but the Meyer-Wallach measure is zero to "
                    "floating point: the circuit only ever produces product states, so it is a "
                    "set of independent one-qubit models and a laptop can simulate it exactly "
                    "at any width.",
                    fix="Add an entangler, e.g. + EntanglerLayer('cz', 'ring').",
                    value=float(q),
                )
            )

    live_weights = [
        i for i in range(ansatz.n_inputs, ansatz.n_params) if i not in set(dead.tolist())
    ]
    if live_weights:
        var = gradient_variance(
            ansatz,
            obs,
            n_samples=max(20, n_samples),
            param_index=live_weights[0],
            seed=seed,
            backend=backend,
        )
        if 0.0 < var < _FLAT:
            budget = shots_for_precision(float(np.sqrt(var)))
            found.append(
                Finding(
                    "FLAT_GRADIENTS",
                    "warning",
                    f"gradient variance {var:.2e} for weight {live_weights[0] - ansatz.n_inputs} "
                    f"at random initialisation (observable {obs or Z(0)}), so a typical entry is "
                    f"around {np.sqrt(var):.1e} and resolving one against shot noise would take "
                    f"about {budget:,} shots. Gradients are exact here, so this is what the same "
                    "model would cost on a sampling device, not a failure now.",
                    fix="Initialise near identity with ansatz.init('small'), reduce depth, or "
                    "measure a local observable. Whether it is a barren plateau rather than a "
                    "merely small gradient is a question about scaling: metrics.barren_plateau_"
                    "scan answers it across widths.",
                    value=float(var),
                )
            )

    return found


def _diagnose_kernel(
    gram: npt.NDArray[Any], *, n_qubits: int | None, shots: int | None
) -> list[Finding]:
    arr = np.asarray(gram, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"a Gram matrix must be square, got shape {arr.shape}. Build one with "
            "kernel_matrix(X) or QuantumKernel(...).matrix(X)."
        )
    if arr.shape[0] < 2:
        return []

    found: list[Finding] = []
    off = arr[~np.eye(arr.shape[0], dtype=bool)]
    spread = float(off.std())

    if not is_psd(arr):
        low = min_eigenvalue(arr)
        found.append(
            Finding(
                "KERNEL_NOT_PSD",
                "warning",
                f"minimum eigenvalue {low:.3e}: the matrix is not positive semi-definite, so "
                "it is not a kernel and an SVM's dual problem is no longer convex.",
                fix="K = qk.closest_psd_matrix(K)  # or method='displace' / 'flip'",
                value=float(low),
            )
        )

    if spread < _CONCENTRATED:
        found.append(
            Finding(
                "KERNEL_CONCENTRATED",
                "error",
                f"off-diagonal spread {spread:.2e}: every pair of inputs has effectively the "
                "same similarity, so no model built on this matrix can separate them.",
                fix="Reduce the qubit count or the encoding depth, scale the features down, or "
                "use projected_kernel_matrix, which resists concentration.",
                value=spread,
            )
        )

    if shots is not None:
        noise = float(np.sqrt(0.25 / shots))
        if spread <= noise:
            found.append(
                Finding(
                    "KERNEL_UNRESOLVABLE",
                    "error",
                    f"the signal in this matrix ({spread:.2e}) is at or below the shot noise at "
                    f"{shots:,} shots ({noise:.2e}), so what is being measured is the sampling "
                    "error, not the kernel.",
                    fix=f"Raise shots above {int(np.ceil(0.25 / spread**2)):,} if the spread is "
                    "real, or fix the concentration instead.",
                    value=spread,
                )
            )

    if n_qubits is not None and spread < 2 * kernel_spread(n_qubits):
        found.append(
            Finding(
                "KERNEL_AT_CONCENTRATION_SCALE",
                "info",
                f"spread {spread:.2e} is at or below the 2^-n scale predicted for {n_qubits} "
                f"qubits ({kernel_spread(n_qubits):.2e}), which is what exponential "
                "concentration looks like arriving.",
                fix=f"On hardware this would need about {shots_to_resolve(n_qubits):,} shots. "
                "Check with kernels.concentration_report before scaling the width up.",
                value=spread,
            )
        )

    return found


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def _find_ansatz(subject: object) -> Ansatz | None:
    """The ansatz inside a model, wherever a torch module happens to keep it."""
    if isinstance(subject, Ansatz):
        return subject
    direct = getattr(subject, "ansatz", None)
    if isinstance(direct, Ansatz):
        return direct
    children = getattr(subject, "modules", None)
    if callable(children):  # an nn.Module: the layer may be nested at any depth
        for module in children():
            held = getattr(module, "ansatz", None)
            if isinstance(held, Ansatz):
                return held
    return None


def diagnose(
    subject: object,
    *,
    obs: Observable | None = None,
    n_samples: int = 30,
    probes: int = 3,
    seed: int | None = 0,
    backend: BackendLike = None,
    shots: int | None = None,
    n_qubits: int | None = None,
) -> Report:
    """Check a model or a Gram matrix for the failures that do not raise.

    Parameters
    ----------
    subject
        An :class:`~qmlkit.ansatz.library.Ansatz`, anything holding one (a
        ``QuantumLayer``, ``VQC``, ``VQRegressor``, or an ``nn.Sequential``
        containing one), or a square Gram matrix.
    obs
        Observable for the trainability probe. Defaults to ``Z(0)``, matching
        :func:`~qmlkit.metrics.barren_plateau_scan`.
    n_samples
        Sample count for the statistical checks — entanglement and gradient
        variance. The exact checks ignore it.
    probes
        Random points at which to test whether a parameter can move the state. A
        parameter is dead if it moves nothing at any of them; three is already
        conclusive, since the points are random and the test is exact.
    shots, n_qubits
        Gram matrices only. ``shots`` enables the check for whether the signal
        survives sampling noise; ``n_qubits`` enables the comparison against the
        ``2^-n`` concentration scale.

    Returns
    -------
    Report
        Falsy when nothing was found. Sorted worst first.

    Examples
    --------
    >>> import qmlkit as qk
    >>> healthy = qk.diagnose(qk.hardware_efficient(3, 2))
    >>> bool(healthy)
    False

    A model whose weights share the encoding's generator is the trap the
    re-uploading literature warns about, and it is silent without this:

    >>> from qmlkit.ansatz import Ansatz, EncodingLayer, RotationLayer, repeat
    >>> fmap = qk.AngleFeatureMap(1, rotation="ry")
    >>> block = EncodingLayer(fmap) + RotationLayer("ry")
    >>> broken = qk.diagnose(Ansatz(1, repeat(3, block), n_inputs=1))
    >>> "ENCODING_COMMUTES" in broken.codes
    True
    """
    if isinstance(subject, np.ndarray) or (
        isinstance(subject, list | tuple) and subject and isinstance(subject[0], list | tuple)
    ):
        gram = np.asarray(subject, dtype=float)
        return Report(
            f"Gram matrix {gram.shape[0]}x{gram.shape[0]}",
            _sorted(_diagnose_kernel(gram, n_qubits=n_qubits, shots=shots)),
        )

    ansatz = _find_ansatz(subject)
    if ansatz is None:
        raise TypeError(
            f"diagnose() cannot inspect a {type(subject).__name__}. It takes an Ansatz, "
            "anything holding one (QuantumLayer, VQC, VQRegressor, or an nn.Sequential "
            "containing one), or a Gram matrix as a square array."
        )
    findings = _diagnose_ansatz(
        ansatz, obs=obs, n_samples=n_samples, probes=probes, seed=seed, backend=backend
    )
    label = f"{type(subject).__name__} ({ansatz.name})" if subject is not ansatz else ansatz.name
    return Report(f"{label} on {ansatz.n_qubits} qubits", _sorted(findings))


def _sorted(findings: list[Finding]) -> tuple[Finding, ...]:
    return tuple(sorted(findings, key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.code)))
