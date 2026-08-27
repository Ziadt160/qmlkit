"""What running qmlkit on a real device would actually take.

    python examples/toward_hardware.py

qmlkit is simulator-only for the whole 0.x line. That is a scope decision, not a
technical wall: the pieces that make hardware possible are already here, and this
script demonstrates them against a **mock device** that behaves like a QPU — no
statevector, shots only, readout error, and a counter for every circuit submitted.

What it shows, in order:

1. the backend abstraction genuinely supports a device (one method, ``counts``);
2. the methods that cannot work on hardware refuse rather than silently degrade;
3. a real training run completes on shots alone and still learns;
4. what that costs in circuits, which is the number that decides feasibility;
5. what is honestly still missing.
"""

from __future__ import annotations

import numpy as np

import qmlkit as qk
from qmlkit.core.backends.base import Backend
from qmlkit.core.observables import as_sum, group_qubit_wise_commuting

SHOTS = 1024
READOUT_ERROR = 0.02


def header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
# A device, as far as qmlkit is concerned
# --------------------------------------------------------------------------- #
class MockDevice(Backend):
    """Everything a QPU is, and nothing it is not.

    A real device cannot hand you amplitudes and cannot give you a shot-free
    expectation value. It can run a circuit and report bitstrings. That is exactly
    one method — ``counts`` — and the base class derives basis rotation, term
    grouping and expectation values from it, so this backend inherits measurement
    semantics identical to every simulator's.
    """

    name = "mock_device"
    supports_statevector = False
    supports_exact = False

    def __init__(self, seed: int | None = None, readout_error: float = READOUT_ERROR) -> None:
        super().__init__(seed)
        self.readout_error = readout_error
        self.circuits_run = 0
        self.shots_used = 0

    def counts(self, spec, shots, seed=None):
        self._check_bound(spec)
        self.circuits_run += 1
        self.shots_used += shots
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        # a real device would submit the circuit here; we sample it and then corrupt
        # the bitstrings, which is the dominant error on current hardware
        probs = np.abs(qk.get_backend("numpy").statevector(spec)) ** 2
        out: dict[str, int] = {}
        for index, n in enumerate(rng.multinomial(shots, probs / probs.sum())):
            if not n:
                continue
            bits = format(index, f"0{spec.n_qubits}b")
            for _ in range(int(n)):
                noisy = "".join(
                    b if rng.random() > self.readout_error else ("1" if b == "0" else "0")
                    for b in bits
                )
                out[noisy] = out.get(noisy, 0) + 1
        return out


device = MockDevice(seed=0)
ansatz = qk.hardware_efficient(2, n_layers=2)
spec, theta0 = ansatz.build(), ansatz.init("uniform", seed=3)
OBS = qk.Z(0) + 0.5 * qk.ZZ(0, 1)

# --------------------------------------------------------------------------- #
header("1. A device is one method")
# --------------------------------------------------------------------------- #
print(
    f"  {type(device).__name__}: statevector={device.supports_statevector}"
    f"  exact={device.supports_exact}  readout error={device.readout_error:.0%}"
)
exact = qk.expval(spec, OBS, theta=theta0)
sampled = qk.expectation(spec, OBS, theta=theta0, shots=SHOTS, backend=device)
print(f"  <O> exact (simulator)      {exact:+.6f}")
print(
    f"  <O> device, {SHOTS} shots    {sampled:+.6f}"
    f"   (shot noise + {device.readout_error:.0%} readout error)"
)

# --------------------------------------------------------------------------- #
header("2. What a device cannot do, it refuses")
# --------------------------------------------------------------------------- #
for label, call in [
    ("an exact, shot-free expectation", lambda: qk.expval(spec, OBS, theta=theta0, backend=device)),
    (
        "adjoint differentiation",
        lambda: qk.grad(spec, theta0, OBS, method="adjoint", backend=device),
    ),
    ("a statevector", lambda: qk.statevector(spec.bind(theta0), backend=device)),
]:
    try:
        call()
        print(f"  {label:<34} SILENTLY SUCCEEDED — that would be a bug")
    except Exception as exc:  # noqa: BLE001 - showing exactly what a user would see
        print(f"  {label:<34} {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")

print("\n  Nothing here degrades quietly into a wrong answer. The methods that need")
print("  amplitudes say so, and parameter-shift and the Hadamard test remain.")

# --------------------------------------------------------------------------- #
header("3. Train on shots alone")
# --------------------------------------------------------------------------- #
X, y_binary = qk.datasets.make_moons(n_samples=20, noise=0.1, seed=0)
y = 2.0 * y_binary - 1.0
X = qk.AngleScaler().fit(X).transform(X)

model = qk.reupload(
    qk.AngleFeatureMap(2, entangle=False),
    n_layers=1,
    block=qk.repeat(2, qk.RotationLayer(("ry", "rz")) + qk.EntanglerLayer("cx", "chain")),
)
model_spec = model.build()
P = ansatz.n_params
STEPS = 8
LR = 0.5


def loss_exact(theta: np.ndarray) -> float:
    """Scored on the simulator, so the learning curve is not itself noisy."""
    preds = np.array(
        [qk.expval(model_spec, OBS, theta=np.concatenate([model.angles(x), theta])) for x in X]
    )
    return float(np.mean((preds - y) ** 2))


theta = theta0.copy()
history = [loss_exact(theta)]
device.circuits_run = device.shots_used = 0
for _ in range(STEPS):
    total = np.zeros(P)
    for x, target in zip(X, y, strict=True):
        full = np.concatenate([model.angles(x), theta])
        pred = qk.expectation(model_spec, OBS, theta=full, shots=SHOTS, backend=device)
        grad = qk.grad(model_spec, full, OBS, method="parameter-shift", shots=SHOTS, backend=device)
        total += 2.0 * (pred - target) * grad[model.n_inputs :]
    theta = theta - LR * total / len(X)
    history.append(loss_exact(theta))

print(
    f"  {STEPS} steps, {len(X)} samples, parameter-shift at {SHOTS} shots,"
    f" {READOUT_ERROR:.0%} readout error"
)
print(f"  loss {history[0]:.6f} -> {history[-1]:.6f}")
print(
    f"  {'it still learns' if history[-1] < history[0] else 'it did NOT learn'},"
    f" on sampled gradients from a noisy device"
)

# --------------------------------------------------------------------------- #
header("4. The number that decides feasibility")
# --------------------------------------------------------------------------- #
per_step = device.circuits_run / STEPS
print(f"  circuits submitted   {device.circuits_run:,}")
print(f"  shots consumed       {device.shots_used:,}")
print(
    f"  circuits per step    {per_step:,.0f}   ({len(X)} samples x"
    f" {qk.gradient_cost(model_spec, 'parameter-shift')} shift circuits, grouped)"
)

print("\n  Qubit-wise-commuting grouping is doing real work here: Z0 and Z0Z1 are both")
print("  diagonal in Z, so they share one circuit instead of taking two.")
print(
    f"  {len(as_sum(OBS).terms)} observable terms ->"
    f" {len(group_qubit_wise_commuting(OBS))} measurement setting"
    f"  ({device.circuits_run:,} circuits instead of {device.circuits_run * 2:,})"
)

for seconds in (0.5, 2.0):
    hours = device.circuits_run * seconds / 3600
    print(f"  at {seconds:.1f} s per circuit in a queue: {hours:.1f} hours for this 8-step run")

# --------------------------------------------------------------------------- #
header("5. What is honestly still missing")
# --------------------------------------------------------------------------- #
print("""  Already here
    - parameter-shift and Hadamard-test gradients: exact, measurement-only
    - shots everywhere, with standard errors
    - qubit-wise-commuting grouping, so a k-term observable is not k circuits
    - circuits built only from ordinary gates -- even amplitude encoding, which is
      uniformly-controlled rotations rather than a backend state-prep primitive
    - `register_backend`, and a base class that derives every measurement semantic
      from `counts` alone, as this file demonstrates

  Genuinely missing, in the order it would bite
    1. Batched submission. This run made one blocking call per circuit. Real
       providers take a *list* and return a job; one-at-a-time against a queue turns
       the run above from minutes into weeks. `expectation_batch` exists but no
       backend exploits it, and Estimator-style primitives would need a new method
       on the Backend protocol.
    2. Transpilation and routing. Devices have limited connectivity. A `chain`
       entangler maps onto a line; `ring` and `full` need SWAPs that nothing here
       inserts. `to_qiskit()` hands the circuit to Qiskit's transpiler, which is the
       intended escape hatch, but it is a hand-off rather than a feature.
    3. Error mitigation. Readout mitigation, zero-noise extrapolation, dynamical
       decoupling. At 2% readout error the run above still learned; at realistic
       two-qubit gate error it would not.
    4. Asynchronous jobs. Submit, poll, retrieve, resume. The API is synchronous
       throughout, which is fine for a simulator and wrong for a queue.

  None of that is blocked by the design -- items 1 and 4 are the ones that would
  actually change the Backend protocol, and they are why 0.x says simulator-only
  rather than pretending otherwise.""")
