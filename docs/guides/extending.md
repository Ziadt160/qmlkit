# Extending qmlkit

Almost every extension point is a registry. Register something and it becomes reachable
by name everywhere the library takes one — no subclassing, no plugin manifest, no
coordination with anything else, and no fork.

| I want to change | Use | And it becomes reachable from |
|---|---|---|
| The circuit shape | `register_ansatz` — or build an `Ansatz` inline | `get_ansatz`, `qk.search`'s `ansatz=` axis, `compare_ansatze` |
| A gate the library lacks | `register_gate` | every circuit, every backend, every gradient method |
| The two-qubit block inside a QCNN | `register_conv_filter` | `conv_block`, `qcnn_ansatz`, `mps_ansatz`, `tree_tensor_network` |
| How data becomes angles | `register_feature_map` | `qk.search`'s `feature_map=` axis |
| How gradients are estimated | `register_gradient` | `method=` anywhere, including `QuantumLayer` |
| Where circuits run | `register_backend` | `backend=`, `QMLKIT_BACKEND`, `backend_report()` |
| What the classical bar is | `register_baseline` | `qk.baseline`'s comparison table |
| Reading circuits in from elsewhere | `register_importer` | `get_importer`, `list_importers` |

Two things are extended **without** a registry, and that is deliberate — see
[a new optimiser](#a-new-optimiser) and [a new algorithm](#a-new-algorithm).

## A new ansatz

```python
import qmlkit as qk

@qk.register_ansatz("my_ladder")
def my_ladder(n_qubits, n_layers=2):
    block = qk.RotationLayer(("ry", "rz")) + qk.EntanglerLayer("cx", "chain")
    return qk.Ansatz(n_qubits, qk.repeat(n_layers, block), "my_ladder")

ansatz = qk.get_ansatz("my_ladder", n_qubits=3, n_layers=2)
print(ansatz)
print(qk.draw(ansatz.build()))
```

It now has correct gradients, resource counting, drawing, an `AnsatzReport`, and a
`QuantumLayer` — none of which you wrote. The parameter count is inferred from a dry
build, so there is nothing to miscount.

## A new gate

A gate needs a matrix. Declare its **generator frequencies** and parameter-shift
works on it; add a **derivative matrix** and adjoint differentiation works too.

```python
import numpy as np
import qmlkit as qk

def _sqrt_x(_=None):
    return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=complex)

qk.register_gate(qk.GateDef("sx", n_qubits=1, n_params=0, matrix=_sqrt_x))

qc = qk.QCircuit(1)
qc.apply("sx", 0)
print(np.round(qk.statevector(qc.to_spec()), 4))
```

For a *parameterised* gate, the two optional fields are what unlock differentiation:

```python
import numpy as np
import qmlkit as qk

def _rzz(theta):
    return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta),
                    np.exp(0.5j * theta), np.exp(-0.5j * theta)])

def _d_rzz(theta):
    return np.diag([-0.5j * np.exp(-0.5j * theta), 0.5j * np.exp(0.5j * theta),
                    0.5j * np.exp(0.5j * theta), -0.5j * np.exp(-0.5j * theta)])

qk.register_gate(qk.GateDef(
    "rzz", n_qubits=2, n_params=1,
    matrix=_rzz,
    frequencies=(1.0,),   # -> a correct 2-term shift rule, derived not transcribed
    dmatrix=_d_rzz,       # -> adjoint differentiation
))

qc = qk.QCircuit(2)
qc.h(0).apply("rzz", (0, 1), qk.ParamRef(0))
spec, theta = qc.to_spec(), np.array([0.7])

shift = qk.grad(spec, theta, qk.X(0), method="parameter-shift")
adj = qk.grad(spec, theta, qk.X(0), method="adjoint")
print(f"parameter-shift {shift[0]:+.10f}")
print(f"adjoint         {adj[0]:+.10f}")
print(f"agree to        {abs(shift[0] - adj[0]):.2e}")
```

Getting the same number from two independent routes is the check worth making on any
gate you add — see [The parameter-shift rule](parameter-shift.md) for why the
frequencies matter.

!!! warning "Gate registration is global"
    The registry is process-wide, so a gate registered in a test is visible to every
    later test. If you register throwaway gates, snapshot the registry rather than
    reading it live — the parity suite learned this the hard way.

### How far does your gate travel?

The obvious worry: Qiskit and Cirq have their own gate tables, and your gate is not in
either. So what happens when you ask for `backend="qiskit"`?

It works. Neither SDK has heard of your gate, so it is emitted **as its matrix** —
`UnitaryGate` on Qiskit, `MatrixGate` on Cirq — built by calling the `matrix=` you
registered. Nothing else about your circuit changes, and the result agrees with the
NumPy reference to machine precision.

```python
# docs: skip
qk.register_gate(qk.GateDef("xy", n_qubits=2, n_params=1, matrix=xy, frequencies=(1.0,)))

qk.statevector(spec, backend="qiskit")   # UnitaryGate, exact
qk.statevector(spec, backend="cirq")     # MatrixGate,  exact
qk.grad(spec, theta, qk.Z(1), method="parameter-shift", backend="qiskit")
```

The subtlety this hides is qubit order, and it is the kind that does not raise. qmlkit
is big-endian and Qiskit is little-endian, so `to_qiskit` already maps qubit `i` to
`n-1-i` — but a raw matrix carries its qubit order in its *basis* rather than in its
wire list, so the basis has to be reversed as well or a two-qubit gate on `(a, b)`
quietly acts as though it were on `(b, a)`. Cirq needs no reversal at all, being
big-endian like qmlkit. Rather than reason about that, the cross-backend suite asserts
it against the NumPy reference over ascending, descending and non-adjacent wire
orders, and over a three-qubit custom gate.

| Backend | A registered gate |
|---|---|
| `numpy` · `qiskit` · `cirq` | **works**, to machine precision |
| `cirq-density` · `qiskit-aer` | **works** — they inherit the same translation |
| `spinqit` | refuses. Its builder takes named gates, not an arbitrary matrix |
| `torch` | refuses for `backprop`. It differentiates *through* the gate, which needs a torch-native form; your `matrix=` is NumPy and no gradient flows through it. `parameter-shift` works, because a shift rule never inspects a state |

Both refusals say that, and what to do instead, rather than telling you to edit the
library.

## A new gradient estimator

```python
import numpy as np
import qmlkit as qk

@qk.register_gradient("forward_diff")
def forward_diff(spec, theta, obs, *, backend=None, shots=None, eps=1e-6, **kwargs):
    base = qk.expval(spec, obs, theta=theta, backend=backend, shots=shots)
    out = np.zeros(spec.n_params)
    for k in range(spec.n_params):
        step = np.zeros_like(theta)
        step[k] = eps
        out[k] = (qk.expval(spec, obs, theta=theta + step, backend=backend, shots=shots) - base) / eps
    return out

ansatz = qk.hardware_efficient(2, 1)
spec, theta = ansatz.build(), ansatz.init(seed=0)
mine = qk.grad(spec, theta, qk.Z(0), method="forward_diff")
exact = qk.grad(spec, theta, qk.Z(0), method="adjoint")
print(f"max deviation from adjoint: {np.abs(mine - exact).max():.2e}")
```

The signature is fixed: `(spec, theta, obs, *, backend, shots, **kwargs)`, returning
an array of length `spec.n_params`. Anything else you need arrives through `kwargs`,
and callers pass it straight through `qk.grad(..., your_kwarg=...)`.

## A new backend

Subclass `Backend` and implement **one** method — `statevector`. The base class
supplies the measurement *semantics*: sampling, basis rotation, expectation values,
seeded counts. That is deliberate: if every backend re-implemented those, agreement
between them would be a coincidence rather than a property.

```python
import numpy as np
import qmlkit as qk
from qmlkit.core.backends.base import Backend

class MirrorBackend(Backend):
    """The NumPy reference, but proving the extension point works."""

    name = "mirror"
    supports_statevector = True
    supports_exact = True

    def statevector(self, spec):
        self._check_bound(spec)
        return qk.get_backend("numpy").statevector(spec)

qk.register_backend("mirror", MirrorBackend)

qc = qk.QCircuit(2)
qc.h(0).cx(0, 1)
spec = qc.to_spec()
print(np.round(qk.statevector(spec, backend="mirror"), 4))
print("agrees with numpy:", np.allclose(
    qk.statevector(spec, backend="mirror"), qk.statevector(spec, backend="numpy")))
```

If you add a real backend, the thing to run is `tests/test_cross_backend.py` — it
parametrises over every installed backend automatically, so yours is covered the
moment it is registered.

### A device, which cannot hand you a state

A real QPU has no statevector and no shot-free expectation. It can run a circuit and
report bitstrings, and that is *also* one method:

```python
import numpy as np
import qmlkit as qk
from qmlkit.core.backends.base import Backend

class Device(Backend):
    """Everything a QPU is, and nothing it is not."""

    name = "example_device"
    supports_statevector = False        # no amplitudes
    supports_exact = False              # no shot-free expectation

    def counts(self, spec, shots, seed=None):
        self._check_bound(spec)
        # a real provider would submit the circuit here
        probabilities = np.abs(qk.get_backend("numpy").statevector(spec)) ** 2
        rng = np.random.default_rng(0 if seed is None else seed)
        drawn = rng.multinomial(shots, probabilities / probabilities.sum())
        return {format(i, f"0{spec.n_qubits}b"): int(n) for i, n in enumerate(drawn) if n}

device = Device()
ansatz = qk.hardware_efficient(3, 2)
spec, theta = ansatz.build(), ansatz.init(seed=0)
observable = qk.Z(0) + 0.5 * qk.ZZ(0, 2)
print(f"sampled expectation {qk.expectation(spec, observable, theta, shots=4096, backend=device):+.3f}")
```

### What that one method gets you

Everything above it, derived once in the base class:

```python
thetas = np.random.default_rng(0).uniform(-np.pi, np.pi, (5, ansatz.n_params))

values = qk.expectation_over(spec, thetas, observable, shots=4096, backend=device)
print(f"batched expectations {values.shape}")

gradients = qk.grad_batch(spec, thetas, observable,
                          method="parameter-shift", shots=4096, backend=device)
print(f"batched gradients    {gradients.shape}")

kernel = qk.QuantumKernel(qk.AngleFeatureMap(3), shots=4096, backend=device, seed=0)
print(f"Gram matrix          {kernel(np.random.default_rng(1).uniform(0, np.pi, (4, 3))).shape}")
```

Qubit-wise-commuting grouping comes with it, so a four-term observable diagonal in `Z`
costs one circuit rather than four — on a device, where circuit count is the binding
constraint, that is the difference between a feasible run and an infeasible one.

`param_shift_grad_batch` matters most here. A shift rule only ever needs the circuit
*run* at shifted angles, so a whole batch's gradient is one set of evaluations with no
state inspection anywhere — which on hardware is a single job submission instead of
`batch x 2P` blocking calls.

### And what it refuses

```python
for method in ("adjoint", "backprop"):
    try:
        qk.grad(spec, theta, observable, method=method, backend=device)
    except ValueError as error:
        print(f"{method}: {str(error)[:70]}...")
```

Both need the statevector, so both refuse and name `parameter-shift` instead. So does
exact mode:

```python
try:
    qk.expectation(spec, observable, theta, backend=device)
except ValueError as error:
    print(error)
```

That is the half of "backend-agnostic" that matters. Anything can *run* everywhere; the
useful property is that what cannot work is refused by name rather than silently
computed on a simulator and handed back looking perfect. `backprop` was doing exactly
that until it was caught by writing this section.

### What is still missing for real hardware

The 0.x line ships no device backend, and the protocol supporting one is a different
claim from having one. `examples/toward_hardware.py` runs a mock QPU end to end and
states the four gaps in the order they would bite: **batched submission** (now largely
in place — `expectation_over_slots` is the call a provider would turn into a job),
**transpilation and routing**, **error mitigation**, and **asynchronous jobs**. The
last is the one that would still change the `Backend` protocol.

## A new optimiser

There is no `register_optimizer`, because there is nothing to look up — `OPTIMIZERS` is
an open dict and an optimiser is just a function:

```python
# docs: skip
from qmlkit.algorithms.vqe import OPTIMIZERS

def my_optimiser(loss, theta0, *, grad, n_steps=200, lr=0.1):
    theta = np.asarray(theta0, dtype=float).copy()
    history = [float(loss(theta))]
    for _ in range(n_steps):
        theta = theta - lr * grad(theta)      # qk.grad, exact, not a difference
        history.append(float(loss(theta)))
    return theta, history                     # the contract: (theta, history)

OPTIMIZERS["mine"] = my_optimiser
VQE(hamiltonian, n_qubits=3, optimizer="mine").run(n_steps=50, lr=0.05)
```

The contract is `fn(loss, theta0, **kw) -> (theta, history)`. `VQE`, `QAOA` and
`AdaptVQE` all take `optimizer=` as either a name from that dict **or the function
itself**, so you do not have to register anything to use one.

!!! warning "If your optimiser needs the gradient, the caller has to inject it"
    Each solver decides which optimisers get `grad=` passed in. Naming only some of
    them there is how `optimizer="adam"` came to raise `TypeError: _adam() missing 1
    required keyword-only argument: 'grad'` from all three algorithms that advertised
    it — a documented option that had never run. `tests/test_optimizer_wiring.py` now
    parametrises over `OPTIMIZERS` itself, so a new entry cannot be added without
    being covered.

Each optimiser also spells its own iteration count (`n_sweeps`, `n_iterations`,
`n_steps`), because `run(**optimizer_kwargs)` passes them straight through.

## A new algorithm

There is no `register_algorithm` either, and this one is worth explaining because it
looks like an omission.

A registry exists so the library can look something up **on your behalf** — a gate name
inside a circuit, a `method=` string, a `backend=` string. An algorithm is the outermost
layer. Nothing inside qmlkit needs to find your VQE by name; you call it. So an
algorithm is not a plugin, it is a loop:

> Each one is a *loop* over machinery that already exists — ansatz, gradients,
> optimisers, observables — so each is thin, and every structural choice it makes is an
> argument rather than something baked in.
>
> — `qmlkit/algorithms/__init__.py`

`VQE` is 200 lines and the shape is worth copying:

```python
# docs: skip
@dataclass(frozen=True)
class MyResult:
    value: float
    theta: np.ndarray
    history: list[float]
    exact: float | None = None          # what it was checked against

class MyAlgorithm:
    def __init__(self, hamiltonian, n_qubits, *, ansatz=None,
                 optimizer="rotosolve", backend=None, shots=None):
        self.ansatz = ansatz or qk.hardware_efficient(n_qubits, 2)   # default, not hardcode

    def cost(self, theta): ...              # one qk.expectation call
    def gradient_of_cost(self, theta): ...  # one qk.grad call
    def run(self, seed=None, compare_exact=None, **optimizer_kwargs) -> MyResult: ...
```

Four things make it belong here rather than merely work:

**Take the structure as an argument, and actually use it.** `tests/test_injection.py`
injects two ansätze of *different parameter counts* and asserts the model's count
follows — because a constructor that accepts `ansatz=` and quietly ignores it looks
identical from the outside.

**Check yourself against something exact while you still can.** `VQE`, `QAOA` and
`AdaptVQE` diagonalise densely when it is affordable (12 qubits or fewer by default)
and report `error_vs_exact`. A variational algorithm converging confidently on the
wrong answer is the ordinary failure here, not the exotic one.

**Refuse, or warn, when a component's assumptions do not hold.** `QAOA` consults
`supports_rotosolve(spec)` and warns when the optimiser is invalid for the circuit it
just built: Rotosolve assumes each angle drives a single sinusoid, and QAOA's cost angle
drives one `rz` per edge — five frequencies on a five-edge problem, measured. It
converges immediately, on the wrong point, and reports it as a result.

**Use the exact gradient.** `qk.grad` already gives you one on any circuit and
observable. Finite differences are for debugging and tests, and say so in their own
docstring.

### Where to put it

If it is your own research, **it does not need to be in the library at all** — import
qmlkit, write your loop, keep it in your project. That is what the three layers are
for, and nothing is gained by vendoring it in.

If it is genuinely general, it goes in `src/qmlkit/algorithms/`, exported from that
package's `__init__`, with a reference page entry and a changelog entry. And register
the *pieces* it introduces — a gate, an ansatz shape, a filter, an estimator — because
those are the parts other people's code will want to reach by name.
