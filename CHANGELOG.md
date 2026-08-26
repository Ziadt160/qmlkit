# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — Phase 0: foundation

- **Backend-neutral circuit IR** (`CircuitSpec`, `Op`, `ParamRef`, `Slot`). A circuit
  is data; backends compile it, and gradients, resource counting and drawing all read
  the same representation.
- **Slot abstraction** — one angle site per (operation, parameter position). A logical
  parameter may fill several slots (weight tying), which is what makes correct
  per-occurrence gradients expressible.
- **Exact NumPy statevector backend**, double-precision throughout, plus the `Backend`
  protocol and a backend registry with a lazy, clearly-diagnosed SpinQit entry.
- **Pauli observables** — `PauliString`, `PauliSum`, `Z`/`X`/`Y`/`ZZ` shorthands, exact
  and sampled expectations, automatic basis rotation, and qubit-wise-commuting grouping.
- **Execution API** — `statevector`, `run_counts`, `probabilities`, `expectation`,
  `expectation_batch`. `shots=None` is the default and returns exact values.
- **Parameter-shift gradients**, with shift rules *derived* from each gate's declared
  generator frequencies rather than transcribed:
  - per-gate rule lookup, so a circuit mixing `ry` (two-term) with `crz` (four-term)
    is differentiated correctly;
  - per-occurrence shifting, so weight-tied parameters sum over their occurrences;
  - `grad_circuit_cost`, which sums real per-slot rule costs instead of assuming `2P`;
  - gradients with respect to encoded inputs, via `angle_encode(..., trainable=True)`.
- **Fluent `QCircuit` builder** with rotation layers, named entanglement patterns
  (chain, ring, full, alternating) and parametric entanglers.
- **Shot arithmetic** — standard error, variance, `shots_for_precision`, runtime estimates.

### Notes

- Simulator-only for the whole `0.x` line. The NumPy backend is the default and the
  reference implementation.
- Requires Python 3.10+. Python 3.9 reached end-of-life in October 2025, and 3.10 is
  also SpinQit's highest supported version, so it is the overlap that matters.

### Added — Phase 1: multi-backend support

- **SpinQit backend**, verified against a live SpinQit 0.2.x install on Python 3.10.
  Supports both exact statevectors (`result.states`) and sampled counts, plus
  `verify_conventions()` — a one-call self-check that bit order and gate definitions
  still match what the backend assumes.
- **Qiskit backend** and **Cirq backend**, both agreeing with the NumPy reference to
  machine precision. Each exposes its native circuit (`to_qiskit`, `to_cirq`,
  `to_spinqit`) so it can be drawn, transpiled, or handed to that SDK's own tooling.
- **Cross-backend equivalence suite** — one circuit zoo run through every installed
  backend, asserting agreement on statevectors, probabilities, expectations over
  X/Y/Z and two-body terms, seeded sampling, and parameter-shift gradients.
- **Backend registry with availability detection** — `available_backends()`,
  `is_available()`, `backend_report()`. Every SDK import is lazy, so `import qmlkit`
  requires none of them, and a missing SDK produces an install command rather than an
  `ImportError`. `QMLKIT_BACKEND` sets the default from the environment.
- **Shared seeded sampling** (`_sampling.py`), so a seed reproduces identical counts on
  any simulator backend. Qiskit's `sample_counts` takes no seed and Cirq carries its
  own RNG, which would otherwise make results irreproducible and incomparable.
- `sdg` / `tdg` added to the circuit builder.

### Changed

- `Backend` now supplies measurement *semantics* (sampling, basis rotation,
  expectation) while subclasses supply only a statevector. Previously each backend
  would have re-implemented these; one definition is what makes cross-backend
  equivalence meaningful rather than coincidental.

### Fixed — upstream discrepancies worked around

- **SpinQit's `CY` applies `-iY`** to the control-1 subspace instead of `Y`. That is a
  relative phase between control branches, so it is physically observable — a control
  qubit in superposition gives different measurement statistics. The backend emits
  `Sd·CX·S` instead, which reproduces the standard gate exactly. SpinQit's
  single-qubit `Y` is correct; only the controlled form is affected.
- SpinQit's simulator carries a precision floor near `1e-10`, not machine precision;
  cross-backend comparisons use a per-backend tolerance so this is not mistaken for a
  translation error.

### Added — Phase 2: the encoding layer

- **`PauliFeatureMap`**, with `ZFeatureMap` and `ZZFeatureMap` on top of it — and the
  two helpers `Lecture3` cell 42 references but never defines, `basis_change` (`_basis`)
  and `default_data_map` (`_phi`). That cell cannot run as written; this one is tested
  against the analytic kernels each map induces.
- **`amplitude_encode`**, built from uniformly-controlled rotations rather than a
  backend state-preparation primitive. Emits only `ry`/`rz`/`cx`, so it runs identically
  on every backend and its exponential cost is visible. Handles real, signed and complex
  amplitudes, exact to machine precision; `check=True` re-simulates and asserts the
  result. Exposes `uniformly_controlled_rotation` and `state_preparation_angles`.
- **`hamiltonian_encode`** — Trotterised Ising evolution. Every term is diagonal so the
  terms commute and the split is *exact at any step count*: `steps` buys depth and
  nothing else, which is pinned by a test.
- **`DataReuploadEncoder`** — interleaved `S(x)` / `W(theta)` blocks, with
  `trainable_input=True` to expose `df/dx`. A test confirms by FFT that more uploads
  reach genuinely higher frequencies.
- **`AngleScaler`** and **`PCAReducer`** (plain SVD, no sklearn) plus `reduce_to_qubits`,
  for the two problems every model hits before any quantum step: wrong feature scale, and
  more features than qubits.
- `AngleFeatureMap`, `FeatureMap` base with a free `adjoint()`, and `pauli_terms`.

### Added — Phases 3 & 4: ansätze, gradient methods, and the PyTorch bridge

**A composable ansatz vocabulary.** `RotationLayer`, `EntanglerLayer`,
`ParametricEntangler`, `PoolLayer`, `Custom`, composed with `+`, `repeat` and `share`.
Every built-in is a short expression in it, so a new ansatz is one line and inherits
correct gradients, resource counting and a torch layer without opting into any of them.
Parameter counts are **inferred** from a dry build — a miscount is not a failure mode.

- Zoo: `hardware_efficient`, `strongly_entangling`, `simplified_two_design`,
  `tree_tensor_network`, `mps_ansatz`, `qcnn_ansatz`, `qaoa_ansatz`, plus `conv_block`.
- `share()` and `conv_block(tied=True)` implement genuine weight tying. An 8-qubit QCNN
  has 6 tied parameters against 22 free ones at identical gradient cost — the real
  convolutional tradeoff, and the case the per-occurrence gradient sum exists for.
- `register_ansatz` / `get_ansatz` / `list_ansatze`.

**Adjoint differentiation** — exact gradients in one backward pass, independent of the
parameter count. Agrees with parameter-shift to machine precision (1e-16) on every
ansatz in the zoo, including weight-tied and four-term-rule circuits. Measured 11×
faster at `P=20` and 64× at `P=120`. Requires closed-form gate derivatives, which are
now declared on every parameterised gate and verified against finite differences.

**SPSA** — two evaluations per gradient whatever `P` is, with Spall's decay schedules
including the stability constant `A` the lecture version omits. `minimize_spsa` for the
optimisation loop.

**One `grad()` with a method registry.** `method="auto"` picks adjoint when every gate
has a derivative and the backend can produce a statevector, parameter-shift otherwise;
shots force parameter-shift. `register_gradient` makes a custom estimator a keyword
everywhere the library takes `method=`.

**The PyTorch bridge.**

- `QuantumFunction` / `QuantumLayer` — a circuit as an `nn.Module`. **Inputs receive
  gradients**, so a classical layer placed before the quantum one actually trains; this
  is the Lecture 6 dressed-circuit defect, now fixed and asserted by test. `df/dx`
  through a *nonlinear* feature map works by differentiating the circuit with respect
  to its encoding angles and finishing the chain rule classically — no circuits spent
  on the classical half.
- `VQC`, `VQRegressor`, `HybridModel` — the two-line path, with `fit`/`predict`/`score`.
  Every default is one keyword away from being something else.
- `torch.autograd.gradcheck` passes for inputs and weights, on linear and nonlinear maps.

**Feature maps** gained `angles`, `n_angles`, `build_parametric` and `angle_jacobian`
(closed-form for the standard data map), which is what makes input gradients possible.

### Changed

- `expval()` added as a float-only convenience alongside `expectation()`.
- PyTorch is an optional extra; `qmlkit.nn` is imported lazily so `import qmlkit` never
  requires it.

### Added — Phase 5 and parity with Qiskit ML / PennyLane

**Quantum kernels (Phase 5).**

- Three estimators: `fidelity_kernel` (compute-uncompute, the default — fewest qubits,
  no ancilla), `swap_test_kernel` (two registers plus an ancilla), and `hadamard_test`,
  the only one that keeps the **sign** of the inner product. All three match the exact
  overlap to machine precision.
- `QuantumKernel` with a **symmetric cache**, so `k(a,b)` and `k(b,a)` share one entry;
  a training Gram matrix costs exactly `m(m-1)/2` circuits.
- PSD repair — `threshold_matrix`, `displace_matrix`, `flip_matrix`, `closest_psd_matrix`.
  Shot noise really does push an estimated Gram matrix out of the cone; a test asserts it
  and that each method brings it back.
- `target_alignment` (kernel-target alignment), `center_kernel`, `normalize_kernel`.
- `QSVC` / `QSVR` over scikit-learn's precomputed-kernel solver, `NearestFidelityClassifier`
  (no solver at all), and `TrainableKernel`, which optimises the *embedding* by maximising
  alignment before any classifier is fitted.
- `projected_kernel_matrix` — compares one-qubit reduced density matrices instead of the
  global fidelity, so it stays informative where the fidelity kernel has concentrated.
  Measured at 8 qubits: fidelity spread 0.005 against projected 0.037.
- `concentration_report`, `geometric_difference`, `kernel_shot_cost`.

**Filling the gaps against Qiskit Machine Learning and PennyLane.**

- `qmlkit.metrics` — `expressibility` (KL from the Haar fidelity distribution),
  `meyer_wallach`, `entangling_capability`, `gradient_variance`, `barren_plateau_scan`,
  `effective_dimension` (the normalised-Fisher construction, not the eigenvalue-count
  shortcut), `generalization_bound`, and `AnsatzReport` / `compare_ansatze`.
- `qmlkit.fourier` — `fourier_coefficients`, `spectrum`, `model_spectrum`. Turns the
  central claim of the re-uploading literature into a measurement.
- `qmlkit.info` — `reduced_dm`, `purity`, `vn_entropy`, `mutual_info`, `state_fidelity`,
  `concurrence`, `bloch_vector`.
- `qmlkit.optim` — **Rotosolve** (closed-form per-coordinate minimum, no learning rate),
  the **Fubini-Study metric tensor**, `quantum_fisher_information` (= 4x the metric), and
  **quantum natural gradient**.
- Three more ansatz templates: `basic_entangler`, `two_local`, `random_layers` — ten total.
- `qmlkit.datasets` — `ad_hoc_data` (separable by a quantum kernel by construction),
  `bars_and_stripes`, `make_moons`, `make_circles`, `make_blobs`, `make_parity`,
  `train_test_split`. No downloads, no sklearn.
- `qmlkit.draw` / `qmlkit.specs` — text circuit diagrams and a full cost summary.

### Fixed

- **A commuting trainable block silently collapses data re-uploading.** `Ry(x)Ry(t1)Ry(x)Ry(t2)`
  equals `Ry(2x + t1 + t2)`, so the model reaches a single frequency and every weight
  becomes a phase shift — measured: one frequency at amplitude 1.0, against the full
  `0..L` spectrum with a non-commuting block. `DataReuploadEncoder` now warns.
- `np.trapezoid` in a test made the suite NumPy-2-only; SpinQit pins `numpy<2`, so the
  suite has to run on both. Replaced, and the whole codebase scanned for other NumPy-2-only
  APIs (there were none).

### Changed — re-uploading generalized

**Data re-uploading is a pattern, not a structure**, and the library now treats it as
one. `EncodingLayer` makes a feature map a composable block, so re-uploading is any
interleaving of any encoding with any trainable block:

- `reupload(fmap, n_layers, block=..., order="SW"|"WS", share_weights=...)` builds the
  common shapes; anything else composes directly from the block vocabulary — including
  **two different feature maps in one model**, which the previous fixed class could not
  express at all.
- `BuildContext` gained an input namespace, so encoding angles and trainable weights
  occupy separate ranges of one flat vector. That is what keeps `df/dx` and `df/dtheta`
  separable while a re-uploading model drops straight into `QuantumLayer`.
- `Ansatz.bind(x, weights)` binds data and weights separately; `build(theta)` still
  takes the full vector and now says so when the sizes disagree.
- `DataReuploadEncoder` remains as the plain angle-encoding shortcut, documented as one
  convenient choice rather than the definition.

### Added — Phase 6

- `QCNNLayer`, `MPSLayer`, `QLSTMCell`, `QLSTM`, `DressedQuantumNet` — structured
  architectures as ordinary `nn.Module`s.
- `qmlkit.generative` — `QCBM` (trained by MMD, because a Born machine is implicit and
  admits no likelihood), `QGAN`, `QuantumBoltzmannMachine`, `QuantumHopfield`, plus
  `mmd_squared`, `kl_divergence`, `total_variation`, `boltzmann`, `ising_energy`.

### Fixed

- `test_reupload_widens_the_reachable_spectrum` used a trainable block that commutes
  with its encoding, so it measured the *collapsed* single-frequency case and passed
  for the wrong reason. Now uses a non-commuting block, which is what the claim needs.

### Fixed — a flaky test, and its cause

`test_qcbm_training_reduces_mmd` failed roughly one run in five. The cause was real,
not a tolerance problem: `QCBM.score()` samples without a seed, so it draws on the
backend's shared RNG, whose state depends on whatever ran before. Comparing two noisy
estimates before and after training can go the wrong way by chance even when training
worked.

- `QCBM.score()` gained a `seed` argument for reproducibility.
- `QCBM.exact_distance()` added — measures the distance to the target on the model's
  **exact** distribution, no sampling at all, which is what a before/after comparison
  needs on a simulator.
- The test now asserts on the exact distance. Fifteen consecutive full runs across both
  environments are clean.

### Added — two more gradient algorithms, and cross-validation against PennyLane

**`method="hadamard"`** — the Hadamard-test gradient. For a Pauli-generated rotation,
`d_k E = -Im<phi|O|psi>`, and that imaginary part is exactly what a Hadamard test reads
out: put an ancilla in `|+>`, insert a *controlled* generator right after gate `k`, and
measure `<Y_a (x) O>`. One circuit per parameter instead of parameter-shift's two, and
unlike adjoint it is a real measurement, so it is valid on hardware. It refuses
controlled rotations with an explanation rather than guessing — `CRZ`'s generator is
not a Pauli, so there is no controlled form to insert.

**`method="backprop"`** — a differentiable statevector simulator written in torch
(`TorchBackend`, `torch_expectation`), so autograd differentiates the circuit directly.
This is the least physical method in the library and is documented as such: it reads
intermediate states no device will expose, and its memory grows with depth. It exists
because a circuit inside an autograd graph is genuinely useful; for a standalone
gradient, adjoint is both faster and lighter.

- `hessian()` — second derivatives by differencing the *exact* gradient, so only the
  outer derivative is approximated. Returned symmetrised.
- `gradient_cost(spec, method)` — circuits one gradient costs under any method, so the
  tradeoff is queryable instead of folklore.

All four exact methods (adjoint, backprop, hadamard, parameter-shift) agree to machine
precision across the ansatz zoo, including weight-tied circuits and parameter scaling.
Measured on 5 qubits with a two-term observable, `P=120`: adjoint 12.6 ms, backprop
50 ms, hadamard 404 ms, parameter-shift 823 ms.

**Cross-validated against PennyLane.** `examples/compare_pennylane.py` checks 19
quantities — expectations, every gradient method in both libraries, feature-map
kernels, reduced density matrices and entropies, and the Fourier spectrum of a
re-uploading model — and agrees to `1e-16` on all of them. `examples/quickstart.py`
walks every layer of the library end to end.

One genuine convention difference surfaced: PennyLane's `IQPEmbedding` emits `RZ(x_i)`
and `MultiRZ(x_i x_j)` where qmlkit's `PauliFeatureMap` follows the Qiskit convention
and emits `Rz(2 phi)`. Neither is wrong; they match once the data map absorbs the
factor of two. It is recorded because a kernel differing by exactly this factor would
be very hard to spot.

### Fixed

- `hadamard_grad` called `expval` once **per observable term**, re-preparing the state
  each time — so a `k`-term observable cost `k` circuits per parameter, not one, and
  the method's whole circuit-count advantage over parameter-shift disappeared into it.
  It now passes the lifted observable as a single `PauliSum`, which the backend
  accumulates from one state preparation. Wall-clock at `P=120` went from 749 ms to
  404 ms — against parameter-shift's 823 ms, which is the 2x the circuit count
  predicted all along. Found by benchmarking rather than by a test, because every test
  still passed: the answer was always correct, only the cost was wrong.
- `hessian()` annotated `Sequence[float]` without importing `Sequence`. Invisible at
  runtime under `from __future__ import annotations`, caught by mypy.
- `torch_expectation` reused the loop variable `p` for both gate parameters and Pauli
  letters, which mypy flagged as a genuine type collision.

### Added — a real cross-validation suite against PennyLane

`tests/test_pennylane_parity.py` — **301 parity cases**, run as part of the normal
suite rather than as a one-off script, so they guard every future change. Coverage:
all 20 gate matrices at 6 angles; every closed-form `dU/dtheta` against a differenced
PennyLane matrix; 40 **randomly generated** circuits over the full gate set on 1-5
qubits (statevectors, probabilities, random multi-term observables); gradients across
5 ansatze x 4 observables, all four exact methods, PennyLane's own four methods back
against ours, and randomised circuits; angle/amplitude/basis/IQP encodings;
`BasicEntanglerLayers` and `StronglyEntanglingLayers`; kernel Gram matrices; reduced
density matrices, entropies, purity, mutual information and fidelity over random
states; re-uploading Fourier spectra; the Fubini-Study metric and QFIM; and Rotosolve
and QNG trajectories compared step by step rather than only at their endpoints.

`examples/benchmark_pennylane.py` — wall-clock on identical work. qmlkit is faster on
14/14 cases, median 6.1x, with the caveats stated in the file: sections 1-4 are
largely dispatch overhead and narrow as `2^n` grows, while the metric-tensor result is
an algorithmic difference that widens.

**Four genuine convention differences surfaced, each now pinned by its own test.**
None is a bug in either library, and all four are the kind that produce a plausible
wrong number rather than an exception:

- PennyLane's `IQPEmbedding` emits `RZ(x_i)` / `MultiRZ(x_i x_j)`; qmlkit follows the
  Qiskit convention and emits `Rz(2 phi)`.
- `amplitude_encode` drops one overall phase, since it is built from uniformly
  controlled rotations rather than a state-prep primitive. Unobservable in isolation,
  observable inside a controlled block — which the docstring already warned about.
- A `"ring"` on two qubits collapses to a single `CX` here; PennyLane's templates emit
  both `CNOT(0,1)` and `CNOT(1,0)`. `entangler_pairs` now documents this.
- `approx="block-diag"` means something narrower in PennyLane: it blocks the metric by
  circuit layer and zeroes cross-layer entries. The practical consequence is measured
  in `test_qng_beats_pennylanes_default_because_its_metric_is_exact` — on a 3-qubit,
  2-layer problem at equal steps and step size, qmlkit's QNG reaches `-2.9999999`
  where PennyLane's default stalls at `-2.22`. Pointed at the exact metric, PennyLane
  traces qmlkit's trajectory to `1e-8`.

### Changed — the metric tensor is now exact

`metric_tensor` differenced the *circuit* to get each `|d psi / d theta_k>`, at
`eps=1e-4`. Every other exact quantity in the library is closed-form, and this one was
not: it agreed with PennyLane only to `2e-10`, and it was the input to QNG.

It now takes one forward sweep, carrying each open derivative state through the next
gate and opening a new one at each parameterised gate from that gate's declared
`dU/dtheta`. Same `(P, 2^n)` memory as before, **fewer** simulations than `2P`
differenced circuits, no step size, and agreement with PennyLane improves from `2e-10`
to `1.7e-16`. Finite differences remain as the fallback for a custom gate registered
without a `dmatrix`, which is the only case that still consults `eps`.

### Fixed

- `qmlkit` is measurably more accurate than PennyLane on one quantity, now recorded so
  a future tolerance change is deliberate: `state_fidelity` hits the analytic
  `|<a|b>|^2` to `1e-16`, whereas `qml.math.fidelity` takes matrix square roots of
  rank-1 density matrices and loses about eight digits.
- The parity fuzzer drew gate names from the live registry, which other test modules
  write throwaway gates into at run time — so it passed alone and failed in a full
  run. It now draws from a snapshot taken at import, and a separate test asserts the
  PennyLane mapping covers every built-in gate, so adding a gate cannot silently
  escape this file's coverage.
