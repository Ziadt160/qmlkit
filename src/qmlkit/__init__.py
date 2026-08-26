"""qmlkit — a backend-agnostic quantum machine learning library.

The same circuit runs on SpinQit, Qiskit, Cirq or the built-in exact NumPy
reference. Simulator-only for the 0.x line; expectations are exact unless you
ask for shots:

    >>> import qmlkit as qk
    >>> import numpy as np
    >>> spec = qk.angle_encode([0.7])
    >>> round(qk.expectation(spec, qk.Z(0)), 12) == round(float(np.cos(0.7)), 12)
    True
"""

from __future__ import annotations

__version__ = "0.1.0"

from qmlkit import datasets, fourier, generative, info, kernels, metrics, optim
from qmlkit.ansatz import (
    Ansatz,
    Custom,
    EncodingLayer,
    EntanglerLayer,
    ParametricEntangler,
    PoolLayer,
    RotationLayer,
    basic_entangler,
    conv_block,
    get_ansatz,
    hardware_efficient,
    list_ansatze,
    mps_ansatz,
    qaoa_ansatz,
    qcnn_ansatz,
    random_layers,
    register_ansatz,
    repeat,
    reupload,
    share,
    simplified_two_design,
    strongly_entangling,
    tree_tensor_network,
    two_local,
)
from qmlkit.core.backends.base import Backend, BackendNotAvailable
from qmlkit.core.backends.numpy_backend import NumpyBackend
from qmlkit.core.backends.registry import (
    available_backends,
    backend_report,
    default_backend,
    get_backend,
    is_available,
    list_backends,
    register_backend,
    set_default_backend,
)
from qmlkit.core.builder import QCircuit, entangler_pairs
from qmlkit.core.execute import (
    expectation,
    expectation_batch,
    expval,
    probabilities,
    run_counts,
    statevector,
)
from qmlkit.core.gates import GateDef, get_gate, list_gates, register_gate
from qmlkit.core.ir import CircuitSpec, Op, ParamRef, Slot
from qmlkit.core.observables import ZZ, I, PauliString, PauliSum, X, Y, Z
from qmlkit.draw import draw, specs
from qmlkit.encoding import (
    AngleFeatureMap,
    AngleScaler,
    DataReuploadEncoder,
    FeatureMap,
    PauliFeatureMap,
    PCAReducer,
    ZFeatureMap,
    ZZFeatureMap,
    amplitude_encode,
    angle_encode,
    basis_encode,
    basis_index,
    hamiltonian_encode,
    n_qubits_for,
    reduce_to_qubits,
    to_angle_range,
)
from qmlkit.gradients import (
    SPSASchedule,
    adjoint_grad,
    choose_method,
    grad,
    gradient_cost,
    hadamard_grad,
    hessian,
    list_gradient_methods,
    minimize_spsa,
    register_gradient,
    spsa_grad,
    supports_adjoint,
)
from qmlkit.gradients.parameter_shift import (
    finite_diff_grad,
    grad_circuit_cost,
    param_shift_grad,
    param_shift_grad_circuit,
)
from qmlkit.gradients.rules import (
    ShiftRule,
    four_term_rule,
    general_shift_rule,
    rule_for_gate,
    two_term_rule,
)
from qmlkit.info import (
    bloch_vector,
    concurrence,
    mutual_info,
    purity,
    reduced_dm,
    state_fidelity,
    vn_entropy,
)
from qmlkit.kernels import (
    QSVC,
    QSVR,
    NearestFidelityClassifier,
    QuantumKernel,
    TrainableKernel,
    closest_psd_matrix,
    concentration_report,
    displace_matrix,
    fidelity_kernel,
    flip_matrix,
    geometric_difference,
    hadamard_test,
    is_psd,
    kernel_matrix,
    min_eigenvalue,
    projected_kernel_matrix,
    swap_test_kernel,
    target_alignment,
    threshold_matrix,
)
from qmlkit.metrics import (
    AnsatzReport,
    barren_plateau_scan,
    compare_ansatze,
    effective_dimension,
    entangling_capability,
    expressibility,
    generalization_bound,
    gradient_variance,
    meyer_wallach,
)
from qmlkit.optim import (
    metric_tensor,
    minimize_qng,
    minimize_rotosolve,
    quantum_fisher_information,
    rotosolve_step,
)
from qmlkit.utils.shots import (
    p0_from_z,
    shots_for_precision,
    standard_error,
    variance,
    z_from_p0,
)

__all__ = [
    "__version__",
    # core
    "CircuitSpec",
    "Op",
    "ParamRef",
    "Slot",
    "QCircuit",
    "entangler_pairs",
    # gates
    "GateDef",
    "get_gate",
    "list_gates",
    "register_gate",
    # observables
    "PauliString",
    "PauliSum",
    "I",
    "X",
    "Y",
    "Z",
    "ZZ",
    # backends
    "Backend",
    "BackendNotAvailable",
    "NumpyBackend",
    "get_backend",
    "default_backend",
    "set_default_backend",
    "register_backend",
    "list_backends",
    "available_backends",
    "is_available",
    "backend_report",
    # execution
    "statevector",
    "run_counts",
    "probabilities",
    "expectation",
    "expectation_batch",
    # encoding
    "angle_encode",
    "basis_encode",
    "basis_index",
    "n_qubits_for",
    "amplitude_encode",
    "hamiltonian_encode",
    "FeatureMap",
    "PauliFeatureMap",
    "ZFeatureMap",
    "ZZFeatureMap",
    "AngleFeatureMap",
    "DataReuploadEncoder",
    "to_angle_range",
    "AngleScaler",
    "reduce_to_qubits",
    "PCAReducer",
    # gradients
    "ShiftRule",
    "two_term_rule",
    "four_term_rule",
    "general_shift_rule",
    "rule_for_gate",
    "param_shift_grad",
    "param_shift_grad_circuit",
    "grad_circuit_cost",
    "finite_diff_grad",
    # submodules
    "kernels",
    "metrics",
    "optim",
    "fourier",
    "info",
    "datasets",
    "generative",
    # kernels
    "QuantumKernel",
    "fidelity_kernel",
    "swap_test_kernel",
    "hadamard_test",
    "kernel_matrix",
    "target_alignment",
    "is_psd",
    "closest_psd_matrix",
    "threshold_matrix",
    "displace_matrix",
    "flip_matrix",
    "min_eigenvalue",
    "projected_kernel_matrix",
    "concentration_report",
    "geometric_difference",
    "QSVC",
    "QSVR",
    "NearestFidelityClassifier",
    "TrainableKernel",
    # metrics
    "expressibility",
    "meyer_wallach",
    "entangling_capability",
    "gradient_variance",
    "barren_plateau_scan",
    "effective_dimension",
    "generalization_bound",
    "AnsatzReport",
    "compare_ansatze",
    # optimisers
    "minimize_rotosolve",
    "rotosolve_step",
    "metric_tensor",
    "quantum_fisher_information",
    "minimize_qng",
    # quantum information
    "reduced_dm",
    "purity",
    "vn_entropy",
    "mutual_info",
    "state_fidelity",
    "concurrence",
    "bloch_vector",
    # visualisation
    "draw",
    "specs",
    "expval",
    # ansatz
    "Ansatz",
    "register_ansatz",
    "get_ansatz",
    "list_ansatze",
    "RotationLayer",
    "EntanglerLayer",
    "ParametricEntangler",
    "PoolLayer",
    "Custom",
    "EncodingLayer",
    "reupload",
    "repeat",
    "share",
    "hardware_efficient",
    "strongly_entangling",
    "simplified_two_design",
    "tree_tensor_network",
    "mps_ansatz",
    "qcnn_ansatz",
    "qaoa_ansatz",
    "conv_block",
    "basic_entangler",
    "two_local",
    "random_layers",
    # gradient dispatch
    "grad",
    "choose_method",
    "register_gradient",
    "list_gradient_methods",
    "adjoint_grad",
    "hadamard_grad",
    "hessian",
    "gradient_cost",
    "supports_adjoint",
    "spsa_grad",
    "minimize_spsa",
    "SPSASchedule",
    # shots
    "standard_error",
    "variance",
    "shots_for_precision",
    "p0_from_z",
    "z_from_p0",
]


def __getattr__(name: str):
    """Expose the torch bridge lazily, so ``import qmlkit`` never requires torch."""
    if name in (
        "QuantumLayer",
        "QuantumFunction",
        "VQC",
        "VQRegressor",
        "HybridModel",
        "QCNNLayer",
        "MPSLayer",
        "QLSTMCell",
        "QLSTM",
        "DressedQuantumNet",
        "nn",
    ):
        try:
            import qmlkit.nn as _nn
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"{name} needs PyTorch, which is an optional extra:\n"
                "    pip install 'qmlkit[torch]'"
            ) from exc
        return _nn if name == "nn" else getattr(_nn, name)
    raise AttributeError(f"module 'qmlkit' has no attribute {name!r}")
