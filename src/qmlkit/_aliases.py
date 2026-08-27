"""What the other libraries call it.

A model writing qmlkit has read far more PennyLane and Qiskit than qmlkit, so its
first guess at a name is usually *their* name. ``qk.AngleEmbedding``,
``qk.QuantumCircuit``, ``qk.PauliZ`` — all reasonable, all wrong here, and all
producing the same bare ``AttributeError`` that says only that the name is absent.

This table turns that dead end into a correction. It is a translation, not an
alias: the foreign name still raises. Two reasons for that choice.

*Aliases become API.* Anything importable is something somebody depends on, and a
shadow vocabulary of thirty PennyLane spellings is a second public surface to keep
working forever.

*Aliases hide semantic drift.* ``qml.expval`` takes a QNode; ``qk.expectation``
takes a :class:`~qmlkit.core.ir.CircuitSpec` and an observable. A name that
silently resolves would fail later, further from its cause, with a worse message.
Correcting costs one round trip and leaves the caller holding the real name.

Genuine aliases are still fine where the semantics are *identical* and only the
spelling differs — ``expval`` for ``expectation`` is one, and it is a real export.

``tests/test_agent_api.py`` asserts every target below exists, so the table cannot
drift out of date without the build going red.
"""

from __future__ import annotations

__all__ = ["ELSEWHERE", "advice"]

#: foreign name -> (qmlkit attribute, extra guidance). ``None`` means there is no
#: single equivalent and the note carries the whole answer.
_PENNYLANE: dict[str, tuple[str | None, str]] = {
    "qnode": (None, "qmlkit has no QNode - build a CircuitSpec and call expectation(spec, obs)"),
    "QNode": (None, "qmlkit has no QNode - build a CircuitSpec and call expectation(spec, obs)"),
    "device": ("get_backend", ""),
    "probs": ("probabilities", ""),
    "state": ("statevector", ""),
    "sample": ("run_counts", ""),
    "counts": ("run_counts", ""),
    "jacobian": ("grad", ""),
    "about": ("backend_report", ""),
    "AngleEmbedding": ("AngleFeatureMap", "or angle_encode(x) for a one-shot circuit"),
    "AmplitudeEmbedding": ("amplitude_encode", "or the AmplitudeEncoder feature map"),
    "BasisEmbedding": ("basis_encode", ""),
    "IQPEmbedding": ("ZZFeatureMap", ""),
    "StatePrep": ("amplitude_encode", ""),
    "QubitStateVector": ("amplitude_encode", ""),
    "StronglyEntanglingLayers": ("strongly_entangling", ""),
    "BasicEntanglerLayers": ("basic_entangler", ""),
    "SimplifiedTwoDesign": ("simplified_two_design", ""),
    "RandomLayers": ("random_layers", ""),
    "MPS": ("mps_ansatz", ""),
    "TTN": ("tree_tensor_network", ""),
    "PauliX": ("X", ""),
    "PauliY": ("Y", ""),
    "PauliZ": ("Z", ""),
    "Identity": ("I", ""),
    "Hamiltonian": ("PauliSum", ""),
    "density_matrix": ("reduced_dm", ""),
    "TorchLayer": ("QuantumLayer", ""),
}

_QISKIT: dict[str, tuple[str | None, str]] = {
    "QuantumCircuit": ("QCircuit", ""),
    "Statevector": ("statevector", ""),
    "SparsePauliOp": ("PauliSum", ""),
    "Pauli": ("PauliString", ""),
    "Operator": (None, "build observables from Z/X/Y/I, PauliString or PauliSum"),
    "Estimator": ("expectation", "qmlkit needs no primitive object"),
    "Sampler": ("run_counts", "qmlkit needs no primitive object"),
    "AerSimulator": ("get_backend", 'get_backend("qiskit") runs the same circuit on Aer'),
    "transpile": (None, "qmlkit has no transpiler - a simulator runs the circuit as written"),
    "EfficientSU2": ("hardware_efficient", ""),
    "RealAmplitudes": ("two_local", 'two_local(n, rotations=("ry",), entangler="cx")'),
    "TwoLocal": ("two_local", ""),
    "NLocal": ("two_local", ""),
    "FidelityQuantumKernel": ("QuantumKernel", ""),
    "TrainableFidelityQuantumKernel": ("TrainableKernel", ""),
    "TorchConnector": ("QuantumLayer", "a QuantumLayer already is an nn.Module"),
    "EstimatorQNN": ("QuantumLayer", ""),
    "SamplerQNN": ("QuantumLayer", ""),
    "NeuralNetworkClassifier": ("VQC", ""),
    "NeuralNetworkRegressor": ("VQRegressor", ""),
}

#: Plausible names that belong to no library in particular - the shape of guess a
#: model makes when it is reasoning from the domain rather than from another API.
_GUESSES: dict[str, tuple[str | None, str]] = {
    "Circuit": ("QCircuit", ""),
    "expectation_value": ("expectation", ""),
    "expected_value": ("expectation", ""),
    "parameter_shift": ("param_shift_grad", 'or grad(..., method="parameter-shift")'),
    "parameter_shift_grad": ("param_shift_grad", ""),
    "gradient": ("grad", ""),
    "QNN": ("QuantumLayer", ""),
    "Model": ("VQC", "or VQRegressor for regression"),
    "Classifier": ("VQC", ""),
    "Regressor": ("VQRegressor", ""),
    "encode": ("angle_encode", "or a FeatureMap for something reusable"),
    "measure": ("expectation", "or run_counts for samples"),
    "simulate": ("statevector", ""),
    "Observable": (None, "build observables from Z/X/Y/I, PauliString or PauliSum"),
    "Kernel": ("QuantumKernel", ""),
    "FeatureMapBase": ("FeatureMap", ""),
}

#: Every foreign name, with the library it came from.
ELSEWHERE: dict[str, tuple[str, str | None, str]] = {
    **{k: ("PennyLane", *v) for k, v in _PENNYLANE.items()},
    **{k: ("Qiskit", *v) for k, v in _QISKIT.items()},
    **{k: ("", *v) for k, v in _GUESSES.items()},
}


def advice(name: str) -> str | None:
    """One sentence naming the qmlkit equivalent of ``name``, or ``None``.

    >>> advice("PauliZ")
    "'PauliZ' is PennyLane's name for qmlkit.Z."
    >>> advice("gradient")
    "qmlkit calls that 'grad'."
    >>> advice("not_a_real_name") is None
    True
    """
    if name not in ELSEWHERE:
        return None
    source, target, note = ELSEWHERE[name]
    tail = f" ({note})" if note else ""
    if target is None:
        lead = f"{name!r} is {source}'s;" if source else f"There is no {name!r};"
        return f"{lead} {note}."
    if source:
        return f"{name!r} is {source}'s name for qmlkit.{target}{tail}."
    return f"qmlkit calls that {target!r}{tail}."
