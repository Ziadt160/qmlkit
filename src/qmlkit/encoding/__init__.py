"""Encoding: getting classical data into a circuit."""

from qmlkit.encoding.amplitude import (
    amplitude_encode,
    pad_to_power_of_two,
    state_preparation_angles,
    uniformly_controlled_rotation,
)
from qmlkit.encoding.angle import angle_encode, basis_encode, basis_index, n_qubits_for
from qmlkit.encoding.feature_maps import (
    AngleFeatureMap,
    FeatureMap,
    PauliFeatureMap,
    ZFeatureMap,
    ZZFeatureMap,
    basis_change,
    default_data_map,
    pauli_terms,
)
from qmlkit.encoding.hamiltonian import (
    DataReuploadEncoder,
    hamiltonian_encode,
    n_reachable_frequencies,
    trotter_rz_angle,
    trotter_zz_angle,
)
from qmlkit.encoding.pipeline import FeaturePipeline, SklearnCompatible
from qmlkit.encoding.scaling import (
    AngleScaler,
    PCAReducer,
    reduce_to_qubits,
    to_angle_range,
)

__all__ = [
    "angle_encode",
    "basis_encode",
    "basis_index",
    "n_qubits_for",
    "amplitude_encode",
    "pad_to_power_of_two",
    "uniformly_controlled_rotation",
    "state_preparation_angles",
    "FeatureMap",
    "PauliFeatureMap",
    "ZFeatureMap",
    "ZZFeatureMap",
    "AngleFeatureMap",
    "default_data_map",
    "basis_change",
    "pauli_terms",
    "hamiltonian_encode",
    "trotter_rz_angle",
    "trotter_zz_angle",
    "DataReuploadEncoder",
    "n_reachable_frequencies",
    "to_angle_range",
    "FeaturePipeline",
    "SklearnCompatible",
    "AngleScaler",
    "reduce_to_qubits",
    "PCAReducer",
]
