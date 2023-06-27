"""Wrapper class for Qiskit's ZFeatureMap """

import numpy as np
from typing import Union, Optional, Callable
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap

from ..feature_map_base import FeatureMapBase


class QiskitZFeatureMap(FeatureMapBase):
    """
    Wrapper to use Qiskit's ZFeatureMap
    (https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZFeatureMap.html)
    within sQulearn. The feature map represents the first order Pauli Z-evolution
    circuit.

    Args:
        num_qubits (int): The number of qubits
        num_features (int): The number of features
        reps (int): The number of repeated circuits. Defaults to 2, has a
            minimum value of 1
        data_map_func (Optional[Callable[[np.ndarray], float]]):
            A mapping function for data x which can be supplied to override
            the default mapping from self_product()
        parameter_prefix (str): The prefix used if default parameters are generated
        insert_barriers (bool): If True, barriers are inserted in between the evolution
            instructions and hadamard layers
    """

    def __init__(
        self,
        num_qubits: int,
        num_features: int,
        reps: int = 2,
        data_map_func: Optional[Callable[[np.ndarray], float]] = None,
        parameter_prefix: str = "x",
        insert_barriers: bool = False,
    ) -> None:
        super().__init__(num_qubits, num_features)

        self._reps = reps
        self._data_map_func = data_map_func
        self._parameter_prefix = parameter_prefix
        self._insert_barriers = insert_barriers

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map"""
        return 0

    @property
    def num_layers(self) -> int:
        """Returns the number of layers specified for the feature map"""
        return self._reps

    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray],
        parameters: Union[ParameterVector, np.ndarray],
    ) -> QuantumCircuit:
        """
        Returns the circuit of the ZFeatureMap as Qiskit QuantumCircuit object

        Args:
            features (Union[ParameterVector, np.ndarray]): Vector containing the
                data features to be encoded into respective gates
            parameters (Union[ParamaterVector, np.ndarray]): Vector containing
                the trainable parameters of the PQC. For the ZFeatureMap this
                is only required to meet the sQUlearn format.
        """
        if self._num_features != len(features):
            raise ValueError("Wrong number of features in supplied vector")

        # if self.num_parameters != len(parameters):
        #    raise ValueError("Wrong number of parameters in supplied vector")

        circuit_z_feature_map = ZFeatureMap(
            feature_dimension=self._num_features,
            reps=self._reps,
            data_map_func=self._data_map_func,
            parameter_prefix=self._parameter_prefix,
            insert_barriers=self._insert_barriers,
        )

        return circuit_z_feature_map
