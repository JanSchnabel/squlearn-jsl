""" Class for a modified version of Qiskit's ZFeatureMap"""

import numpy as np
from typing import Union

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from ..feature_map_base import FeatureMapBase


class ZFeatureMap_CX(FeatureMapBase):
    """
    Creates Qiskit's ZFeatureMap 
        (https://qiskit.org/documentation/stubs/qiskit.circuit.library.ZFeatureMap.html)
    with additional CNOT gates between the default layers.

    The number of qubits and the number of features have to be the same!
    
    Args:
        num_qubits (int): The number of qubits
        num_features (int): The number of features
        reps (int): The number of repeated circuits
    """

    def __init__(self, num_qubits: int, num_features: int, reps: int = 2) -> None:
        super().__init__(num_qubits, num_features)
        self._reps = reps

        if self._num_features != self._num_qubits:
            raise ValueError(
                "The number of qubits and the number of features have to be the same!"
            )

    @property
    def num_parameters(self) -> int:
        """Returns the number of trainable parameters of the feature map"""
        return self._num_qubits * self._reps

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
                the trainable parameters of the PQC. Can be either parameter objects
                or numeric values.
        """
        if self._num_features != len(features):
            raise ValueError("Wrong number of features!")

        circuit = QuantumCircuit(self._num_qubits)
        ioff = 0
        for _ in range(self._reps):
            for i in range(self._num_qubits):
                circuit.h(i)
                circuit.p(parameters[ioff] * features[i], i)
                ioff += 1
            if self._reps % 2 == 0:
                for j in range(self._num_qubits - 1):
                    circuit.cx(j, j + 1)
            else:
                for j in range(1, self._num_qubits - 1, 2):
                    circuit.cx(j, j + 1)

        return circuit
