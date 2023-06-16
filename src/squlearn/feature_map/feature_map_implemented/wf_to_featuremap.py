import numpy as np
from typing import Union, Optional
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from ..feature_map_base import FeatureMapBase

class WavefunctionToFeatureMap(FeatureMapBase):

    def __init__(
        self,
        wf_circuit: QuantumCircuit,
        unitary: QuantumCircuit,
        num_qubits: int,
        num_features: Union[int, None]
    ) -> None:
        super().__init__(num_qubits, num_features)
        self.wf_circuit = wf_circuit
        self.unitary = unitary
        if self.unitary.num_qubits != self.wf_circuit.num_qubits:
            raise ValueError("Number of qubits must be equal to those in wf_circuit")

    @property
    def num_parameters(self) -> int:
        return self.unitary.num_parameters
    
    def get_circuit(
        self,
        features: Union[ParameterVector, np.ndarray] = None,
        parameters: Union[ParameterVector, np.ndarray] = None
    ) -> QuantumCircuit:
        """
        Returns the processed wavefunction U\ket{\psi} as
        FeatureMapBase object
        """
        qc = QuantumCircuit(self.num_qubits)
        circ1 = self.wf_circuit
        circ2 = self.unitary.assign_parameters(parameters)
        qc = circ1.compose(circ2)
        return qc
    
