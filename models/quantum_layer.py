# models/quantum_layer.py
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
import torch.nn as nn

def create_qnn(n_qubits=8):
    qc = QuantumCircuit(n_qubits)

    params = [Parameter(f"θ{i}") for i in range(n_qubits)]

    for i in range(n_qubits):
        qc.ry(params[i], i)

    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    qnn = EstimatorQNN(
        circuit=qc,
        input_params=[],
        weight_params=params
    )

    return TorchConnector(qnn)
