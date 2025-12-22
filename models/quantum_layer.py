import pennylane as qml
import torch
import torch.nn as nn
import numpy as np

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(3, n_qubits, 3)
        )

    def forward(self, x):
        """
        x shape: [batch_size, n_qubits]
        """
        outputs = []

        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.weights)
            q_out = torch.stack(q_out)   # convert list → tensor
            outputs.append(q_out)

        return torch.stack(outputs).float()

