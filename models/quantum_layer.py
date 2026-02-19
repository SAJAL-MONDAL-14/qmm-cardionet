import pennylane as qml
import torch
import torch.nn as nn

from qiskit_ibm_runtime import QiskitRuntimeService

USE_IBM_QUANTUM = True
USE_NOISE = False

n_qubits = 4
n_layers = 1

# =========================
# DEVICE
# =========================
if USE_IBM_QUANTUM:
    service = QiskitRuntimeService()
    backend = service.backend("ibm_fez")

    dev = qml.device(
        "qiskit.remote",
        wires=n_qubits,
        backend=backend,
        shots=1024
    )
else:
    dev = qml.device("default.qubit", wires=n_qubits)


# =========================
# QUANTUM CIRCUIT (DEFINE BEFORE CLASS)
# =========================
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    if USE_NOISE:
        for i in range(n_qubits):
            qml.DepolarizingChannel(0.02, wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# =========================
# QUANTUM LAYER
# =========================
class QuantumLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3)
        )

    def forward(self, x):

        outputs = []

        for i in range(x.shape[0]):
            q_out = quantum_circuit(x[i], self.weights)
            q_out = torch.stack(q_out)
            outputs.append(q_out)

        return torch.stack(outputs).float()
