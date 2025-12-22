import torch
import torch.nn as nn
from models.quantum_layer import QuantumLayer

class ECGBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.net(x)
        return x.view(x.size(0), -1)

class ClinicalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 16),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class MultimodalQuantumNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg = ECGBranch()
        self.clinical = ClinicalBranch()

        self.fusion = nn.Linear(16 + 16, 4)  # must match n_qubits
        self.quantum = QuantumLayer()

        self.classifier = nn.Linear(4, 1)

    def forward(self, clinical_x, ecg_x):
        ecg_feat = self.ecg(ecg_x)
        clin_feat = self.clinical(clinical_x)

        fused = torch.cat([ecg_feat, clin_feat], dim=1)
        fused = self.fusion(fused)

        q_out = self.quantum(fused)
        q_out = q_out.float()          
        return self.classifier(q_out)