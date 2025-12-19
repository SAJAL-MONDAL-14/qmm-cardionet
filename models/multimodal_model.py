import torch
import torch.nn as nn

class ECGBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        return x.view(x.size(0), -1)


class ClinicalBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc(x)


class MultimodalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.ecg = ECGBranch()
        self.clinical = ClinicalBranch()

        self.classifier = nn.Sequential(
            nn.Linear(32 + 32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, clinical_x, ecg_x):
        ecg_feat = self.ecg(ecg_x)
        clin_feat = self.clinical(clinical_x)

        fused = torch.cat([ecg_feat, clin_feat], dim=1)
        return self.classifier(fused)
