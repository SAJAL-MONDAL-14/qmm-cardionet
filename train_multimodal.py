print("USING QUANTUM MULTIMODAL TRAINING SCRIPT")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_multimodal import MultimodalDataset
from models.multimodal_model import MultimodalQuantumNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultimodalDataset(
    clinical_csv="data/clinical.csv",
    ptbxl_csv="data/ptbxl/ptbxl_database.csv"
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = MultimodalQuantumNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting multimodal training...")

for epoch in range(5):
    total_loss = 0
    for clinical, ecg, y in loader:
        clinical = clinical.to(device)
        ecg = ecg.to(device)
        y = y.unsqueeze(1).to(device)

        optimizer.zero_grad()
        out = model(clinical, ecg)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "best_multimodal_model.pth")
print("Multimodal model saved")
