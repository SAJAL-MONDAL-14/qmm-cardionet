import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PTBXLDataset
from models.ecg_1dcnn import ECG1DCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading dataset...")

dataset = PTBXLDataset(
    csv_path="data/ptbxl/ptbxl_database.csv"
)

print("Dataset size:", len(dataset))

loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = ECG1DCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting training...")

for epoch in range(3):
    model.train()
    loss_sum = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch+1}, Loss: {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(), "best_ecg_model.pth")
print("Training finished and model saved")
