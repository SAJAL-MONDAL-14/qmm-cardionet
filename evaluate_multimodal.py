import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataset_multimodal import MultimodalDataset
from models.multimodal_model import MultimodalQuantumNet

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MultimodalDataset(
    clinical_csv="data/clinical.csv",
    ptbxl_csv="data/ptbxl/ptbxl_database.csv"
)

loader = DataLoader(dataset, batch_size=16, shuffle=False)

model = MultimodalQuantumNet().to(device)
model.load_state_dict(torch.load("best_multimodal_model.pth", map_location=device))
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for clinical, ecg, y in loader:
        clinical = clinical.to(device)
        ecg = ecg.to(device)

        out = model(clinical, ecg)
        prob = torch.sigmoid(out).cpu().numpy()

        y_true.extend(y.numpy())
        y_pred.extend((prob > 0.5).astype(int).flatten())
        y_prob.extend(prob.flatten())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_prob))
