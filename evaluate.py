import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataset import PTBXLDataset
from models.ecg_1dcnn import ECG1DCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataset
dataset = PTBXLDataset(csv_path="data/ptbxl/ptbxl_database.csv")
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model
model = ECG1DCNN().to(device)
model.load_state_dict(torch.load("best_ecg_model.pth", map_location=device))
model.eval()

y_true, y_pred, y_prob = [], [], []

with torch.no_grad():
    for x, y in loader:
        x = x.to(device)
        out = model(x)

        probs = torch.sigmoid(out).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        y_true.extend(y.numpy())
        y_pred.extend(preds.flatten())
        y_prob.extend(probs.flatten())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1-score :", f1_score(y_true, y_pred))
print("ROC-AUC  :", roc_auc_score(y_true, y_prob))
