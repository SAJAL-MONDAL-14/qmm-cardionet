import torch
import wfdb
from models.multimodal_model import MultimodalNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MultimodalNet().to(device)
model.load_state_dict(torch.load("best_multimodal_model.pth", map_location=device))
model.eval()

print("\n HEART DISEASE PREDICTION (MULTIMODAL)\n")

# ------------------ Clinical Input ------------------
inputs = []
inputs.append(float(input("Age: ")))
inputs.append(float(input("Sex (1=Male, 0=Female): ")))
inputs.append(float(input("Chest pain type (0-3): ")))
inputs.append(float(input("Resting BP: ")))
inputs.append(float(input("Cholesterol: ")))
inputs.append(float(input("Fasting blood sugar (1=True, 0=False): ")))
inputs.append(float(input("Rest ECG (0-2): ")))
inputs.append(float(input("Max heart rate: ")))
inputs.append(float(input("Exercise angina (1=Yes, 0=No): ")))
inputs.append(float(input("Oldpeak: ")))
inputs.append(float(input("ST slope (0-2): ")))
inputs.append(float(input("Major vessels (0-3): ")))
inputs.append(float(input("Thal (0=normal,1=fixed,2=reversible): ")))

clinical_x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)

# ------------------ ECG Input ------------------
ecg_path = input("\nEnter ECG record path (without .dat/.hea): ")

signal, _ = wfdb.rdsamp(ecg_path)
signal = torch.tensor(signal[:, 0], dtype=torch.float32)

if len(signal) > 5000:
    signal = signal[:5000]
else:
    signal = torch.nn.functional.pad(signal, (0, 5000 - len(signal)))

ecg_x = signal.unsqueeze(0).to(device)

# ------------------ Prediction ------------------
with torch.no_grad():
    out = model(clinical_x, ecg_x)
    prob = torch.sigmoid(out).item()

print("\n FINAL RESULT")
print("---------------------")

if prob > 0.5:
    print("Heart Disease: YES (High Risk)")
else:
    print("Heart Disease: NO (Low Risk)")

print(f"Confidence: {prob*100:.2f}%")
