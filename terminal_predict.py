import torch
from models.clinical_net import ClinicalNet

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ClinicalNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

print("\n HEART DISEASE PREDICTION (CLINICAL DATA)\n")

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

x = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(x)
    prob = torch.sigmoid(output).item()

print("\n RESULT")
print("---------------------")

if prob > 0.5:
    print("Heart Disease: YES (High Risk)")
else:
    print("Heart Disease: NO (Low Risk)")

print(f"Confidence: {prob*100:.2f}%")
