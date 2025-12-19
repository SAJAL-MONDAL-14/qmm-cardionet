import torch
from PIL import Image
from torchvision import transforms
from models.multimodal_model import QMMCardioNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Load trained model
# -----------------------
model = QMMCardioNet().to(device)
model.load_state_dict(torch.load("best_qmm_cardionet.pth"))
model.eval()

# -----------------------
# Image preprocessing
# -----------------------
img_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_heart_disease(clinical_input, ecg_image_path):
    """
    clinical_input: list of 7 values
    ecg_image_path: path to ECG image
    """

    clinical = torch.tensor(clinical_input, dtype=torch.float32).unsqueeze(0).to(device)

    image = Image.open(ecg_image_path)
    image = img_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(clinical, image)
        prob = torch.sigmoid(output).item()

    if prob > 0.5:
        return "YES (High Risk)", prob
    else:
        return "NO (Low Risk)", prob
