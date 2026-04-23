import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from models.cnn_lstm import CNNLSTM

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_lstm.pth")


# =========================================================
# DEVICE
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# TRANSFORMS
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================================================
# LOAD MODEL
# =========================================================
model = CNNLSTM(num_classes=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# =========================================================
# FRAME EXTRACTION (TEMP)
# =========================================================
def extract_frames(video_path, seq_len=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < seq_len:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = transform(img)
        frames.append(img)

    cap.release()

    if len(frames) < seq_len:
        frames += [frames[-1]] * (seq_len - len(frames))

    return torch.stack(frames).unsqueeze(0)

# =========================================================
# PREDICTION
# =========================================================
def predict(video_path):
    sequence = extract_frames(video_path).to(device)

    with torch.no_grad():
        outputs = model(sequence)
        probs = F.softmax(outputs, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "VIOLENCE" if prediction.item() == 1 else "NON-VIOLENCE"
    return label, confidence.item()

# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    test_video = input("Enter video path: ").strip()

    label, conf = predict(test_video)

    print("\n==============================")
    print(f"Prediction : {label}")
    print(f"Confidence : {conf * 100:.2f}%")
    print("==============================\n")
