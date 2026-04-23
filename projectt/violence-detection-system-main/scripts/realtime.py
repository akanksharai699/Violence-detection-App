import os
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from collections import deque
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
print("🔥 Realtime running on:", device)

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
# PARAMETERS
# =========================================================
SEQ_LEN = 16           # frames per prediction
PRED_EVERY = 4         # predict every N frames
THRESHOLD = 0.45       # violence probability threshold

# =========================================================
# VIDEO CAPTURE
# =========================================================
cap = cv2.VideoCapture("test_videos/test_video2.mp4")

frame_buffer = deque(maxlen=SEQ_LEN)
frame_count = 0

label = "WAITING"
confidence = 0.0

print("🎥 Press 'q' to quit")

# =========================================================
# MAIN LOOP
# =========================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Prepare frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = transform(img)
    frame_buffer.append(img)

    # Run prediction every PRED_EVERY frames
    if len(frame_buffer) == SEQ_LEN and frame_count % PRED_EVERY == 0:
        clip = torch.stack(list(frame_buffer))          # (T, C, H, W)
        clip = clip.unsqueeze(0).to(device)              # (1, T, C, H, W)

        with torch.no_grad():
            outputs = model(clip)
            probs = F.softmax(outputs, dim=1)
            violence_prob = probs[0][1].item()

        if violence_prob > THRESHOLD:
            label = "VIOLENCE"
        else:
            label = "NON-VIOLENCE"

        confidence = violence_prob * 100

    # Overlay
    color = (0, 0, 255) if label == "VIOLENCE" else (0, 255, 0)
    text = f"{label} ({confidence:.1f}%)"

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2
    )

    cv2.imshow("Violence Detection (Realtime)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
