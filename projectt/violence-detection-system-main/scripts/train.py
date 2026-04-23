import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.cnn_lstm import CNNLSTM

# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "train", "violence")
TRAIN_NON_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "train", "non_violence")

VAL_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "val", "violence")
VAL_NON_VIOLENCE = os.path.join(BASE_DIR, "data", "frames", "val", "non_violence")

MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "cnn_lstm.pth")

# =========================================================
# DATASET
# =========================================================
class ViolenceDataset(Dataset):
    def __init__(self, root_dir, label, seq_len=32, transform=None):
        self.root_dir = root_dir
        self.label = label
        self.seq_len = seq_len
        self.transform = transform
        self.videos = os.listdir(root_dir)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.videos[idx])
        frames = sorted(os.listdir(video_path))

        # 🔴 HANDLE EMPTY FRAME FOLDERS
        if len(frames) == 0:
            # randomly pick another sample
            new_idx = torch.randint(0, len(self.videos), (1,)).item()
            return self.__getitem__(new_idx)

        # FIX SEQUENCE LENGTH
        if len(frames) >= self.seq_len:
            frames = frames[:self.seq_len]
        else:
            frames += [frames[-1]] * (self.seq_len - len(frames))

        imgs = []
        for f in frames:
            img = Image.open(os.path.join(video_path, f)).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        imgs = torch.stack(imgs)
        label = torch.tensor(self.label)

        return imgs, label


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
# DATA LOADERS
# =========================================================
train_dataset = torch.utils.data.ConcatDataset([
    ViolenceDataset(TRAIN_VIOLENCE, 1, transform=transform),
    ViolenceDataset(TRAIN_NON_VIOLENCE, 0, transform=transform)
])

val_dataset = torch.utils.data.ConcatDataset([
    ViolenceDataset(VAL_VIOLENCE, 1, transform=transform),
    ViolenceDataset(VAL_NON_VIOLENCE, 0, transform=transform)
])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# =========================================================
# MODEL
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Training on device:", device)

model = CNNLSTM(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

# =========================================================
# TRAIN + VALIDATE
# =========================================================
epochs = 10
best_val_acc = 0.0

for epoch in range(epochs):
    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]"):
        x, y = x.to(device), y.to(device)

        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_acc = correct / total

    # ---------------- VALIDATE ----------------
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [VAL]"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            val_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_acc = correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss/len(val_loader):.4f} | Val   Acc: {val_acc:.4f}")

    # ---------------- SAVE BEST MODEL ----------------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)
        print("🔥 Best model saved")

print("\n✅ Training complete")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")
