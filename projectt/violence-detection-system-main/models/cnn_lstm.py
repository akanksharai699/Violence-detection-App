import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.cnn = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)

        features = self.cnn(x)
        features = features.view(B, T, -1)

        _, (h_n, _) = self.lstm(features)
        out = self.fc(h_n[-1])

        return out
