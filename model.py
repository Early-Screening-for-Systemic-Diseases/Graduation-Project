import torch
import torch.nn as nn
from torchvision import models

class HbNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=None)  # NO download
        backbone.fc = nn.Identity()               # 512 features
        self.backbone = backbone

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)
