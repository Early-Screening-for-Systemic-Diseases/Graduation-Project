import torch
import torch.nn as nn
from torchvision import models

# Diabetes Classifier Model
class DiabetesClassifier(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm1d(base_model.last_channel)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(base_model.last_channel, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.output(x)
        x = torch.sigmoid(x)
        return x


# Function to load model easily
def load_diabetes_model(weight_path):
    base_model = models.mobilenet_v2(weights=None)
    model = DiabetesClassifier(base_model)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model
