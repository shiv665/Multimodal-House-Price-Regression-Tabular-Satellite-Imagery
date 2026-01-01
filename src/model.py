import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    """
    Late fusion: CNN encoder + MLP on tabular, concatenated head -> regression.
    Swap backbone (resnet18/50, efficientnet) as needed.
    """
    def __init__(self, tab_in: int, tab_hidden: int = 128, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        n_feats = cnn.fc.in_features
        cnn.fc = nn.Identity()
        self.cnn = cnn

        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_in, tab_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(tab_hidden),
            nn.Dropout(0.2),
            nn.Linear(tab_hidden, tab_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(tab_hidden),
        )

        self.head = nn.Sequential(
            nn.Linear(n_feats + tab_hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, img, tab):
        v = self.cnn(img)
        t = self.tab_mlp(tab)
        x = torch.cat([v, t], dim=1)
        out = self.head(x).squeeze(1)
        return out