import torch
import torch.nn as nn
from torchvision import models


class HybridMultimodalModel(nn.Module):
    """
    Hybrid Multimodal Model (Winner Pipeline - R² 0.87):
    
    Two Hemispheres Architecture:
    - Visual Hemisphere: ResNet18 CNN → 512 → 256 (with BatchNorm + Dropout)
    - Logical Hemisphere: Tabular MLP → 128 → 64 (with BatchNorm + Dropout)
    
    Combined: 320 features → 128 → 64 → 1
    
    Can output either:
    - Final price prediction (for PyTorch training)
    - Combined feature vector (for XGBoost training)
    """
    def __init__(self, tabular_input_dim: int, backbone: str = "resnet18"):
        super(HybridMultimodalModel, self).__init__()
        
        # A. Visual Hemisphere (ResNet18/50 CNN)
        if backbone == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            cnn_out_features = 512
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            cnn_out_features = 2048
            
        # Remove classification head, keep feature extractor
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_fc = nn.Sequential(
            nn.Linear(cnn_out_features, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2)
        )

        # B. Logical Hemisphere (Tabular MLP)
        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        
        # C. The Regressor Head
        # Input = 256 (Visual) + 64 (Tabular) = 320
        self.feature_dim = 256 + 64  # Combined feature dimension
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, image, tabular):
        """Standard forward pass for training the CNN."""
        # Visual features
        img_feat = self.image_encoder(image)
        img_feat = img_feat.view(img_feat.size(0), -1)  # Flatten
        img_feat = self.visual_fc(img_feat)
        
        # Tabular features
        tab_feat = self.tabular_encoder(tabular)
        
        # Combine and regress
        combined = torch.cat((img_feat, tab_feat), dim=1)
        price = self.regressor(combined)
        return price.squeeze(1)

    def get_features_only(self, image, tabular):
        """
        Extract the 320-dimensional combined feature vector.
        Used for XGBoost hybrid training on learned representations.
        """
        self.eval()
        with torch.no_grad():
            # Visual features
            img_feat = self.image_encoder(image)
            img_feat = img_feat.view(img_feat.size(0), -1)
            img_feat = self.visual_fc(img_feat)
            
            # Tabular features
            tab_feat = self.tabular_encoder(tabular)
            
            # Combined features
            combined = torch.cat((img_feat, tab_feat), dim=1)
        return combined.cpu().numpy()


# Alias for backward compatibility
FusionModel = HybridMultimodalModel