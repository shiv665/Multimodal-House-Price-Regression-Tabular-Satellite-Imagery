"""
Winner Selection: PyTorch vs XGBoost Hybrid Model Comparison
============================================================
This script implements a hybrid approach that:
1. Trains a PyTorch CNN+Tabular model to learn deep features
2. Extracts those learned features to train an XGBoost regressor
3. Compares both models and selects the winner

The idea: Use the neural network's learned representations (visual + tabular)
as input features for XGBoost, potentially getting the best of both worlds.
"""

import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from tqdm import tqdm

from src.config import cfg
from src.data_fetcher import download
from src.datasets import HouseDataset
from torchvision import models


# ==========================================
# 1. HYBRID MULTIMODAL MODEL
# ==========================================
class HybridMultimodalModel(nn.Module):
    """
    Multimodal model with two hemispheres:
    - Visual Hemisphere: ResNet18 CNN encoder for satellite images
    - Logical Hemisphere: MLP encoder for tabular features
    
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
        Used for Phase 2: Training XGBoost on learned representations.
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


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def rmse(pred, true):
    return math.sqrt(mean_squared_error(true, pred))


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the PyTorch model for one epoch."""
    model.train()
    total_loss = 0
    for img, tab, y in tqdm(loader):
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(img, tab)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_model(model, loader, criterion, device):
    """Evaluate the PyTorch model."""
    model.eval()
    ys, ps = [], []
    total_loss = 0
    for img, tab, y in loader:
        img, tab, y = img.to(device), tab.to(device), y.to(device)
        pred = model(img, tab)
        loss = criterion(pred, y)
        total_loss += loss.item() * len(y)
        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())
    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    return total_loss / len(loader.dataset), rmse(ps, ys), r2_score(ys, ps), mean_absolute_error(ys, ps)


def extract_features(model, loader, device, include_targets=True):
    """Extract combined features from the trained model for XGBoost."""
    model.eval()
    features_list = []
    targets_list = []
    
    with torch.no_grad():
        for batch in loader:
            if include_targets:
                img, tab, y = batch
                targets_list.extend(y.numpy())
            else:
                img, tab, _ = batch
            
            img, tab = img.to(device), tab.to(device)
            feats = model.get_features_only(img, tab)
            features_list.append(feats)
    
    features = np.vstack(features_list)
    if include_targets:
        targets = np.array(targets_list)
        return features, targets
    return features


# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    print("=" * 60)
    print("🏠 HOUSE PRICE PREDICTION: WINNER SELECTION")
    print("   PyTorch CNN+Tabular vs Hybrid XGBoost")
    print("=" * 60)
    
    # Device setup
    device = cfg.device
    print(f"\n📍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Create directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ==========================================
    # PHASE 0: LOAD DATA
    # ==========================================
    print("\n" + "=" * 60)
    print("📂 PHASE 0: Loading Data")
    print("=" * 60)
    
    # Load datasets
    train_df = pd.read_excel(cfg.train_xlsx)
    test_df = pd.read_excel(cfg.test_xlsx)
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")

    # Fetch/load satellite images
    img_paths = download(pd.concat([train_df, test_df], axis=0))

    # Filter rows with valid images
    train_df = train_df[train_df["id"].isin(img_paths.keys())]
    test_df = test_df[test_df["id"].isin(img_paths.keys())]
    print(f"   Train samples with images: {len(train_df)}")
    print(f"   Test samples with images: {len(test_df)}")

    # Scale tabular features
    scaler = StandardScaler()
    scaler.fit(train_df[cfg.tab_feats].astype(float))

    # Split train into train/val
    tr_df, val_df = train_test_split(train_df, test_size=cfg.val_split, random_state=cfg.seed)
    print(f"   Training set: {len(tr_df)}")
    print(f"   Validation set: {len(val_df)}")

    # Create datasets
    tr_ds = HouseDataset(tr_df, img_paths, scaler, train=True)
    val_ds = HouseDataset(val_df, img_paths, scaler, train=True)
    te_ds = HouseDataset(test_df, img_paths, scaler, train=False)

    # Create dataloaders
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ==========================================
    # PHASE 1: TRAIN PYTORCH MODEL
    # ==========================================
    print("\n" + "=" * 60)
    print("🧠 PHASE 1: Training Neural Network (CNN + Tabular)")
    print("=" * 60)
    
    model = HybridMultimodalModel(tabular_input_dim=len(cfg.tab_feats)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    
    best_rmse = float("inf")
    best_epoch = 0
    ckpt = torch.load(os.path.join(cfg.model_test, "hybrid_best_model_2.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])
    print("✅ Loaded existing model checkpoint for evaluation.")
    
    for epoch in tqdm(range(cfg.epochs), desc="Training PyTorch"):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        val_loss, val_rmse, val_r2, val_mae = eval_model(model, val_loader, criterion, device)
        
        print(f"   Epoch {epoch+1}/{cfg.epochs} | "
              f"Train Loss: {tr_loss:.4f} | "
              f"Val RMSE: ${val_rmse:,.0f} | "
              f"Val R²: {val_r2:.4f}")
        
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_epoch = epoch + 1
            torch.save({
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            }, os.path.join(cfg.model_dir, "hybrid_best_model.pt"))
    
    print(f"\n✅ Best PyTorch Model: Epoch {best_epoch} with Val RMSE: ${best_rmse:,.0f}")
    
    # Reload best model
    ckpt = torch.load(os.path.join(cfg.model_test, "hybrid_best_model_2.pt"), map_location=device)
    model.load_state_dict(ckpt["model"])

    # ==========================================
    # PHASE 2: EXTRACT FEATURES FOR XGBOOST
    # ==========================================
    print("\n" + "=" * 60)
    print("🔬 PHASE 2: Extracting Deep Features for XGBoost")
    print("=" * 60)
    
    # Extract features from trained model
    X_train_xgb, y_train_xgb = extract_features(model, tr_loader, device, include_targets=True)
    X_val_xgb, y_val_xgb = extract_features(model, val_loader, device, include_targets=True)
    
    print(f"   Training features shape: {X_train_xgb.shape}")
    print(f"   Validation features shape: {X_val_xgb.shape}")
    print(f"   Feature dimension: {X_train_xgb.shape[1]} (256 visual + 64 tabular)")

    # ==========================================
    # PHASE 3: TRAIN XGBOOST
    # ==========================================
    print("\n" + "=" * 60)
    print("🌲 PHASE 3: Training XGBoost on Deep Features")
    print("=" * 60)
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=cfg.seed,
        n_jobs=-1
    )
    
    # Train XGBoost
    xgb_model.fit(
        X_train_xgb, y_train_xgb,
        eval_set=[(X_val_xgb, y_val_xgb)],
        verbose=False
    )
    print("✅ XGBoost Training Complete")

    # ==========================================
    # PHASE 4: THE FINAL SHOWDOWN
    # ==========================================
    print("\n" + "=" * 60)
    print("🏆 PHASE 4: THE FINAL SHOWDOWN - Comparison on Validation Set")
    print("=" * 60)
    
    # A. Get PyTorch Predictions
    model.eval()
    y_pred_pt = []
    y_true = []
    with torch.no_grad():
        for img, tab, y in val_loader:
            img, tab = img.to(device), tab.to(device)
            pred = model(img, tab)
            y_pred_pt.extend(pred.cpu().numpy())
            y_true.extend(y.numpy())
    y_pred_pt = np.array(y_pred_pt)
    y_true = np.array(y_true)
    
    # B. Get XGBoost Predictions
    y_pred_xgb = xgb_model.predict(X_val_xgb)
    
    # C. Calculate Metrics
    pt_rmse = rmse(y_pred_pt, y_true)
    pt_r2 = r2_score(y_true, y_pred_pt)
    pt_mae = mean_absolute_error(y_true, y_pred_pt)
    
    xgb_rmse_val = rmse(y_pred_xgb, y_true)
    xgb_r2 = r2_score(y_true, y_pred_xgb)
    xgb_mae = mean_absolute_error(y_true, y_pred_xgb)
    
    # D. Print Results
    print("\n" + "-" * 60)
    print(f"{'Model':<30} {'RMSE':>12} {'R²':>10} {'MAE':>12}")
    print("-" * 60)
    print(f"{'PyTorch (CNN + Tabular)':<30} ${pt_rmse:>10,.0f} {pt_r2:>10.4f} ${pt_mae:>10,.0f}")
    print(f"{'XGBoost (on Deep Features)':<30} ${xgb_rmse_val:>10,.0f} {xgb_r2:>10.4f} ${xgb_mae:>10,.0f}")
    print("-" * 60)
    
    # E. Declare Winner
    print("\n" + "=" * 60)
    if xgb_r2 > pt_r2:
        print("🏆 WINNER: XGBoost on Deep Features!")
        print("   The hybrid approach (CNN features → XGBoost) wins!")
        winner = "xgboost"
    else:
        print("🏆 WINNER: PyTorch End-to-End Model!")
        print("   The pure deep learning approach wins!")
        winner = "pytorch"
    print("=" * 60)

    # ==========================================
    # PHASE 5: GENERATE TEST PREDICTIONS
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 PHASE 5: Generating Test Predictions")
    print("=" * 60)
    
    # Extract test features
    test_features = []
    test_ids = []
    model.eval()
    with torch.no_grad():
        for img, tab, pid in te_loader:
            img, tab = img.to(device), tab.to(device)
            feats = model.get_features_only(img, tab)
            test_features.append(feats)
            test_ids.extend(pid.tolist())
    X_test = np.vstack(test_features)
    
    # Generate predictions with both models
    # PyTorch predictions
    pt_preds = []
    model.eval()
    with torch.no_grad():
        for img, tab, pid in te_loader:
            img, tab = img.to(device), tab.to(device)
            pred = model(img, tab).cpu().numpy()
            pt_preds.extend(pred.tolist())
    
    # XGBoost predictions
    xgb_preds = xgb_model.predict(X_test)
    
    # Save both submissions
    sub_pt = pd.DataFrame({"id": test_ids, "predicted_price": pt_preds})
    sub_pt.to_csv(os.path.join(cfg.output_dir, "submission_pytorch.csv"), index=False)
    print(f"   Saved: outputs/submission_pytorch.csv")
    
    sub_xgb = pd.DataFrame({"id": test_ids, "predicted_price": xgb_preds})
    sub_xgb.to_csv(os.path.join(cfg.output_dir, "submission_xgboost.csv"), index=False)
    print(f"   Saved: outputs/submission_xgboost.csv")
    
    # Save winner's submission as main
    if winner == "xgboost":
        sub_xgb.to_csv(os.path.join(cfg.output_dir, "submission_winner.csv"), index=False)
    else:
        sub_pt.to_csv(os.path.join(cfg.output_dir, "submission_winner.csv"), index=False)
    print(f"   Saved: outputs/submission_winner.csv (using {winner})")
    
    # Save XGBoost model
    xgb_model.save_model(os.path.join(cfg.model_dir, "xgboost_hybrid.json"))
    print(f"   Saved: models/xgboost_hybrid.json")
    
    print("\n" + "=" * 60)
    print("✅ COMPLETE! Check outputs/ for predictions")
    print("=" * 60)
    
    return winner, {"pytorch": {"rmse": pt_rmse, "r2": pt_r2, "mae": pt_mae},
                    "xgboost": {"rmse": xgb_rmse_val, "r2": xgb_r2, "mae": xgb_mae}}


if __name__ == "__main__":
    winner, metrics = main()
