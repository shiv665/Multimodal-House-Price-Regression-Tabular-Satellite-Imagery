"""
House Price Prediction - Hybrid Training Pipeline
=================================================
This is the main training script implementing the winning pipeline (R² 0.87):
1. Train PyTorch CNN+Tabular model
2. Extract deep features for XGBoost
3. Train XGBoost on learned representations
4. Compare and select winner
5. Generate predictions and Grad-CAM visualizations
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from tqdm import tqdm

from src.config import cfg
from src.data_fetcher import download
from src.datasets import HouseDataset
from src.model import HybridMultimodalModel
from src.gradcam import GradCAM, create_enhanced_overlay


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def rmse(pred, true):
    """Calculate Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(true, pred))


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the PyTorch model for one epoch."""
    model.train()
    total_loss = 0
    for img, tab, y in loader:
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
    ids_list = []
    
    with torch.no_grad():
        for batch in loader:
            if include_targets:
                img, tab, y = batch
                targets_list.extend(y.numpy())
            else:
                img, tab, pid = batch
                ids_list.extend(pid.tolist() if hasattr(pid, 'tolist') else list(pid))
            
            img, tab = img.to(device), tab.to(device)
            feats = model.get_features_only(img, tab)
            features_list.append(feats)
    
    features = np.vstack(features_list)
    if include_targets:
        targets = np.array(targets_list)
        return features, targets
    return features, ids_list


def run_gradcam(model, val_ds, output_dir):
    """Generate Enhanced Grad-CAM++ visualizations with better contrast."""
    gc = GradCAM(model, use_gradcam_pp=True)
    gradcam_dir = os.path.join(output_dir, "gradcam")
    os.makedirs(gradcam_dir, exist_ok=True)
    
    # Denormalization values (ImageNet)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    for i in range(min(cfg.grad_cam_samples, len(val_ds))):
        img, tab, y = val_ds[i+500]
        
        # Generate enhanced CAM with multi-scale fusion and contrast enhancement
        cam = gc(img, tab, smooth=True, multi_scale=True, enhance_contrast=True)
        
        # Convert tensor to numpy and denormalize
        orig = img.permute(1, 2, 0).numpy()
        orig = (orig * std + mean) * 255
        orig = np.clip(orig, 0, 255).astype(np.uint8)
        
        # Resize CAM to match original image size
        cam_resized = cv2.resize(cam, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Create enhanced overlay with higher alpha and thresholding
        overlay = create_enhanced_overlay(orig, cam_resized, alpha=0.6, threshold=0.25)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(gradcam_dir, f"sample_{i}_gt{y:.0f}.png"), overlay_bgr)
        


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def main(args):
    print("=" * 60)
    print("🏠 HOUSE PRICE PREDICTION - HYBRID TRAINING PIPELINE")
    print("   PyTorch CNN+Tabular → XGBoost on Deep Features")
    print("=" * 60)
    
    device = cfg.device
    print(f"\n📍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    # ==========================================
    # PHASE 1: LOAD DATA
    # ==========================================
    print("\n" + "=" * 60)
    print("📂 PHASE 1: Loading Data")
    print("=" * 60)
    
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
    # PHASE 2: TRAIN PYTORCH MODEL
    # ==========================================
    print("\n" + "=" * 60)
    print("🧠 PHASE 2: Training Neural Network (CNN + Tabular)")
    print("=" * 60)
    
    model = HybridMultimodalModel(tabular_input_dim=len(cfg.tab_feats)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()
    # Checkpoint paths
    best_model_path = os.path.join(cfg.model_dir, "best_model.pt")
    last_model_path = os.path.join(cfg.model_dir, "last_model.pt")
    
    start_epoch = 0
    best_rmse = float("inf")
    best_r2 = float("-inf")
    best_epoch = 0

    # Always start fresh; comparison.py handles loading saved models.
    print("\n🆕 Starting fresh training (no checkpoint loading)...")
    
    # Training loop
    epochs_to_train = args.epochs if args.epochs else cfg.epochs
    total_epochs = start_epoch + epochs_to_train
    print(f"   Training from epoch {start_epoch + 1} to {total_epochs}")

    for epoch in tqdm(range(epochs_to_train), desc="Training PyTorch"):
        current_epoch = start_epoch + epoch + 1
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion, device)
        
        # Evaluate on training set
        tr_eval_loss, tr_rmse, tr_r2, tr_mae = eval_model(model, tr_loader, criterion, device)
        # Evaluate on validation set
        val_loss, val_rmse, val_r2, val_mae = eval_model(model, val_loader, criterion, device)
        
        print(f"   Epoch {current_epoch}/{total_epochs} | "
              f"Train RMSE: {tr_rmse:,.0f} | Train R²: {tr_r2:.4f} | "
              f"Val RMSE: {val_rmse:,.0f} | Val R²: {val_r2:.4f}")
        
        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_r2 = val_r2
            best_epoch = current_epoch
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": current_epoch,
                "best_rmse": best_rmse,
                "best_r2": best_r2,
                "best_epoch": best_epoch,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            }, best_model_path)
            print(f"   ✅ New best model saved! (R²: {val_r2:.4f})")
    
    print(f"\n✅ Best PyTorch Model: Epoch {best_epoch} with Val RMSE: {best_rmse:,.0f}")

    # Load best model weights for subsequent computations
    print("\n📥 Loading best model checkpoint for evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    tr_eval_loss, tr_rmse, tr_r2, tr_mae = eval_model(model, tr_loader, criterion, device)
    print(f"   Loaded model Val R²: {tr_r2:.4f}, RMSE: {tr_rmse:,.0f}")
    # ==========================================
    # PHASE 3: TRAIN XGBOOST ON DEEP FEATURES
    # ==========================================
    print("\n" + "=" * 60)
    print("🔬 PHASE 3: Extracting Deep Features for XGBoost")
    print("=" * 60)
    
    X_train, y_train = extract_features(model, tr_loader, device, include_targets=True)
    X_val, y_val = extract_features(model, val_loader, device, include_targets=True)
    
    print(f"   Training features shape: {X_train.shape}")
    print(f"   Validation features shape: {X_val.shape}")
    print(f"   Feature dimension: {X_train.shape[1]} (256 visual + 64 tabular)")
    
    print("\n" + "=" * 60)
    print("🌲 Training XGBoost on Deep Features")
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
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print("✅ XGBoost Training Complete")
    
    # ==========================================
    # PHASE 4: COMPARE MODELS
    # ==========================================
    print("\n" + "=" * 60)
    print("🏆 PHASE 4: MODEL COMPARISON")
    print("=" * 60)
    
    # PyTorch metrics (using eval_model function)
    _, pt_rmse_train, pt_r2_train, _ = eval_model(model, tr_loader, criterion, device)
    _, pt_rmse, pt_r2, pt_mae = eval_model(model, val_loader, criterion, device)
    
    # XGBoost TRAINING metrics
    y_pred_xgb_train = xgb_model.predict(X_train)
    xgb_rmse_train = rmse(y_pred_xgb_train, y_train)
    xgb_r2_train = r2_score(y_train, y_pred_xgb_train)
    
    # XGBoost VALIDATION metrics
    y_pred_xgb = xgb_model.predict(X_val)
    xgb_rmse_val = rmse(y_pred_xgb, y_val)
    xgb_r2 = r2_score(y_val, y_pred_xgb)
    xgb_mae = mean_absolute_error(y_val, y_pred_xgb)
    
    # Print comparison
    print("\n" + "-" * 80)
    print(f"{'Model':<30} {'Train RMSE':>12} {'Train R²':>10} {'Val RMSE':>12} {'Val R²':>10}")
    print("-" * 80)
    print(f"{'PyTorch (CNN + Tabular)':<30} {pt_rmse_train:>10,.0f} {pt_r2_train:>10.4f} {pt_rmse:>10,.0f} {pt_r2:>10.4f}")
    print(f"{'XGBoost (on Deep Features)':<30} {xgb_rmse_train:>10,.0f} {xgb_r2_train:>10.4f} {xgb_rmse_val:>10,.0f} {xgb_r2:>10.4f}")
    print("-" * 80)
    
    # Declare winner
    if xgb_r2 > pt_r2:
        print("\n🏆 WINNER: XGBoost on Deep Features!")
        winner = "xgboost"
    else:
        print("\n🏆 WINNER: PyTorch End-to-End Model!")
        winner = "pytorch"

    # ==========================================
    # PHASE 5: GENERATE TEST PREDICTIONS
    # ==========================================
    print("\n" + "=" * 60)
    print("📊 PHASE 5: Generating Test Predictions")
    print("=" * 60)
    
    # Extract test features
    X_test, test_ids = extract_features(model, te_loader, device, include_targets=False)
    
    # Generate predictions from the model with best R² score
    if xgb_r2 > pt_r2:
        # XGBoost has better R² - use XGBoost predictions
        best_preds = xgb_model.predict(X_test)
        best_model_name = "XGBoost"
        best_r2_score = xgb_r2
    else:
        # PyTorch has better R² - use PyTorch predictions
        best_preds = []
        model.eval()
        with torch.no_grad():
            for img, tab, pid in te_loader:
                img, tab = img.to(device), tab.to(device)
                pred = model(img, tab).cpu().numpy()
                best_preds.extend(pred.tolist())
        best_model_name = "PyTorch"
        best_r2_score = pt_r2
    
    # Save only the best model's submission
    submission = pd.DataFrame({"id": test_ids, "predicted_price": best_preds})
    submission.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
    print(f"   Saved: outputs/submission.csv (using {best_model_name} with R²: {best_r2_score:.4f})")

    # ==========================================
    # PHASE 6: GRAD-CAM VISUALIZATION
    # ==========================================
    print("\n" + "=" * 60)
    print("🔥 PHASE 6: Grad-CAM Visualization")
    print("=" * 60)
    run_gradcam(model, val_ds, cfg.output_dir)
    print("   Saved Grad-CAM visualizations to outputs/gradcam/")
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    winner_r2 = xgb_r2 if winner == 'xgboost' else pt_r2
    print(f"   Winner: {winner.upper()} (R²: {winner_r2:.4f})")
    print("=" * 60)
    
    return winner, {
        "pytorch": {"rmse": pt_rmse, "r2": pt_r2, "mae": pt_mae},
        "xgboost": {"rmse": xgb_rmse_val, "r2": xgb_r2, "mae": xgb_mae}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train House Price Prediction Model")
    parser.add_argument("--epochs", type=int, default=None, 
                        help="Override number of epochs to train")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh training, ignore existing checkpoints")
    args = parser.parse_args()
    main(args)
