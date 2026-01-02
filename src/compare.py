"""
Model Comparison - Load PyTorch Model and Train XGBoost On-the-Fly
==================================================================
This script:
1. Loads the saved PyTorch hybrid model
2. Extracts deep features from the model
3. Trains XGBoost on those features (fresh each time)
4. Compares PyTorch vs XGBoost performance
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.config import cfg
from src.data_fetcher import download
from src.datasets import HouseDataset
from src.model import HybridMultimodalModel


def rmse(pred, true):
    """Calculate Root Mean Squared Error."""
    return math.sqrt(mean_squared_error(true, pred))


def load_pytorch_model(model_path, device, tabular_dim):
    """Load PyTorch model from checkpoint."""
    model = HybridMultimodalModel(tabular_input_dim=tabular_dim).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    info = {
        "epoch": ckpt.get("epoch", "N/A"),
        "best_rmse": ckpt.get("best_rmse", "N/A"),
        "best_r2": ckpt.get("best_r2", "N/A"),
    }
    return model, info


@torch.no_grad()
def evaluate_pytorch(model, loader, device):
    """Evaluate PyTorch model and return metrics."""
    model.eval()
    ys, ps = [], []
    
    for img, tab, y in loader:
        img, tab = img.to(device), tab.to(device)
        pred = model(img, tab)
        ys.extend(y.numpy())
        ps.extend(pred.cpu().numpy())
    
    ys = np.array(ys)
    ps = np.array(ps)
    
    return {
        "rmse": rmse(ps, ys),
        "r2": r2_score(ys, ps),
        "mae": mean_absolute_error(ys, ps),
        "predictions": ps,
        "targets": ys
    }


def extract_features(model, loader, device, include_targets=True):
    """Extract features from PyTorch model for XGBoost."""
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
        return features, np.array(targets_list)
    return features, ids_list


def main(args):
    print("=" * 70)
    print("🔍 MODEL COMPARISON - PyTorch vs XGBoost (trained on-the-fly)")
    print("=" * 70)
    
    device = cfg.device
    print(f"\n📍 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ==========================================
    # LOAD DATA
    # ==========================================
    print("\n" + "=" * 70)
    print("📂 Loading Data")
    print("=" * 70)
    
    train_df = pd.read_excel(cfg.train_xlsx)
    test_df = pd.read_excel(cfg.test_xlsx)
    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")

    # Fetch/load satellite images
    img_paths = download(pd.concat([train_df, test_df], axis=0))

    # Filter rows with valid images
    train_df = train_df[train_df["id"].isin(img_paths.keys())]
    test_df = test_df[test_df["id"].isin(img_paths.keys())]

    # Scale tabular features
    scaler = StandardScaler()
    scaler.fit(train_df[cfg.tab_feats].astype(float))

    # Split train into train/val (same split as training)
    tr_df, val_df = train_test_split(train_df, test_size=cfg.val_split, random_state=cfg.seed)
    print(f"   Training set: {len(tr_df)}")
    print(f"   Validation set: {len(val_df)}")
    print(f"   Test set: {len(test_df)}")

    # Create datasets
    tr_ds = HouseDataset(tr_df, img_paths, scaler, train=True)
    val_ds = HouseDataset(val_df, img_paths, scaler, train=True)
    te_ds = HouseDataset(test_df, img_paths, scaler, train=False)

    # Create dataloaders
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # ==========================================
    # LOAD PYTORCH MODEL
    # ==========================================
    print("\n" + "=" * 70)
    print("📦 Loading PyTorch Model")
    print("=" * 70)
    
    model_path = os.path.join(cfg.model_test, "hybrid_best_model_2.pt")
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the model first using: python -m src.winner")
        return
    
    model, info = load_pytorch_model(model_path, device, len(cfg.tab_feats))
    print(f"   ✅ Loaded hybrid_best_model.pt")
    print(f"      Epoch: {info['epoch']}")
    print(f"      Best RMSE: {info['best_rmse']}")
    print(f"      Best R²: {info['best_r2']}")

    # ==========================================
    # EXTRACT FEATURES
    # ==========================================
    print("\n" + "=" * 70)
    print("🔬 Extracting Deep Features from PyTorch Model")
    print("=" * 70)
    
    print("   Extracting training features...")
    X_train, y_train = extract_features(model, tr_loader, device)
    print(f"   Training features: {X_train.shape}")
    
    print("   Extracting validation features...")
    X_val, y_val = extract_features(model, val_loader, device)
    print(f"   Validation features: {X_val.shape}")
    
    print(f"   Feature dimension: {X_train.shape[1]} (256 visual + 64 tabular)")

    # ==========================================
    # TRAIN XGBOOST ON-THE-FLY
    # ==========================================
    print("\n" + "=" * 70)
    print("🌲 Training XGBoost on Deep Features")
    print("=" * 70)
    
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
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    print("   ✅ XGBoost Training Complete")

    # ==========================================
    # EVALUATE MODELS
    # ==========================================
    print("\n" + "=" * 70)
    print("📊 Evaluating Models on Validation Set")
    print("=" * 70)
    
    # PyTorch evaluation
    print("\n   Evaluating PyTorch model...")
    pt_metrics = evaluate_pytorch(model, val_loader, device)
    
    # XGBoost evaluation
    print("   Evaluating XGBoost model...")
    xgb_preds = xgb_model.predict(X_val)
    xgb_metrics = {
        "rmse": rmse(xgb_preds, y_val),
        "r2": r2_score(y_val, xgb_preds),
        "mae": mean_absolute_error(y_val, xgb_preds),
        "predictions": xgb_preds,
        "targets": y_val
    }

    # ==========================================
    # PRINT COMPARISON TABLE
    # ==========================================
    print("\n" + "=" * 70)
    print("🏆 MODEL COMPARISON RESULTS (Validation Set)")
    print("=" * 70)
    
    print("\n" + "-" * 70)
    print(f"{'Model':<30} {'RMSE':>12} {'R²':>10} {'MAE':>12}")
    print("-" * 70)
    print(f"{'PyTorch (CNN + Tabular)':<30} {pt_metrics['rmse']:>10,.0f} {pt_metrics['r2']:>10.4f} {pt_metrics['mae']:>10,.0f}")
    print(f"{'XGBoost (on Deep Features)':<30} {xgb_metrics['rmse']:>10,.0f} {xgb_metrics['r2']:>10.4f} {xgb_metrics['mae']:>10,.0f}")
    print("-" * 70)
    
    # Declare winner
    if xgb_metrics['r2'] > pt_metrics['r2']:
        print(f"\n🏆 WINNER: XGBoost on Deep Features! (R²: {xgb_metrics['r2']:.4f})")
        winner = "xgboost"
    else:
        print(f"\n🏆 WINNER: PyTorch End-to-End Model! (R²: {pt_metrics['r2']:.4f})")
        winner = "pytorch"

    # ==========================================
    # GENERATE PREDICTIONS
    # ==========================================
    if args.predict:
        print("\n" + "=" * 70)
        print("📝 Generating Test Predictions")
        print("=" * 70)
        
        os.makedirs(cfg.output_dir, exist_ok=True)
        
        # Extract test features
        print("   Extracting test features...")
        X_test, test_ids = extract_features(model, te_loader, device, include_targets=False)
        
        # PyTorch predictions
        pt_preds = []
        model.eval()
        with torch.no_grad():
            for img, tab, pid in te_loader:
                img, tab = img.to(device), tab.to(device)
                pred = model(img, tab).cpu().numpy()
                pt_preds.extend(pred.tolist())
        
        # XGBoost predictions
        xgb_test_preds = xgb_model.predict(X_test)
        
        # Save predictions
        sub_pt = pd.DataFrame({"id": test_ids, "predicted_price": pt_preds})
        sub_pt.to_csv(os.path.join(cfg.output_dir, "submission_pytorch.csv"), index=False)
        print(f"   Saved: outputs/submission_pytorch.csv")
        
        sub_xgb = pd.DataFrame({"id": test_ids, "predicted_price": xgb_test_preds})
        sub_xgb.to_csv(os.path.join(cfg.output_dir, "submission_xgboost.csv"), index=False)
        print(f"   Saved: outputs/submission_xgboost.csv")
        
        # Save winner's submission
        if winner == "xgboost":
            sub_xgb.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
        else:
            sub_pt.to_csv(os.path.join(cfg.output_dir, "submission.csv"), index=False)
        print(f"   Saved: outputs/submission.csv (using {winner})")

    # ==========================================
    # DETAILED ANALYSIS
    # ==========================================
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    
    return {
        "pytorch": pt_metrics,
        "xgboost": xgb_metrics,
        "winner": winner
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PyTorch and XGBoost Models")
    parser.add_argument("--predict", action="store_true",
                        help="Generate predictions on test set")
    args = parser.parse_args()
    main(args)
