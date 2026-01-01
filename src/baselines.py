import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.config import cfg
import math

def run_tabular_baseline():
    df = pd.read_excel(cfg.train_xlsx)
    X = df[cfg.tab_feats].values
    y = df[cfg.target].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=cfg.val_split, random_state=cfg.seed)
    model = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
    )
    model.fit(Xtr, ytr)
    p = model.predict(Xva)
    rmse = math.sqrt(mean_squared_error(yva, p))
    r2 = r2_score(yva, p)
    print(f"Tabular-only XGB RMSE: {rmse:.2f} | R2: {r2:.3f}")
    return model, scaler

# print(run_tabular_baseline())
# 120484.77 | R2: 0.879 this is my baseline score on tabular data only
# Now I can try to improve it using satellite images, if that provide me better accuracy and 
# reduce the error I can be more confident about my model.