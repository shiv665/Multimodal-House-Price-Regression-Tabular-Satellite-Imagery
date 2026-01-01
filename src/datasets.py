import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from src.config import cfg

class HouseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_paths, scaler=None, train=True):
        self.df = df.reset_index(drop=True)
        self.paths = img_paths
        self.train = train
        self.scaler = scaler
        self.tab = df[cfg.tab_feats].astype(float).values
        if self.scaler:
            self.tab = self.scaler.transform(self.tab)
        self.y = df[cfg.target].values if train else None
        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess_image(self, img):
        """Resize, normalize and convert to tensor using OpenCV."""
        # Resize using OpenCV
        img = cv2.resize(img, (cfg.img_size, cfg.img_size), interpolation=cv2.INTER_LINEAR)
        # Convert to float and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # Apply ImageNet normalization
        img = (img - self.mean) / self.std
        # Convert HWC to CHW format for PyTorch
        img = np.transpose(img, (2, 0, 1)) #because PyTorch expects channel first
        return torch.tensor(img, dtype=torch.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx] #here df.iloc is used for precise extraction of data
        pid = row["id"]
        img_path = self.paths.get(pid, None)
        if img_path and os.path.exists(img_path):
            # OpenCV reads as BGR, convert to RGB
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
            else:
                img = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
        else:
            img = np.zeros((cfg.img_size, cfg.img_size, 3), dtype=np.uint8)
        img = self._preprocess_image(img)
        tab = torch.tensor(self.tab[idx], dtype=torch.float32)
        if self.train:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return img, tab, y
        return img, tab, pid
    # so what this getitem is doing is that for each index it retrieves the corresponding row from the dataframe,
    # gets the image path using the id, reads and preprocesses the image, retrieves the tabular data, and returns them as tensors.