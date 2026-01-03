import os
from dataclasses import dataclass, field
from dotenv import load_dotenv
import torch

# Load environment variables from .env file
load_dotenv()

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    # data
    train_xlsx: str = "data/train(1).xlsx"
    test_xlsx: str = "data/test2.xlsx"
    image_dir: str = "data/satellite"
    output_dir: str = "outputs"
    model_dir: str = "models"
    model_test: str = "saved_model_of_different_epoch"
    # ESRI World Imagery (FREE - no API key needed)
    zoom: int = 18
    tile_size: int = 256
    

    # modeling
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 15
    seed: int = 42
    val_split: float = 0.15
    grad_cam_samples: int = 500
    device: str = field(default_factory=get_device)

    # tabular features
    tab_feats = [
        "bedrooms","bathrooms","sqft_living","sqft_lot","floors","waterfront","view",
        "condition","grade","sqft_above","sqft_basement","yr_built","yr_renovated",
        "zipcode","sqft_living15","sqft_lot15","lat","long"
    ]
    target: str = "price"

cfg = Config()