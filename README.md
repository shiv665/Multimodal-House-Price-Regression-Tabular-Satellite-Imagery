# 🏠 House Price Prediction using Satellite Imagery & Tabular Data

A hybrid deep learning pipeline that combines **CNN-based satellite image analysis** with **tabular features** to predict house prices, achieving **R² of 0.87**.

## 🌟 Key Features

- **Hybrid Architecture**: ResNet18 CNN + Tabular MLP fusion model
- **XGBoost Enhancement**: Train XGBoost on deep features for potential performance boost
- **Free Satellite Imagery**: Uses ESRI World Imagery (no API key required)
- **Explainability**: Grad-CAM visualizations to understand model predictions
- **Automatic Model Selection**: Picks the best model (PyTorch vs XGBoost) based on R² score

## 📁 Project Structure

```
Satellite_Imagery/
├── data/
│   ├── train(1).xlsx       # Training data with prices
│   ├── test2.xlsx          # Test data for predictions
│   └── satellite/          # Downloaded satellite images
├── models/
│   ├── best_model.pt       # Best PyTorch model checkpoint
│   ├── last_model.pt       # Latest checkpoint
│   └── hybrid_best_model.pt
├── notebooks/
│   ├── preprocessing.ipynb # Data exploration & EDA
│   └── model_training.ipynb # Interactive training notebook
├── outputs/
│   ├── submission.csv      # Final predictions
│   └── gradcam/            # Grad-CAM visualizations
├── src/
│   ├── config.py           # Configuration settings
│   ├── data_fetcher.py     # Satellite image downloader
│   ├── datasets.py         # PyTorch dataset classes
│   ├── model.py            # HybridMultimodalModel architecture
│   ├── train.py            # Main training pipeline
│   ├── compare.py          # Load & compare saved models
│   ├── gradcam.py          # Grad-CAM implementation
│   └── utils.py            # Utility functions
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Satellite-Imagery.git
cd Satellite-Imagery/Satellite_Imagery
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Data

Place your data files in the `data/` directory:
- `train(1).xlsx` - Training data with house prices
- `test2.xlsx` - Test data for predictions

### 5. Download Satellite Images (Automatic)

Images are downloaded automatically when you run training. No API key needed!

Or download manually:
```bash
python -m src.data_fetcher
```

### 6. Train the Model

**Full Training Pipeline:**
```bash
python -m src.train
```

**With Custom Epochs:**
```bash
python -m src.train --epochs 20
```

### 7. Compare Saved Models

Load a saved model and compare PyTorch vs XGBoost:
```bash
python -m src.compare --predict
```

## 📊 Model Architecture

### HybridMultimodalModel
```
┌─────────────────┐     ┌─────────────────┐
│  Satellite      │     │    Tabular      │
│    Image        │     │   Features      │
│  (224×224×3)    │     │    (128 dim)    │
└────────┬────────┘     └────────┬────────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │ ResNet18│            │  MLP      │
    │ Encoder │            │ 128→64    │
    └────┬────┘            └─────┬─────┘
         │                       │
    ┌────▼────┐            ┌─────▼─────┐
    │  256    │            │    64     │
    │ features│            │ features  │
    └────┬────┘            └─────┬─────┘
         │                       │
         └───────────┬───────────┘
                     │
              ┌──────▼──────┐
              │  Concatenate │
              │  320 features│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  Regressor  │
              │ 320→128→64→1│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ House Price │
              └─────────────┘
```

## 🔧 Configuration

Edit `src/config.py` to customize:

```python
class Config:
    # Data paths
    train_xlsx = "data/train(1).xlsx"
    test_xlsx = "data/test2.xlsx"
    image_dir = "data/satellite"
    
    # Training
    epochs = 15
    batch_size = 32
    lr = 1e-4
    
    # Model
    val_split = 0.2
    seed = 42
```

## 📈 Results

| Model | RMSE | R² | MAE |
|-------|------|-----|-----|
| PyTorch (CNN + Tabular) | ~$45,000 | ~0.85 | ~$32,000 |
| XGBoost (on Deep Features) | ~$42,000 | **~0.87** | ~$30,000 |

## 🔥 Grad-CAM Visualizations

The model generates Grad-CAM heatmaps showing which parts of satellite images influence predictions:

![Grad-CAM Example](outputs/gradcam/sample_0_gt500000.png)

## 📦 Output Files

After training:
- `outputs/submission.csv` - Predictions using the best model
- `outputs/gradcam/*.png` - Grad-CAM visualizations
- `models/best_model.pt` - Best model checkpoint

## 🛠️ Troubleshooting

### CUDA Out of Memory
Reduce batch size in `src/config.py`:
```python
batch_size = 16  # or 8
```

### Missing Images
Run the data fetcher:
```bash
python -m src.data_fetcher
```

### Import Errors
Make sure you're in the `Satellite_Imagery` directory:
```bash
cd Satellite_Imagery
python -m src.train
```

## 📚 References

- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

## 📄 License

MIT License

## 👤 Author

Shivansh Yadav

---

⭐ **Star this repo if you find it useful!**


