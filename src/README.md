# 🔬 Source Scripts — `src/`

This folder contains the core ML scripts for training, evaluating, and running inference on the pneumonia detection CNN model.

---

## 📄 Scripts Overview

### `train.py` — Train the CNN Model
Trains a grayscale CNN on chest X-ray images from the Kaggle Chest X-Ray dataset.

**Requires dataset at:**
```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

**Run from project root:**
```bash
python src/train.py
```

**Output:**
- `models/pneumonia_model_best.h5` — Best checkpoint (by val_accuracy)
- `models/pneumonia_model.h5` — Final model after all epochs
- `models/training_history.png` — Accuracy / Loss / Precision-Recall plots

---

### `predict.py` — Run Inference from CLI
Loads `models/pneumonia_model_best.h5` and classifies a single X-ray image.

**Run from project root:**
```bash
python src/predict.py path/to/xray_image.jpg
```

**Output:**
```
Prediction: PNEUMONIA (Confidence: 91.34%)
```

---

### `evaluate.py` — Evaluate on Test Set
Runs the model against the full test set and prints a classification report, confusion matrix, and ROC-AUC score.

**Run from project root:**
```bash
python src/evaluate.py
```

**Output:**
- Console: Classification Report + AUC-ROC score
- `models/confusion_matrix.png`
- `models/roc_curve.png`

---

## ⚙️ Model Architecture

| Layer | Details |
|-------|---------|
| Input | 224×224×1 grayscale |
| Conv2D × 4 | 32 → 64 → 128 → 128 filters, ReLU + BatchNorm + MaxPool |
| Flatten + Dropout | 0.5 dropout |
| Dense | 512 units, ReLU, 0.3 dropout |
| Output | 1 unit, Sigmoid (binary: NORMAL / PNEUMONIA) |

- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam (lr=0.0001)  
- **Callbacks:** EarlyStopping (patience=5), ModelCheckpoint (best val_accuracy)

---

## ⚠️ Disclaimer
These scripts are for educational and research purposes only. Not intended for clinical use.