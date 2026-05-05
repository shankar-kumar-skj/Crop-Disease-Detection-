# 🌿 Crop Disease Detection
### Computer Vision with CNN + Transfer Learning | PlantVillage Dataset

---

## Overview

A deep learning pipeline for detecting crop diseases from leaf images using Convolutional Neural Networks and Transfer Learning (ResNet50, MobileNetV2). Built on the PlantVillage dataset with 54,303 images across 38 classes, 14 crop species, and 26 distinct diseases.

> **Dataset:** [PlantVillage on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)  
> 54,303 images · 38 classes · 14 crop species · 26 diseases

---

## Quick Start

### 1. Install Dependencies

```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-learn pillow kaggle
```

### 2. Download Dataset

```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/
```

### 3. Run the Pipeline (in order)

```bash
python week1_eda_preprocessing.py       # EDA + train/val/test splits
python week2_cnn_baseline.py            # Custom CNN training
python week3_transfer_learning.py       # ResNet50 + MobileNetV2
python week4_evaluation_inference.py    # Evaluation + reports
```

### 4. Run Inference on a New Image

```bash
python week4_evaluation_inference.py --infer path/to/your/leaf.jpg
```

---

## Project Structure

```
crop_disease_detection/
├── week1_eda_preprocessing.py      # EDA, augmentation, dataset splits
├── week2_cnn_baseline.py           # Custom CNN training
├── week3_transfer_learning.py      # Transfer learning + fine-tuning
├── week4_evaluation_inference.py   # Evaluation, confusion matrix, inference
│
├── data/
│   ├── plantvillage dataset/       # Raw dataset (after unzip)
│   └── split/
│       ├── train/                  # 70% per class
│       ├── val/                    # 15% per class
│       └── test/                   # 15% per class
│
├── models/
│   ├── custom_cnn_best.keras
│   ├── final_model_resnet50.keras
│   └── final_model_mobilenetv2.keras
│
├── plots/                          # All visualizations
├── reports/                        # Metrics + classification reports
└── logs/                           # Training history CSVs + TensorBoard logs
```

---

## Expected Results

| Stage | Model | Accuracy |
|-------|-------|----------|
| Week 2 | Custom CNN | ~75–82% |
| Week 3 | ResNet50 (Transfer Learning) | ~92–97% |
| Week 3 | MobileNetV2 (Transfer Learning) | ~88–94% |

---

## Key Metric: Recall

**Recall is the priority metric** — a missed disease (false negative) allows infection to spread across the entire field before detection. The model is optimized to minimize false negatives over false positives.

---

## Hardware Recommendations

| Setup | Notes |
|-------|-------|
| **GPU (recommended)** | Full batch size and epoch count |
| **CPU only** | Reduce batch size to `16`, epochs to `10` for testing |
| **Google Colab** | Free T4 GPU available — recommended if no local GPU |

---

## Week-by-Week Breakdown

| Script | What it does |
|--------|-------------|
| `week1_eda_preprocessing.py` | Exploratory data analysis, image augmentation, stratified train/val/test splits |
| `week2_cnn_baseline.py` | Trains a custom CNN from scratch as a performance baseline |