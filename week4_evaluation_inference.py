"""
Week 4: Evaluation, Confusion Matrix, and Inference Script
- Full evaluation on held-out test set
- Per-class precision / recall / F1
- Confusion matrix visualization
- Standalone inference function for production use
Run after week3_transfer_learning.py
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from PIL import Image

# ─── Configuration ────────────────────────────────────────────────────────────
SPLIT_DIR = Path("data/split")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
MODEL_PATH = "models/final_model_resnet50.keras"   # switch to mobilenetv2 if preferred

with open("dataset_metadata.json") as f:
    META = json.load(f)
NUM_CLASSES = META["num_classes"]
CLASS_NAMES = META["classes"]

Path("plots").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)


# ─── 1. Load model & test generator ──────────────────────────────────────────
def load_model_and_data():
    print(f"\n  Loading model: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)

    test_gen = ImageDataGenerator(rescale=1.0 / 255)
    test_ds = test_gen.flow_from_directory(
        SPLIT_DIR / "test",
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )
    return model, test_ds


# ─── 2. Get predictions ───────────────────────────────────────────────────────
def get_predictions(model, test_ds):
    print("  Running predictions on test set...")
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = test_ds.classes
    return y_true, y_pred, y_probs


# ─── 3. Overall metrics ───────────────────────────────────────────────────────
def print_overall_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n{'='*50}")
    print(f"  FINAL TEST SET METRICS")
    print(f"{'='*50}")
    print(f"  Accuracy:           {acc*100:.2f}%")
    print(f"  Macro Precision:    {prec*100:.2f}%")
    print(f"  Macro Recall:       {rec*100:.2f}%   ← critical metric")
    print(f"  Macro F1-Score:     {f1*100:.2f}%")
    print(f"{'='*50}")

    results = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open("reports/final_metrics.json", "w") as f:
        json.dump({k: round(v, 4) for k, v in results.items()}, f, indent=2)
    return results


# ─── 4. Per-class classification report ──────────────────────────────────────
def save_classification_report(y_true, y_pred):
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print("\n  Per-class Classification Report:")
    print(report_text)

    with open("reports/classification_report.txt", "w") as f:
        f.write(report_text)

    # Plot per-class recall (critical for disease detection)
    recalls = [report[c]["recall"] for c in CLASS_NAMES if c in report]
    fig, ax = plt.subplots(figsize=(14, 9))
    colors = ["#e74c3c" if r < 0.85 else "#2ecc71" for r in recalls]
    bars = ax.barh(CLASS_NAMES, recalls, color=colors, edgecolor="white", height=0.7)
    ax.axvline(0.85, color="black", linestyle="--", linewidth=1.5, label="0.85 threshold")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_title("Per-class Recall — Test Set\n(Red = below 0.85 threshold)", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/per_class_recall.png", dpi=150, bbox_inches="tight")
    plt.show()


# ─── 5. Confusion matrix ──────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, normalize=True):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (normalized)"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix (counts)"

    n = len(CLASS_NAMES)
    fig_size = max(16, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))

    sns.heatmap(
        cm_plot,
        annot=(n <= 38),   # skip numbers for huge matrices
        fmt=fmt,
        cmap="YlOrRd",
        ax=ax,
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        linewidths=0.3,
        cbar_kws={"shrink": 0.6}
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", labelsize=6, rotation=90)
    ax.tick_params(axis="y", labelsize=6, rotation=0)
    plt.tight_layout()
    plt.savefig(f"plots/confusion_matrix{'_norm' if normalize else ''}.png",
                dpi=150, bbox_inches="tight")
    plt.show()

    # Also find the most confused pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    top_k = 10
    flat_idx = np.argsort(cm_copy.ravel())[-top_k:][::-1]
    print(f"\n  Top {top_k} most confused class pairs:")
    for idx in flat_idx:
        r, c = divmod(idx, n)
        if cm_copy[r, c] > 0:
            print(f"    True: {CLASS_NAMES[r]:<45}  → Pred: {CLASS_NAMES[c]:<45}  ({cm_copy[r,c]} cases)")


# ─── 6. Misclassification gallery ────────────────────────────────────────────
def plot_misclassifications(test_ds, y_true, y_pred, y_probs, n=12):
    wrong_idx = np.where(y_true != y_pred)[0]
    sample_idx = wrong_idx[:n]

    all_paths = [Path(p) for p in test_ds.filepaths]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.ravel()

    for i, idx in enumerate(sample_idx):
        img = Image.open(all_paths[idx]).convert("RGB")
        conf = y_probs[idx][y_pred[idx]] * 100
        axes[i].imshow(img)
        axes[i].axis("off")
        true_name = CLASS_NAMES[y_true[idx]].split("___")[-1].replace("_", " ")
        pred_name = CLASS_NAMES[y_pred[idx]].split("___")[-1].replace("_", " ")
        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name} ({conf:.1f}%)",
            fontsize=8, color="#c0392b"
        )

    plt.suptitle("Misclassified Samples", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plots/misclassifications.png", dpi=120, bbox_inches="tight")
    plt.show()


# ─── 7. Inference script ─────────────────────────────────────────────────────
class CropDiseasePredictor:
    """
    Production inference class.
    Usage:
        predictor = CropDiseasePredictor("models/final_model_resnet50.keras")
        result = predictor.predict("path/to/leaf.jpg")
        print(result)
    """

    TREATMENT_MAP = {
        "healthy": "No treatment needed. Continue regular care.",
        "Early_blight": "Apply copper-based fungicide. Remove infected leaves.",
        "Late_blight": "Use mancozeb or chlorothalonil. Improve drainage.",
        "Leaf_Mold": "Improve air circulation. Apply fungicide if severe.",
        "Septoria_leaf_spot": "Remove infected foliage. Apply fungicide.",
        "Spider_mites": "Apply miticide or neem oil.",
        "Target_Spot": "Apply strobilurin fungicide. Rotate crops.",
        "Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Remove infected plants.",
        "mosaic_virus": "Remove infected plants. Control aphid vectors.",
        "Bacterial_spot": "Use copper bactericide. Avoid overhead watering.",
        "Black_rot": "Prune infected areas. Apply copper fungicide.",
        "Cedar_apple_rust": "Apply myclobutanil fungicide preventively.",
        "Powdery_mildew": "Apply sulfur or potassium bicarbonate.",
        "Leaf_scorch": "Check irrigation. Reduce fertilizer.",
        "Haunglongbing": "Remove infected trees. Control psyllid vectors.",
        "Common_rust": "Apply fungicide containing azoxystrobin.",
        "Northern_Leaf_Blight": "Apply fungicide at tassel stage.",
        "Gray_leaf_spot": "Crop rotation. Foliar fungicide application.",
        "Esca": "Remove infected wood. Prune in dry weather.",
        "Leaf_blight": "Apply copper-based bactericide.",
    }

    def __init__(self, model_path: str, metadata_path: str = "dataset_metadata.json"):
        self.model = tf.keras.models.load_model(model_path)
        with open(metadata_path) as f:
            meta = json.load(f)
        self.class_names = meta["classes"]
        self.img_size = tuple(meta["img_size"])

    def _preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.img_size, Image.LANCZOS)
        arr = np.array(img) / 255.0
        return np.expand_dims(arr, axis=0)

    def _get_treatment(self, class_name: str) -> str:
        for keyword, treatment in self.TREATMENT_MAP.items():
            if keyword.lower() in class_name.lower():
                return treatment
        return "Consult a local agronomist for targeted treatment."

    def predict(self, image_path: str, top_k: int = 3) -> dict:
        arr = self._preprocess(image_path)
        probs = self.model.predict(arr, verbose=0)[0]
        top_indices = np.argsort(probs)[-top_k:][::-1]

        predicted_class = self.class_names[top_indices[0]]
        confidence = float(probs[top_indices[0]])
        is_healthy = "healthy" in predicted_class.lower()

        parts = predicted_class.split("___")
        crop = parts[0].replace("_", " ").title() if len(parts) > 0 else "Unknown"
        disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"

        return {
            "image_path": str(image_path),
            "predicted_class": predicted_class,
            "crop": crop,
            "disease": disease,
            "is_healthy": is_healthy,
            "confidence": round(confidence * 100, 2),
            "treatment": self._get_treatment(predicted_class),
            "top_predictions": [
                {
                    "class": self.class_names[i],
                    "confidence": round(float(probs[i]) * 100, 2)
                }
                for i in top_indices
            ]
        }

    def predict_batch(self, image_paths: list) -> list:
        return [self.predict(p) for p in image_paths]

    def visualize_prediction(self, image_path: str):
        result = self.predict(image_path)
        img = Image.open(image_path).convert("RGB")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(img)
        axes[0].axis("off")
        color = "#27ae60" if result["is_healthy"] else "#e74c3c"
        status = "HEALTHY" if result["is_healthy"] else "DISEASED"
        axes[0].set_title(
            f"{status} — {result['confidence']}% confidence",
            fontsize=12, fontweight="bold", color=color
        )

        classes = [p["class"].split("___")[-1][:20] for p in result["top_predictions"]]
        confs = [p["confidence"] for p in result["top_predictions"]]
        bar_colors = [color] + ["#95a5a6"] * (len(classes) - 1)
        axes[1].barh(classes[::-1], confs[::-1], color=bar_colors[::-1])
        axes[1].set_xlabel("Confidence (%)")
        axes[1].set_title("Top Predictions")
        axes[1].set_xlim(0, 100)

        plt.suptitle(
            f"Crop: {result['crop']}  |  Disease: {result['disease']}\n"
            f"Treatment: {result['treatment']}",
            fontsize=10, style="italic"
        )
        plt.tight_layout()
        plt.savefig("plots/inference_result.png", dpi=150, bbox_inches="tight")
        plt.show()
        return result


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop Disease Detector — Week 4")
    parser.add_argument("--infer", type=str, default=None,
                        help="Path to a single leaf image for inference")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to saved .keras model")
    args = parser.parse_args()

    if args.infer:
        # ── Inference mode ─────────────────────────────────────────────────────
        print(f"\n  Running inference on: {args.infer}")
        predictor = CropDiseasePredictor(args.model)
        result = predictor.visualize_prediction(args.infer)
        print("\n  ╔══════════════════════════════════════╗")
        print(f"  ║  Crop:       {result['crop']:<24}║")
        print(f"  ║  Disease:    {result['disease']:<24}║")
        print(f"  ║  Confidence: {result['confidence']:>5.1f}%                  ║")
        print(f"  ║  Status:     {'HEALTHY' if result['is_healthy'] else 'DISEASED':<24}║")
        print("  ╠══════════════════════════════════════╣")
        print(f"  ║  Treatment: {result['treatment'][:37]}")
        print("  ╚══════════════════════════════════════╝")

    else:
        # ── Full evaluation mode ────────────────────────────────────────────────
        model, test_ds = load_model_and_data()
        y_true, y_pred, y_probs = get_predictions(model, test_ds)

        print_overall_metrics(y_true, y_pred)
        save_classification_report(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred, normalize=True)
        plot_confusion_matrix(y_true, y_pred, normalize=False)
        plot_misclassifications(test_ds, y_true, y_pred, y_probs)

        print("\n  All reports saved to reports/ and plots/")
        print("\n  To run inference on a new leaf image:")
        print("    python week4_evaluation_inference.py --infer path/to/leaf.jpg")
        print("\n  Week 4 complete. Project done!")
