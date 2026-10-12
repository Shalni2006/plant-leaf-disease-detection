"""
Evaluation Script for Plant Leaf Disease Detection
Evaluates the trained model strictly on the held-out test set:
- Test Loss & Test Accuracy
- Per-Class Classification Report (Precision, Recall, F1-Score)
- Confusion Matrix Heatmap (Saved as model/confusion_matrix.png)
- Structured evaluation metrics saved as model/evaluation_metrics.json
"""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt

import model_config as config
from dataset_loader import build_stratified_split_paths, create_tf_dataset, load_class_indices

def plot_and_save_confusion_matrix(cm, class_names, output_path=config.CONFUSION_MATRIX_PATH):
    """
    Renders and saves a readable confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Greens)
    ax.figure.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Plant Leaf Disease Detection - Confusion Matrix",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=9)
    plt.setp(ax.get_yticklabels(), fontsize=9)

    # Annotate cells with counts
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[OK] Confusion matrix heatmap saved to: {output_path}")

def evaluate():
    print("=" * 65)
    print("      HELD-OUT TEST SET MODEL EVALUATION")
    print("=" * 65)

    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {config.MODEL_PATH}. Please train the model first."
        )

    # 1. Load Model & Class Names
    print(f"Loading trained model from: {config.MODEL_PATH}")
    model = tf.keras.models.load_model(config.MODEL_PATH)

    class_names = load_class_indices()
    print(f"Loaded {len(class_names)} classes.")

    # 2. Load the isolated Test Set
    split_data = build_stratified_split_paths()
    test_paths, test_labels = split_data["test"]
    print(f"Test Set Size: {len(test_paths)} images across {len(class_names)} classes.")

    test_ds = create_tf_dataset(test_paths, test_labels, is_training=False, batch_size=config.BATCH_SIZE)

    # 3. Compute Measured Scalar Evaluation
    print("\nEvaluating test set performance...")
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\n-> Measured Test Loss     : {test_loss:.4f}")
    print(f"-> Measured Test Accuracy : {test_acc * 100:.2f}%")

    # 4. Predict on Test Set
    print("\nGenerating predictions for classification report and confusion matrix...")
    y_pred_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.array(test_labels)

    # 5. Classification Report
    print("\n" + "=" * 65)
    print("              PER-CLASS CLASSIFICATION REPORT")
    print("=" * 65)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report_str)

    # 6. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_and_save_confusion_matrix(cm, class_names)

    # 7. Save Metrics JSON
    metrics_summary = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_test_samples": len(test_labels),
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist()
    }
    with open(config.EVALUATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[OK] Evaluation metrics saved to: {config.EVALUATION_METRICS_PATH}")
    print("=" * 65)

if __name__ == "__main__":
    evaluate()
