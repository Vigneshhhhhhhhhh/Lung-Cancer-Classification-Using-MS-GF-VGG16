import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from backend.config import CLASS_NAMES, DISPLAY_NAMES, RESULTS_DIR, PLOTS_DIR, METRICS_PATH


def compute_roc_curve(y_true_binary, y_scores):
    """
    Computes false positive rates, true positive rates, and AUC using pure NumPy.
    """
    desc_score_indices = np.argsort(y_scores, kind="mergesort")[::-1]
    y_scores = y_scores[desc_score_indices]
    y_true_binary = y_true_binary[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_scores))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true_binary.size - 1]

    tps = np.cumsum(y_true_binary)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add zero point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0:
        fpr = np.zeros_like(fps, dtype=float)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.zeros_like(tps, dtype=float)
    else:
        tpr = tps / tps[-1]

    # Trapezoidal rule AUC
    auc_val = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, max(0.0, min(1.0, auc_val))


def calculate_detailed_metrics(y_true_indices, y_pred_indices, y_pred_probs):
    """
    Computes all standard and per-class clinical evaluation metrics using pure NumPy.
    """
    n_classes = len(CLASS_NAMES)
    total_samples = len(y_true_indices)

    # 1. Confusion Matrix (n_classes x n_classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true_indices, y_pred_indices):
        cm[t, p] += 1

    # 2. Per-class metrics
    per_class_precision = []
    per_class_recall = []
    per_class_f1 = []
    per_class_specificity = {}
    supports = []

    for idx, cls in enumerate(CLASS_NAMES):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        fp = np.sum(cm[:, idx]) - tp
        tn = total_samples - (tp + fn + fp)

        p = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        r = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        per_class_precision.append(p)
        per_class_recall.append(r)
        per_class_f1.append(f1)
        per_class_specificity[cls] = round(spec, 4)
        supports.append(int(tp + fn))

    # 3. Overall / Macro / Weighted metrics
    accuracy = float(np.mean(y_true_indices == y_pred_indices))
    macro_precision = float(np.mean(per_class_precision))
    macro_recall = float(np.mean(per_class_recall))
    macro_f1 = float(np.mean(per_class_f1))
    macro_specificity = float(np.mean(list(per_class_specificity.values())))

    total_support = float(np.sum(supports))
    weighted_precision = float(np.sum(np.array(per_class_precision) * np.array(supports)) / total_support)
    weighted_recall = float(np.sum(np.array(per_class_recall) * np.array(supports)) / total_support)
    weighted_f1 = float(np.sum(np.array(per_class_f1) * np.array(supports)) / total_support)

    # 4. ROC-AUC One-vs-Rest
    y_true_onehot = np.eye(n_classes)[y_true_indices]
    per_class_roc_auc = {}
    roc_curves = {}

    for idx, cls in enumerate(CLASS_NAMES):
        fpr, tpr, roc_val = compute_roc_curve(y_true_onehot[:, idx], y_pred_probs[:, idx])
        per_class_roc_auc[cls] = round(roc_val, 4)
        roc_curves[cls] = (fpr.tolist(), tpr.tolist(), round(roc_val, 4))

    macro_roc_auc = float(np.mean(list(per_class_roc_auc.values())))
    weighted_roc_auc = float(np.sum(np.array(list(per_class_roc_auc.values())) * np.array(supports)) / total_support)

    # 5. Class breakdown structure
    class_performance = {}
    for idx, cls in enumerate(CLASS_NAMES):
        class_performance[cls] = {
            "display_name": DISPLAY_NAMES[cls],
            "precision": round(per_class_precision[idx], 4),
            "recall": round(per_class_recall[idx], 4),
            "f1_score": round(per_class_f1[idx], 4),
            "specificity": per_class_specificity[cls],
            "roc_auc": per_class_roc_auc[cls],
            "support": supports[idx]
        }

    overall_metrics = {
        "accuracy": round(accuracy, 4),
        "precision_macro": round(macro_precision, 4),
        "precision_weighted": round(weighted_precision, 4),
        "recall_macro": round(macro_recall, 4),
        "recall_weighted": round(weighted_recall, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(weighted_f1, 4),
        "specificity_macro": round(macro_specificity, 4),
        "roc_auc_macro": round(macro_roc_auc, 4),
        "roc_auc_weighted": round(weighted_roc_auc, 4),
        "total_test_samples": total_samples,
        "confusion_matrix": cm.tolist(),
        "class_labels": [DISPLAY_NAMES[c] for c in CLASS_NAMES],
        "class_performance": class_performance,
        "roc_curves": roc_curves
    }

    return overall_metrics


def generate_evaluation_plots(metrics, y_true_indices, y_pred_probs, history=None):
    """
    Generates and saves professional evaluation figures:
    1. Confusion Matrix Heatmap
    2. ROC Curves (One-vs-Rest)
    3. Training & Validation loss/accuracy curves
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    class_labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    cm = np.array(metrics["confusion_matrix"])

    # 1. Confusion Matrix Heatmap
    plt.figure(figsize=(8, 6), dpi=150)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_labels, yticklabels=class_labels,
        cbar=True, annot_kws={"size": 12, "weight": "bold"}
    )
    plt.title(f"MS-GF-VGG16 Test Confusion Matrix (Accuracy: {metrics['accuracy']*100:.2f}%)", fontsize=13, pad=15)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10)
    plt.ylabel("True Class", fontsize=11, labelpad=10)
    plt.tight_layout()
    cm_path = PLOTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # 2. Per-class ROC Curves
    plt.figure(figsize=(8, 6), dpi=150)
    colors = ["#2563eb", "#059669", "#d97706", "#7c3aed"]

    for idx, (cls, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, roc_val = metrics["roc_curves"][cls]
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{DISPLAY_NAMES[cls]} (AUC = {roc_val:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("Multi-Class ROC Curves (One-vs-Rest)", fontsize=13, pad=15)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    roc_path = PLOTS_DIR / "roc_curves.png"
    plt.savefig(roc_path)
    plt.close()

    # 3. Training & Validation Curves (if history provided)
    if history and "accuracy" in history and "val_accuracy" in history:
        epochs = range(1, len(history["accuracy"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Accuracy
        ax1.plot(epochs, history["accuracy"], 'b-o', markersize=3, label="Train Accuracy", color="#2563eb")
        ax1.plot(epochs, history["val_accuracy"], 'r-s', markersize=3, label="Val Accuracy", color="#059669")
        ax1.set_title("Training & Validation Accuracy", fontsize=12, pad=10)
        ax1.set_xlabel("Epoch", fontsize=10)
        ax1.set_ylabel("Accuracy", fontsize=10)
        ax1.legend()
        ax1.grid(alpha=0.3)

        # Loss
        ax2.plot(epochs, history["loss"], 'b-o', markersize=3, label="Train Loss", color="#2563eb")
        ax2.plot(epochs, history["val_loss"], 'r-s', markersize=3, label="Val Loss", color="#dc2626")
        ax2.set_title("Training & Validation Loss", fontsize=12, pad=10)
        ax2.set_xlabel("Epoch", fontsize=10)
        ax2.set_ylabel("Categorical Crossentropy Loss", fontsize=10)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        hist_path = PLOTS_DIR / "training_curves.png"
        plt.savefig(hist_path)
        plt.close()

    return {
        "confusion_matrix_plot": str(cm_path.name),
        "roc_curves_plot": str(roc_path.name)
    }


def save_metrics_to_disk(metrics_dict, history_dict=None):
    """Saves metrics and history to JSON in backend/results."""
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    if history_dict is not None:
        from backend.config import HISTORY_PATH
        with open(HISTORY_PATH, "w") as f:
            json.dump(history_dict, f, indent=2)
