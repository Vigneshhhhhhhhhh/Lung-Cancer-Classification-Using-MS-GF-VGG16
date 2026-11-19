import os
import sys
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    DATA_DIR, BEST_MODEL_PATH, FINAL_MODEL_PATH, METRICS_PATH,
    HISTORY_PATH, ABLATION_PATH, PLOTS_DIR, BATCH_SIZE, INITIAL_LR,
    FINE_TUNE_LR, EPOCHS_STAGE1, EPOCHS_STAGE2, CLASS_NAMES
)
from backend.training.dataset import (
    scan_and_validate_dataset, load_dataset_arrays,
    create_augmented_train_pipeline, create_deterministic_pipeline
)
from backend.training.model import (
    build_ms_gf_vgg16, build_standard_vgg16, build_multiscale_vgg16_no_gating,
    ScaleGatedFeatureFusionUnit
)
from backend.training.evaluate import (
    calculate_detailed_metrics, generate_evaluation_plots, save_metrics_to_disk
)


def train_ms_gf_vgg16():
    """
    Executes the complete two-stage training, optimization, and evaluation
    pipeline for the proposed MS-GF-VGG16 architecture.
    """
    print("=" * 70)
    print("MS-GF-VGG16 (Multi-Scale Gated Feature Fusion VGG16) Training Pipeline")
    print("=" * 70)

    # 1. Scan and validate dataset
    print("\n[Step 1/6] Scanning and validating dataset integrity...")
    stats = scan_and_validate_dataset(DATA_DIR)
    print(f"Dataset location: {stats['path']}")
    print(f"Total valid images: {stats['totals']['valid_images']}, Corrupt images: {stats['totals']['corrupt_images']}")
    for split, sinfo in stats["splits"].items():
        print(f" - {split.upper()}: {sinfo['valid']} valid images | Class breakdown: {sinfo['classes']}")

    # 2. Load dataset into memory
    print("\n[Step 2/6] Loading image arrays and computing class weights...")
    (X_train, y_train, y_train_idx), (X_val, y_val, y_val_idx), (X_test, y_test, y_test_idx), class_weights = load_dataset_arrays(DATA_DIR)
    print(f"Train set: {X_train.shape[0]} images, Val set: {X_val.shape[0]} images, Test set: {X_test.shape[0]} images")
    print(f"Class weights: {class_weights}")

    # Build data pipelines
    train_ds = create_augmented_train_pipeline(X_train, y_train, batch_size=BATCH_SIZE)
    val_ds = create_deterministic_pipeline(X_val, y_val, batch_size=BATCH_SIZE)
    test_ds = create_deterministic_pipeline(X_test, y_test, batch_size=BATCH_SIZE)

    # 3. Instantiate Proposed Model
    print("\n[Step 3/6] Building MS-GF-VGG16 model architecture...")
    model = build_ms_gf_vgg16(input_shape=(128, 128, 3), num_classes=len(CLASS_NAMES))
    model.summary()

    # Stage 1: Train feature projection, SGFU, and classification head
    print("\n" + "=" * 50)
    print(f"STAGE 1: Feature Projection, SGFU & Head Training (Adam LR={INITIAL_LR})")
    print("=" * 50)

    optimizer_stage1 = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer_stage1,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage1 = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=1)
    ]

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE1,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Stage 2: Fine-tuning deeper layers of VGG16 (Block 4 and Block 5)
    print("\n" + "=" * 50)
    print(f"STAGE 2: Partial Fine-Tuning Deeper Layers (Adam LR={FINE_TUNE_LR})")
    print("=" * 50)

    # Find VGG16 layers and unfreeze block4 and block5 conv layers
    backbone_layer_names = ["block4_conv1", "block4_conv2", "block4_conv3", "block5_conv1", "block5_conv2", "block5_conv3"]
    for layer in model.layers:
        if layer.name in backbone_layer_names:
            layer.trainable = True

    optimizer_stage2 = tf.keras.optimizers.Adam(learning_rate=FINE_TUNE_LR)
    model.compile(
        optimizer=optimizer_stage2,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        ModelCheckpoint(
            filepath=str(BEST_MODEL_PATH),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_STAGE2,
        class_weight=class_weights,
        callbacks=callbacks_stage2,
        verbose=1
    )

    # Combine training histories
    combined_history = {
        "accuracy": history1.history["accuracy"] + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
        "loss": history1.history["loss"] + history2.history["loss"],
        "val_loss": history1.history["val_loss"] + history2.history["val_loss"]
    }

    # 4. Load the best saved model checkpoint
    print(f"\n[Step 4/6] Loading best checkpoint from {BEST_MODEL_PATH}...")
    best_model = tf.keras.models.load_model(
        str(BEST_MODEL_PATH),
        custom_objects={"ScaleGatedFeatureFusionUnit": ScaleGatedFeatureFusionUnit}
    )

    # 5. Rigorous untouched test set evaluation
    print("\n[Step 5/6] Evaluating best model on the untouched test set (315 images)...")
    test_preds_probs = best_model.predict(test_ds, verbose=1)
    test_preds_idx = np.argmax(test_preds_probs, axis=1)

    metrics = calculate_detailed_metrics(y_test_idx, test_preds_idx, test_preds_probs)
    print("\n" + "=" * 50)
    print("FINAL TEST SET EVALUATION RESULTS (MS-GF-VGG16):")
    print("=" * 50)
    print(f"Overall Accuracy:       {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro Precision:        {metrics['precision_macro']:.4f}")
    print(f"Macro Recall:           {metrics['recall_macro']:.4f}")
    print(f"Macro F1-Score:         {metrics['f1_macro']:.4f}")
    print(f"Weighted F1-Score:      {metrics['f1_weighted']:.4f}")
    print(f"Macro Specificity:      {metrics['specificity_macro']:.4f}")
    print(f"Macro ROC-AUC:          {metrics['roc_auc_macro']:.4f}")
    print("-" * 50)
    print("Per-Class Performance:")
    for cls, cmetrics in metrics["class_performance"].items():
        print(f"  {cmetrics['display_name']:<25}: Precision={cmetrics['precision']:.3f}, Recall={cmetrics['recall']:.3f}, F1={cmetrics['f1_score']:.3f}, Specificity={cmetrics['specificity']:.3f}")

    # Generate evaluation plots
    plots = generate_evaluation_plots(metrics, y_test_idx, test_preds_probs, combined_history)
    save_metrics_to_disk(metrics, combined_history)
    print(f"\nPlots saved to {PLOTS_DIR}")

    # 6. Run Ablation Study: Standard VGG16 vs MS-VGG16 (No Gating) vs MS-GF-VGG16
    print("\n[Step 6/6] Conducting Ablation Study Experiments...")
    ablation_results = run_ablation_study(
        train_ds, val_ds, test_ds, y_test_idx, class_weights,
        msgf_accuracy=metrics['accuracy'], msgf_f1=metrics['f1_macro']
    )

    return metrics, ablation_results


def run_ablation_study(train_ds, val_ds, test_ds, y_test_idx, class_weights, msgf_accuracy, msgf_f1):
    """
    Trains and benchmarks the baseline architectures to demonstrate the empirical
    advantage of the Scale-Gated Feature Fusion Unit (SGFU).
    """
    ablation_data = [
        {
            "experiment": "Exp 1: Standard VGG16",
            "description": "Baseline VGG16 with only block5 top features and dense head",
            "accuracy": None,
            "f1_macro": None
        },
        {
            "experiment": "Exp 2: MS-VGG16 (No Gating)",
            "description": "Multi-scale feature fusion using static uniform averaging",
            "accuracy": None,
            "f1_macro": None
        },
        {
            "experiment": "Exp 3: MS-GF-VGG16 (Proposed)",
            "description": "Multi-scale fusion with custom Scale-Gated Feature Fusion Unit (SGFU)",
            "accuracy": msgf_accuracy,
            "f1_macro": msgf_f1
        }
    ]

    # Train Exp 1 (Standard VGG16)
    print("\n--- Training Ablation Exp 1: Standard VGG16 ---")
    m1 = build_standard_vgg16(input_shape=(128, 128, 3), num_classes=len(CLASS_NAMES))
    m1.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    m1.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weights, verbose=0)
    p1 = m1.predict(test_ds, verbose=0)
    idx1 = np.argmax(p1, axis=1)
    # Accuracy and Macro F1
    acc1 = round(float(np.mean(y_test_idx == idx1)), 4)
    # macro f1
    f1s_1 = []
    for c in range(len(CLASS_NAMES)):
        tp = np.sum((y_test_idx == c) & (idx1 == c))
        fp = np.sum((y_test_idx != c) & (idx1 == c))
        fn = np.sum((y_test_idx == c) & (idx1 != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s_1.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    f1_1 = round(float(np.mean(f1s_1)), 4)
    ablation_data[0]["accuracy"] = acc1
    ablation_data[0]["f1_macro"] = f1_1
    print(f"Exp 1 Standard VGG16: Accuracy = {acc1*100:.2f}%, F1 = {f1_1:.4f}")

    # Train Exp 2 (MS-VGG16 No Gating)
    print("\n--- Training Ablation Exp 2: MS-VGG16 (Static Average) ---")
    m2 = build_multiscale_vgg16_no_gating(input_shape=(128, 128, 3), num_classes=len(CLASS_NAMES))
    m2.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
    m2.fit(train_ds, validation_data=val_ds, epochs=15, class_weight=class_weights, verbose=0)
    p2 = m2.predict(test_ds, verbose=0)
    idx2 = np.argmax(p2, axis=1)
    acc2 = round(float(np.mean(y_test_idx == idx2)), 4)
    f1s_2 = []
    for c in range(len(CLASS_NAMES)):
        tp = np.sum((y_test_idx == c) & (idx2 == c))
        fp = np.sum((y_test_idx != c) & (idx2 == c))
        fn = np.sum((y_test_idx == c) & (idx2 != c))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s_2.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    f1_2 = round(float(np.mean(f1s_2)), 4)
    ablation_data[1]["accuracy"] = acc2
    ablation_data[1]["f1_macro"] = f1_2
    ablation_data[1]["accuracy"] = acc2
    ablation_data[1]["f1_macro"] = f1_2
    print(f"Exp 2 MS-VGG16 (No Gating): Accuracy = {acc2*100:.2f}%, F1 = {f1_2:.4f}")
    print(f"Exp 3 MS-GF-VGG16 (Proposed): Accuracy = {msgf_accuracy*100:.2f}%, F1 = {msgf_f1:.4f}")

    with open(ABLATION_PATH, "w") as f:
        json.dump(ablation_data, f, indent=2)

    return ablation_data


if __name__ == "__main__":
    train_ms_gf_vgg16()
