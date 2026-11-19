import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from backend.config import CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE, DATA_DIR


def resolve_canonical_class(folder_name: str) -> str:
    """
    Maps folder names with spaces, periods, or suffixes into canonical class names.
    Examples:
      - 'adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib' -> 'adenocarcinoma'
      - 'large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa' -> 'large_cell_carcinoma'
      - 'large.cell.carcinoma' -> 'large_cell_carcinoma'
      - 'normal' -> 'normal'
      - 'squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa' -> 'squamous_cell_carcinoma'
      - 'squamous.cell.carcinoma' -> 'squamous_cell_carcinoma'
    """
    clean = folder_name.lower().replace("-", "_").replace(".", "_").replace(" ", "_")
    if "adeno" in clean:
        return "adenocarcinoma"
    elif "large" in clean:
        return "large_cell_carcinoma"
    elif "squamous" in clean:
        return "squamous_cell_carcinoma"
    elif "normal" in clean:
        return "normal"
    else:
        raise ValueError(f"Unrecognized class folder: {folder_name}")


def scan_and_validate_dataset(data_dir: Path = DATA_DIR):
    """
    Scans the dataset directory, validates image integrity,
    and returns detailed statistics and validation status.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found at: {data_dir.resolve()}")

    stats = {
        "exists": True,
        "path": str(data_dir.resolve()),
        "splits": {},
        "totals": {
            "total_images": 0,
            "valid_images": 0,
            "corrupt_images": 0,
            "by_class": {cls: 0 for cls in CLASS_NAMES}
        },
        "class_imbalance": {}
    }

    supported_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    for split in ["train", "valid", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing required split directory: {split_dir}")

        stats["splits"][split] = {
            "total": 0,
            "valid": 0,
            "corrupt": 0,
            "classes": {cls: 0 for cls in CLASS_NAMES},
            "folder_mapping": {}
        }

        # Inspect subdirectories in split
        for sub_dir in sorted(split_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            canonical = resolve_canonical_class(sub_dir.name)
            stats["splits"][split]["folder_mapping"][sub_dir.name] = canonical

            for file_path in sub_dir.iterdir():
                if file_path.suffix in supported_exts:
                    stats["totals"]["total_images"] += 1
                    stats["splits"][split]["total"] += 1
                    try:
                        with Image.open(file_path) as img:
                            img.verify()
                        stats["splits"][split]["valid"] += 1
                        stats["splits"][split]["classes"][canonical] += 1
                        stats["totals"]["valid_images"] += 1
                        stats["totals"]["by_class"][canonical] += 1
                    except Exception as e:
                        stats["splits"][split]["corrupt"] += 1
                        stats["totals"]["corrupt_images"] += 1

    # Calculate class distribution percentages
    total_valid = stats["totals"]["valid_images"]
    if total_valid > 0:
        for cls, count in stats["totals"]["by_class"].items():
            stats["class_imbalance"][cls] = {
                "count": count,
                "percentage": round((count / total_valid) * 100, 2),
                "display_name": DISPLAY_NAMES[cls]
            }

    return stats


def load_dataset_arrays(data_dir: Path = DATA_DIR, target_size=IMAGE_SIZE):
    """
    Loads images for train, valid, and test sets into raw uint8 NumPy arrays [0, 255].
    Returns (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights.
    """
    supported_exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}

    splits_data = {}

    for split in ["train", "valid", "test"]:
        split_dir = Path(data_dir) / split
        images = []
        labels = []

        for sub_dir in sorted(split_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            canonical = resolve_canonical_class(sub_dir.name)
            label_idx = class_to_idx[canonical]

            for f in sorted(sub_dir.iterdir()):
                if f.suffix in supported_exts:
                    try:
                        with Image.open(f) as img:
                            img_rgb = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
                            arr = np.array(img_rgb, dtype=np.uint8)
                            images.append(arr)
                            labels.append(label_idx)
                    except Exception as e:
                        print(f"Skipping unreadable image {f}: {e}")

        splits_data[split] = (
            np.array(images, dtype=np.float32),
            np.array(labels, dtype=np.int32)
        )

    X_train, y_train_int = splits_data["train"]
    X_val, y_val_int = splits_data["valid"]
    X_test, y_test_int = splits_data["test"]

    # One-hot encode targets
    y_train = tf.keras.utils.to_categorical(y_train_int, num_classes=len(CLASS_NAMES))
    y_val = tf.keras.utils.to_categorical(y_val_int, num_classes=len(CLASS_NAMES))
    y_test = tf.keras.utils.to_categorical(y_test_int, num_classes=len(CLASS_NAMES))

    # Compute balanced class weights: total_samples / (n_classes * count_per_class)
    n_samples = len(y_train_int)
    n_classes = len(CLASS_NAMES)
    class_weights_dict = {}
    for c in range(n_classes):
        count_c = np.sum(y_train_int == c)
        class_weights_dict[c] = float(n_samples / (n_classes * count_c)) if count_c > 0 else 1.0

    return (X_train, y_train, y_train_int), (X_val, y_val, y_val_int), (X_test, y_test, y_test_int), class_weights_dict


def create_augmented_train_pipeline(X_train, y_train, batch_size=32):
    """
    Creates a tf.data pipeline with online data augmentation and VGG16 ImageNet preprocessing.
    Augmentation ONLY applies to training set.
    """
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        tf.keras.layers.RandomZoom(height_factor=(-0.08, 0.08), fill_mode="nearest"),
        tf.keras.layers.RandomFlip("horizontal"),
    ], name="train_augmentation")

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train), seed=42)

    def augment_and_preprocess(images, labels):
        # Augment on raw [0, 255]
        aug_images = data_augmentation(images, training=True)
        # Preprocess compatible with ImageNet VGG16 (zero-centers with BGR channel means)
        prep_images = preprocess_input(aug_images)
        return prep_images, labels

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(augment_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset


def create_deterministic_pipeline(X, y, batch_size=32):
    """
    Creates a deterministic evaluation pipeline (validation / test).
    NO data augmentation. Only deterministic VGG16 ImageNet preprocessing.
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    def deterministic_preprocess(images, labels):
        prep_images = preprocess_input(images)
        return prep_images, labels

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(deterministic_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
