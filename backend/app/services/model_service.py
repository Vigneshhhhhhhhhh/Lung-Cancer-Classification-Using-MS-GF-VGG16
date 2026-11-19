import io
import base64
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

from backend.config import (
    BEST_MODEL_PATH, IMAGE_SIZE, CLASS_NAMES, DISPLAY_NAMES,
    METRICS_PATH, HISTORY_PATH, ABLATION_PATH
)
from backend.training.model import ScaleGatedFeatureFusionUnit
from backend.training.gradcam import compute_gradcam, overlay_gradcam


class ModelService:
    _instance = None

    def __init__(self):
        self.model = None
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelService()
        return cls._instance

    def _load_model(self):
        if BEST_MODEL_PATH.exists():
            try:
                print(f"[ModelService] Loading trained model from {BEST_MODEL_PATH}...")
                self.model = tf.keras.models.load_model(
                    str(BEST_MODEL_PATH),
                    custom_objects={"ScaleGatedFeatureFusionUnit": ScaleGatedFeatureFusionUnit}
                )
                self.is_loaded = True
                print("[ModelService] Model successfully loaded into memory.")
            except Exception as e:
                print(f"[ModelService] Error loading model: {e}")
                self.model = None
                self.is_loaded = False
        else:
            print(f"[ModelService] Checkpoint not yet found at {BEST_MODEL_PATH}")
            self.model = None
            self.is_loaded = False

    def predict_image(self, image_bytes: bytes, generate_cam: bool = True):
        if not self.is_loaded or self.model is None:
            # Attempt reloading in case training completed recently
            self._load_model()
            if not self.is_loaded:
                raise RuntimeError("Model checkpoint not available. Model training is in progress or checkpoint not found.")

        # Open PIL Image
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid image file format: {e}")

        # Preprocess for model input: 128x128x3
        resized_pil = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        img_arr = np.array(resized_pil, dtype=np.float32)
        img_tensor = np.expand_dims(img_arr, axis=0)  # (1, 128, 128, 3)

        # Apply ImageNet VGG16 zero-centering
        preprocessed = preprocess_input(img_tensor.copy())

        # Prediction
        preds = self.model.predict(preprocessed, verbose=0)[0]
        pred_idx = int(np.argmax(preds))
        confidence = float(preds[pred_idx])
        pred_class = CLASS_NAMES[pred_idx]
        pred_display = DISPLAY_NAMES[pred_class]

        # Probabilities dictionary
        class_probabilities = []
        for idx, cls in enumerate(CLASS_NAMES):
            class_probabilities.append({
                "class_id": cls,
                "display_name": DISPLAY_NAMES[cls],
                "probability": round(float(preds[idx]), 4),
                "percentage": round(float(preds[idx]) * 100, 2)
            })

        # Sort by probability descending
        class_probabilities.sort(key=lambda x: x["probability"], reverse=True)

        # Grad-CAM explainability
        gradcam_data = None
        if generate_cam:
            try:
                heatmap = compute_gradcam(self.model, preprocessed, target_layer_name="sgfu", pred_index=pred_idx)
                overlay_pil, heatmap_pil = overlay_gradcam(pil_img, heatmap, alpha=0.45)

                def pil_to_base64(img):
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=90)
                    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

                gradcam_data = {
                    "original_image": pil_to_base64(pil_img),
                    "heatmap_image": pil_to_base64(heatmap_pil),
                    "overlay_image": pil_to_base64(overlay_pil),
                    "target_layer": "sgfu (Multi-Scale Fused Feature Map)"
                }
            except Exception as cam_err:
                print(f"[ModelService] Grad-CAM generation warning: {cam_err}")
                gradcam_data = None

        return {
            "predicted_class": pred_class,
            "display_name": pred_display,
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": class_probabilities,
            "gradcam": gradcam_data
        }

