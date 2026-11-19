import json
from fastapi import APIRouter
from backend.config import (
    BEST_MODEL_PATH, METRICS_PATH, HISTORY_PATH, ABLATION_PATH,
    CLASS_NAMES, DISPLAY_NAMES, IMAGE_SHAPE
)
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="", tags=["Model Information"])


@router.get("/health")
def health_check():
    """Returns backend and model operational health status."""
    service = ModelService.get_instance()
    return {
        "status": "healthy",
        "service": "Lung Cancer MS-GF-VGG16 API",
        "model_loaded": service.is_loaded,
        "model_checkpoint_exists": BEST_MODEL_PATH.exists()
    }


@router.get("/model-info")
def get_model_info():
    """Returns architecture overview, tensor dimensions, and current deployment status."""
    service = ModelService.get_instance()
    
    # Check if actual evaluation metrics are saved
    eval_metrics = None
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                eval_metrics = json.load(f)
        except Exception:
            eval_metrics = None

    return {
        "model_name": "MS-GF-VGG16 (Multi-Scale Gated Feature Fusion VGG16)",
        "backbone": "ImageNet Pretrained VGG16",
        "fusion_mechanism": "Scale-Gated Feature Fusion Unit (SGFU)",
        "input_dimensions": list(IMAGE_SHAPE),
        "num_classes": len(CLASS_NAMES),
        "classes": [
            {"id": c, "display_name": DISPLAY_NAMES[c]} for c in CLASS_NAMES
        ],
        "model_status": "Ready for Inference" if service.is_loaded else "Training in Progress",
        "evaluation_summary": {
            "test_accuracy": eval_metrics.get("accuracy") if eval_metrics else None,
            "macro_f1": eval_metrics.get("f1_macro") if eval_metrics else None,
            "macro_specificity": eval_metrics.get("specificity_macro") if eval_metrics else None,
            "macro_roc_auc": eval_metrics.get("roc_auc_macro") if eval_metrics else None
        } if eval_metrics else None
    }


@router.get("/evaluation")
def get_evaluation_metrics():
    """Returns comprehensive test evaluation metrics, confusion matrix, and ROC data."""
    if not METRICS_PATH.exists():
        return {
            "status": "in_progress",
            "message": "Model evaluation metrics are not yet available. Training is currently running.",
            "metrics": None
        }

    try:
        with open(METRICS_PATH, "r") as f:
            metrics_data = json.load(f)

        history_data = None
        if HISTORY_PATH.exists():
            with open(HISTORY_PATH, "r") as hf:
                history_data = json.load(hf)

        return {
            "status": "completed",
            "metrics": metrics_data,
            "history": history_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error loading evaluation metrics: {str(e)}"
        }


@router.get("/ablation")
def get_ablation_study():
    """Returns empirical ablation study comparison across the three architectural variants."""
    if not ABLATION_PATH.exists():
        return {
            "status": "pending",
            "message": "Ablation study benchmarks are currently being computed.",
            "experiments": []
        }

    try:
        with open(ABLATION_PATH, "r") as f:
            ablation_data = json.load(f)
        return {
            "status": "completed",
            "experiments": ablation_data
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "experiments": []
        }

