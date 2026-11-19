import base64
from pathlib import Path
from fastapi import APIRouter
from backend.config import DATA_DIR, CLASS_NAMES, DISPLAY_NAMES
from backend.training.dataset import scan_and_validate_dataset, resolve_canonical_class

router = APIRouter(prefix="", tags=["Dataset Statistics & Samples"])

_cached_stats = None


@router.get("/dataset-stats")
def get_dataset_statistics():
    """Returns dataset image counts across train, valid, test splits, and class imbalance metrics."""
    global _cached_stats
    if _cached_stats is None:
        try:
            _cached_stats = scan_and_validate_dataset(DATA_DIR)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed scanning dataset: {str(e)}"
            }

    return {
        "status": "success",
        "dataset": _cached_stats
    }


@router.get("/samples")
def get_sample_test_images():
    """
    Returns base64 data URLs for representative CT scans from each of the 4 test classes,
    allowing instant demonstration without needing external file downloads.
    """
    samples = []
    test_dir = DATA_DIR / "test"

    for sub_dir in sorted(test_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        canonical = resolve_canonical_class(sub_dir.name)
        img_files = [f for f in sorted(sub_dir.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if img_files:
            sample_file = img_files[0]
            with open(sample_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            mime = "image/jpeg" if sample_file.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
            samples.append({
                "class_id": canonical,
                "display_name": DISPLAY_NAMES[canonical],
                "filename": sample_file.name,
                "data_url": f"data:{mime};base64,{b64}"
            })

    return {
        "samples": samples
    }
