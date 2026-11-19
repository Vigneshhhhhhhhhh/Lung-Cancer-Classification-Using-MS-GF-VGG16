from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from backend.app.services.model_service import ModelService

router = APIRouter(prefix="", tags=["Prediction"])

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "JPG", "JPEG", "PNG"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB limit


@router.post("/predict")
async def predict_ct_scan(
    file: UploadFile = File(...),
    gradcam: bool = Query(True, description="Whether to generate Grad-CAM visualization")
):
    """
    Accepts an uploaded lung CT scan image (JPG, JPEG, PNG),
    preprocesses it using ImageNet VGG16 standard pipeline,
    and runs inference on the trained MS-GF-VGG16 model.
    """
    filename = file.filename or "unknown.png"
    ext = filename.split(".")[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload a valid JPG, JPEG, or PNG image."
        )

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowable limit of 15MB."
            )
        if len(content) == 0:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        model_service = ModelService.get_instance()
        result = model_service.predict_image(content, generate_cam=gradcam)
        result["filename"] = filename
        result["file_size_kb"] = round(len(content) / 1024, 2)
        return result

    except HTTPException:
        raise
    except ValueError as val_err:
        raise HTTPException(status_code=422, detail=str(val_err))
    except RuntimeError as run_err:
        raise HTTPException(status_code=503, detail=str(run_err))
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(err)}")

