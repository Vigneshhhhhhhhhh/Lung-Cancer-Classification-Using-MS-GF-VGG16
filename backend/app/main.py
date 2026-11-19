from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import PLOTS_DIR
from backend.app.routes.predict import router as predict_router
from backend.app.routes.info import router as info_router
from backend.app.routes.stats import router as stats_router
from backend.app.services.model_service import ModelService

app = FastAPI(
    title="Lung Cancer Classification API (MS-GF-VGG16)",
    description="Multi-Scale Gated Feature Fusion VGG16 (MS-GF-VGG16) Deep Learning Clinical Decision Support API",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static plots directory
if PLOTS_DIR.exists():
    app.mount("/plots", StaticFiles(directory=str(PLOTS_DIR)), name="plots")

# Include Routers
app.include_router(predict_router)
app.include_router(info_router)
app.include_router(stats_router)


@app.on_event("startup")
def on_startup():
    print("[FastAPI] Server startup: Initializing ModelService...")
    ModelService.get_instance()
    print("[FastAPI] Ready to handle requests on port 8000.")


@app.get("/")
def root():
    return {
        "title": "MS-GF-VGG16 Lung Cancer Classification System",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "model_info": "GET /model-info",
            "dataset_stats": "GET /dataset-stats",
            "evaluation": "GET /evaluation",
            "ablation": "GET /ablation"
        }
    }

