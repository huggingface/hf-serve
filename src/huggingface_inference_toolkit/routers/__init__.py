from huggingface_inference_toolkit.routers.health import router as health_router
from huggingface_inference_toolkit.routers.metrics import router as metrics_router
from huggingface_inference_toolkit.routers.predict import router as predict_router

__all__ = ["health_router", "metrics_router", "predict_router"]
