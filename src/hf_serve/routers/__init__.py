from hf_serve.routers.custom import router as custom_router
from hf_serve.routers.health import router as health_router
from hf_serve.routers.metrics import router as metrics_router
from hf_serve.routers.predict import router as predict_router
from hf_serve.routers.predict_image import router as predict_image_router
from hf_serve.routers.predict_media import media_router as predict_media_router

__all__ = [
    "custom_router",
    "health_router",
    "metrics_router",
    "predict_image_router",
    "predict_media_router",
    "predict_router",
]
