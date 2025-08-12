from huggingface_inference_toolkit.routers.custom import router as custom_router
from huggingface_inference_toolkit.routers.health import router as health_router
from huggingface_inference_toolkit.routers.metrics import router as metrics_router
from huggingface_inference_toolkit.routers.predict import router as predict_router
from huggingface_inference_toolkit.routers.predict import audio_router as predict_audio_router

__all__ = ["custom_router", "health_router", "metrics_router", "predict_audio_router", "predict_router"]
