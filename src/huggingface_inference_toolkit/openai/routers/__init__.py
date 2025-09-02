from huggingface_inference_toolkit.openai.routers.chat_completions import router as chat_completions_router
from huggingface_inference_toolkit.openai.routers.images_generations import router as images_generations_router
from huggingface_inference_toolkit.openai.routers.models import router as models_router

__all__ = ["chat_completions_router", "images_generations_router", "models_router"]
