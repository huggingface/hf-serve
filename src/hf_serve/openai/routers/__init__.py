from hf_serve.openai.routers.chat_completions import router as chat_completions_router
from hf_serve.openai.routers.embeddings import router as embeddings_router
from hf_serve.openai.routers.images_generations import router as images_generations_router
from hf_serve.openai.routers.models import router as models_router
from hf_serve.openai.routers.speech import router as speech_router

__all__ = [
    "chat_completions_router",
    "embeddings_router",
    "images_generations_router",
    "models_router",
    "speech_router",
]
