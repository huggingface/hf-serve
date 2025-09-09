from hf_serve.openai.routers.audio_transcriptions import router as audio_transcriptions_router
from hf_serve.openai.routers.chat_completions import router as chat_completions_router
from hf_serve.openai.routers.embeddings import router as embeddings_router
from hf_serve.openai.routers.images_generations import router as images_generations_router
from hf_serve.openai.routers.models import router as models_router

__all__ = ["audio_transcriptions_router", "chat_completions_router", "embeddings_router", "images_generations_router", "models_router"]
