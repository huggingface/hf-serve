from typing import Literal

TaskTypes = Literal[
    "image-text-to-text",
    "text-generation",
    "text2text-generation",  # NOTE: deprecated in favour of `text-generation`
    "conversational",  # NOTE: deprecated in favour of `text-generation`
    "chat-completion",  # NOTE: alternative naming for `text-generation` and `image-text-to-text` (used in e.g. Azure AI)
    "text-to-image",
    "sentence-similarity",
    "feature-extraction",
    "sentence-embeddings",  # NOTE: former Inference API task name for `feature-extraction`
    "embeddings",  # NOTE: alternative naming for `feature-extraction` (used in e.g. Azure AI)
    "text-ranking",
    "sentence-ranking",  # NOTE: former Inference API task name for `text-ranking`
    "text-classification",
    "fill-mask",
    "question-answering",
    "summarization",
    "zero-shot-classification",
    "token-classification",
    "table-question-answering",
    "visual-question-answering",
    "vqa",  # NOTE: shortened name for `visual-question-answering` given that the `pipeline` supports both
    "translation",
    "translation_xx_to_yy",  # NOTE: placeholder task name where `xx` and `yy` are the source and target languages, respectively
    "zero-shot-audio-classification",
    "audio-classification",
    "automatic-speech-recognition",
    "image-classification",
    "zero-shot-image-classification",
    "image-segmentation",
    "object-detection",
    "mask-generation",
    "image-to-image",
    "custom",  # NOTE: ideally not recommended for production use-cases as it requires `trust_remote_code=True`, but here to ensure compatibility with the former `huggingface-inference-toolkit`
]
