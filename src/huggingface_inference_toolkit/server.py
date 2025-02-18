from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI

from huggingface_inference_toolkit.middleware import (
    LoggingMiddleware,
    PrometheusMiddleware,
)
from huggingface_inference_toolkit.routers import (
    health_router,
    metrics_router,
)
from huggingface_inference_toolkit.workers import num_workers

app = FastAPI(title="Hugging Face Inference Toolkit")

app.add_middleware(middleware_class=LoggingMiddleware)
app.add_middleware(middleware_class=PrometheusMiddleware, exclude_paths=["/health"])  # type: ignore

app.include_router(router=health_router)
app.include_router(router=metrics_router)


def launch(
    model_id: str,
    task: str,
    # TODO: maybe we should include `npu` too as supported by `sentence-transformers`?
    device: Optional[Literal["auto", "balanced", "cuda", "cpu", "mps"]] = "auto",
    # TODO: maybe the best default is no default, but handling that separately based on the library as it seems
    # that `float32` is the go to for `sentence-transformers`, `float16` for `diffusers`, and `bfloat16` for
    # `transformers` with some models performing better on `float32` or `float16` too
    dtype: Optional[Literal["float32", "float16", "bfloat16", "float8", "int8", "int4"]] = "float16",
    host: Optional[str] = "0.0.0.0",
    port: Optional[int] = 8080,
) -> None:
    from huggingface_inference_toolkit.logging import logger
    from huggingface_inference_toolkit.routers import predict_router

    if device == "auto" and task in {"text-to-image"}:
        logger.warning(
            f"{device=} is set, but on `diffusers` only `device_map='balanced'` is supported at the moment,"
            " meaning that the different pipeline components will be distributed among the available devices."
        )
        device = "balanced"

    if not dtype and task in {"text-to-image"}:
        dtype = "float16"

    # Python 3.10 should be the minimum supported version (?)
    match task:
        # diffusers
        case "text-to-image":
            from huggingface_inference_toolkit.tasks.diffusers.text_to_image import (
                TextToImage,
                TextToImageInput,
                TextToImageOutput,
            )

            predictor = TextToImage(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = TextToImageInput, TextToImageOutput
        # sentence-transformers
        case "sentence-similarity":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_similarity import (
                SentenceSimilarity,
                SentenceSimilarityInput,
                SentenceSimilarityOutput,
            )

            predictor = SentenceSimilarity(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = SentenceSimilarityInput, SentenceSimilarityOutput
        # case "sentence-embeddings":
        #     ...
        # case "sentence-ranking":
        #     ...
        # transformers
        case "text-classification":
            from huggingface_inference_toolkit.tasks.transformers.text_classification import (
                TextClassification,
                TextClassificationInput,
                TextClassificationOutput,
            )

            predictor = TextClassification(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = TextClassificationInput, TextClassificationOutput
        case _:
            raise ValueError(f"{task=} not supported!")

    app.include_router(
        router=predict_router(
            predictor=predictor,
            input_schema=input_schema,
            output_schema=output_schema,
        )
    )

    logger.info(f"Loaded {model_id=} with {task=} on {device=}.")

    uvicorn.run(
        "huggingface_inference_toolkit.server:app",
        host=host,  # type: ignore
        port=port,  # type: ignore
        log_level=0,
        use_colors=True,
        workers=num_workers(),
    )
