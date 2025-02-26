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
        case "sentence-embeddings":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_embeddings import (
                SentenceEmbeddings,
                SentenceEmbeddingsInput,
                SentenceEmbeddingsOutput,
            )

            predictor = SentenceEmbeddings(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = SentenceEmbeddingsInput, SentenceEmbeddingsOutput
        case "sentence-ranking":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_ranking import (
                SentenceRanking,
                SentenceRankingInput,
                SentenceRankingOutput,
            )

            predictor = SentenceRanking(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = SentenceRankingInput, SentenceRankingOutput
        # transformers
        case "text-classification":
            from huggingface_inference_toolkit.tasks.transformers.text_classification import (
                TextClassification,
                TextClassificationInput,
                TextClassificationOutput,
            )

            predictor = TextClassification(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = TextClassificationInput, TextClassificationOutput

        case "fill-mask":
            from huggingface_inference_toolkit.tasks.transformers.fill_mask import (
                FillMask,
                FillMaskInput,
                FillMaskOutput,
            )

            predictor = FillMask(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = FillMaskInput, FillMaskOutput

        case "question-answering":
            from huggingface_inference_toolkit.tasks.transformers.question_answering import (
                QuestionAnswering,
                QuestionAnsweringInput,
                QuestionAnsweringOutput,
            )

            predictor = QuestionAnswering(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = QuestionAnsweringInput, QuestionAnsweringOutput

        case "summarization":
            from huggingface_inference_toolkit.tasks.transformers.summarization import (
                Summarization,
                SummarizationInput,
                SummarizationOutput,
            )

            predictor = Summarization(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = SummarizationInput, SummarizationOutput

        case "zero-shot-classification":
            from huggingface_inference_toolkit.tasks.transformers.zero_shot_classification import (
                ZeroShotClassification,
                ZeroShotClassificationInput,
                ZeroShotClassificationOutput,
            )

            predictor = ZeroShotClassification(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = ZeroShotClassificationInput, ZeroShotClassificationOutput

        case "table-question-answering":
            from huggingface_inference_toolkit.tasks.transformers.table_question_answering import (
                TableQuestionAnswering,
                TableQuestionAnsweringInput,
                TableQuestionAnsweringOutput,
            )

            predictor = TableQuestionAnswering(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = TableQuestionAnsweringInput, TableQuestionAnsweringOutput

        case "translation" | "translation_xx_to_yy":
            from huggingface_inference_toolkit.tasks.transformers.translation import (
                Translation,
                TranslationInput,
                TranslationOutput,
            )

            predictor = Translation(model_id=model_id, dtype=dtype, device=device)  # type: ignore
            input_schema, output_schema = TranslationInput, TranslationOutput


        case _:
            raise ValueError(f"{task=} not supported!")

    app.include_router(
        router=predict_router(
            predictor=predictor,
            # TODO: maybe not now, but would be nice to run `mypy` to double check types,
            # for now just adding `# type: ignore` to prevent random warnings that are false
            # negatives in most of the cases as e.g. below due to the Union
            input_schema=input_schema,  # type: ignore
            output_schema=output_schema,  # type: ignore
        )
    )

    logger.info(f"Loaded {model_id=} with {task=} on {device=}.")

    uvicorn.run(
        "huggingface_inference_toolkit.server:app",
        host=host,  # type: ignore
        port=port,  # type: ignore
        log_level=0,
        use_colors=True,
        # NOTE: temporarily removed to just use one worker per replica
        # workers=num_workers(),
        workers=1,
    )
