import os
import time
from pathlib import Path
from typing import Literal, Optional, Union

import uvicorn
from fastapi import FastAPI

from huggingface_inference_toolkit.clouds.azure import router as azure_router
from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.middleware import (
    LoggingMiddleware,
    PrometheusMiddleware,
    RequestIdMiddleware,
)
from huggingface_inference_toolkit.routers import (
    custom_router,
    health_router,
    metrics_router,
    predict_media_router,
    predict_router,
)

app = FastAPI(title="Hugging Face Inference Toolkit")

# NOTE: FastAPI runs the middlewares in reverse order
app.add_middleware(middleware_class=PrometheusMiddleware, exclude_paths=["/health"])  # type: ignore
app.add_middleware(
    middleware_class=LoggingMiddleware, inference_paths=["/", "/predict", "/score", "/v1/chat/completions"]
)
app.add_middleware(middleware_class=RequestIdMiddleware, exclude_paths=["/health"])

app.include_router(router=health_router)
app.include_router(router=metrics_router)
app.include_router(router=azure_router)


def launch(
    model_id: Union[str, None],
    # NOTE: on Inference Endpoints the model is downloaded in advance into the `/repository` directory
    # meaning that the `model_id` will always be None there, and the `model_dir` will be used instead
    model_dir: Union[str, None],
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
    if model_id and model_dir:
        logger.warning(
            f"Both {model_id=} and {model_dir=} have been provided but those are mutually exclusive, if both are provided then `--model-dir` has preference over `--model-id`"
        )
        model_id = None

    if not model_id and not model_dir:
        raise ValueError(
            "Any of `--model-id` or `--model-dir` should be provided but both cannot be None (alternatively those can be provided via the environment variables `MODEL_ID` or `MODEL_DIR`, respectively."
        )

    logger.info(
        f"Starting toolkit server for model {model_id or model_dir=} with task {task=} on device {device=}"
    )

    match task:
        # openai-compatible
        case "image-text-to-text":
            from huggingface_inference_toolkit.tasks.transformers.image_text_to_text import (
                ImageTextToText,
                ImageTextToTextInput,
                ImageTextToTextOutput,
            )

            predictor = ImageTextToText(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_router(
                    predictor=predictor,
                    input_schema=ImageTextToTextInput,
                    output_schema=ImageTextToTextOutput,
                )
            )
            if (
                predictor.pipeline.tokenizer is not None
                and predictor.pipeline.tokenizer.chat_template is not None
            ):
                from huggingface_inference_toolkit.openai.routers import chat_completions_router, models_router
                from huggingface_inference_toolkit.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,
                    tokenizer=predictor.pipeline.tokenizer,  # type: ignore
                )
                app.include_router(router=chat_completions_router(predictor=chat_completions))
                app.include_router(router=models_router(predictor=chat_completions, timestamp=int(time.time())))
        case "text-generation" | "text2text-generation" | "conversational":
            from huggingface_inference_toolkit.tasks.transformers.text_generation import (
                TextGeneration,
                TextGenerationInput,
                TextGenerationOutput,
            )

            predictor = TextGeneration(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_router(
                    predictor=predictor,
                    input_schema=TextGenerationInput,
                    output_schema=TextGenerationOutput,
                )
            )
            if (
                predictor.pipeline.tokenizer is not None
                and predictor.pipeline.tokenizer.chat_template is not None
            ):
                from huggingface_inference_toolkit.openai.routers import chat_completions_router, models_router
                from huggingface_inference_toolkit.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,
                    tokenizer=predictor.pipeline.tokenizer,  # type: ignore
                )
                app.include_router(router=chat_completions_router(predictor=chat_completions))
                app.include_router(router=models_router(predictor=chat_completions, timestamp=int(time.time())))
        # diffusers
        case "text-to-image":
            from huggingface_inference_toolkit.tasks.diffusers.text_to_image import (
                TextToImage,
                TextToImageInput,
                TextToImageOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=TextToImage(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=TextToImageInput,
                    output_schema=TextToImageOutput,
                )
            )
        # sentence-transformers
        case "sentence-similarity":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_similarity import (
                SentenceSimilarity,
                SentenceSimilarityInput,
                SentenceSimilarityOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=SentenceSimilarity(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=SentenceSimilarityInput,
                    output_schema=SentenceSimilarityOutput,
                )
            )
        case "sentence-embeddings":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_embeddings import (
                SentenceEmbeddings,
                SentenceEmbeddingsInput,
                SentenceEmbeddingsOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=SentenceEmbeddings(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=SentenceEmbeddingsInput,
                    output_schema=SentenceEmbeddingsOutput,
                )
            )
        case "sentence-ranking":
            from huggingface_inference_toolkit.tasks.sentence_transformers.sentence_ranking import (
                SentenceRanking,
                SentenceRankingInput,
                SentenceRankingOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=SentenceRanking(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=SentenceRankingInput,  # type: ignore
                    output_schema=SentenceRankingOutput,  # type: ignore
                )
            )
        # transformers
        case "text-classification":
            from huggingface_inference_toolkit.tasks.transformers.text_classification import (
                TextClassification,
                TextClassificationInput,
                TextClassificationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=TextClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=TextClassificationInput,
                    output_schema=TextClassificationOutput,
                )
            )
        case "fill-mask":
            from huggingface_inference_toolkit.tasks.transformers.fill_mask import (
                FillMask,
                FillMaskInput,
                FillMaskOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=FillMask(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=FillMaskInput,
                    output_schema=FillMaskOutput,
                )
            )
        case "question-answering":
            from huggingface_inference_toolkit.tasks.transformers.question_answering import (
                QuestionAnswering,
                QuestionAnsweringInput,
                QuestionAnsweringOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=QuestionAnswering(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=QuestionAnsweringInput,
                    output_schema=QuestionAnsweringOutput,
                )
            )
        case "summarization":
            from huggingface_inference_toolkit.tasks.transformers.summarization import (
                Summarization,
                SummarizationInput,
                SummarizationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=Summarization(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=SummarizationInput,
                    output_schema=SummarizationOutput,
                )
            )
        case "zero-shot-classification":
            from huggingface_inference_toolkit.tasks.transformers.zero_shot_classification import (
                ZeroShotClassification,
                ZeroShotClassificationInput,
                ZeroShotClassificationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=ZeroShotClassification(
                        model_id=model_id or model_dir,  # type: ignore
                        dtype=dtype,  # type: ignore
                        device=device,  # type: ignore
                    ),
                    input_schema=ZeroShotClassificationInput,
                    output_schema=ZeroShotClassificationOutput,
                )
            )
        case "token-classification":
            from huggingface_inference_toolkit.tasks.transformers.token_classification import (
                TokenClassification,
                TokenClassificationInput,
                TokenClassificationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=TokenClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=TokenClassificationInput,
                    output_schema=TokenClassificationOutput,
                )
            )
        case "table-question-answering":
            from huggingface_inference_toolkit.tasks.transformers.table_question_answering import (
                TableQuestionAnswering,
                TableQuestionAnsweringInput,
                TableQuestionAnsweringOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=TableQuestionAnswering(
                        model_id=model_id or model_dir,  # type: ignore
                        dtype=dtype,  # type: ignore
                        device=device,  # type: ignore
                    ),
                    input_schema=TableQuestionAnsweringInput,
                    output_schema=TableQuestionAnsweringOutput,
                )
            )
        case "translation" | "translation_xx_to_yy":
            from huggingface_inference_toolkit.tasks.transformers.translation import (
                Translation,
                TranslationInput,
                TranslationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=Translation(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=TranslationInput,
                    output_schema=TranslationOutput,
                )
            )
        # transformers - audio
        case "zero-shot-audio-classification":
            from huggingface_inference_toolkit.tasks.transformers.zero_shot_audio_classification import (
                ZeroShotAudioClassification,
                ZeroShotAudioClassificationInput,
                ZeroShotAudioClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ZeroShotAudioClassification(
                        model_id=model_id or model_dir, dtype=dtype, device=device
                    ),  # type: ignore
                    input_schema=ZeroShotAudioClassificationInput,
                    output_schema=ZeroShotAudioClassificationOutput,
                    accepted_mimetypes=[
                        "audio/flac",
                        "audio/xflac",
                        "audio/mpeg",
                        "audio/mp4",
                        "audio/ogg",
                        "audio/wav",
                        "audio/webm",
                    ],
                )
            )
        case "audio-classification":
            from huggingface_inference_toolkit.tasks.transformers.audio_classification import (
                AudioClassification,
                AudioClassificationInput,
                AudioClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=AudioClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=AudioClassificationInput,
                    output_schema=AudioClassificationOutput,
                    accepted_mimetypes=[
                        "audio/flac",
                        "audio/xflac",
                        "audio/mpeg",
                        "audio/mp4",
                        "audio/ogg",
                        "audio/wav",
                        "audio/webm",
                    ],
                )
            )
        case "automatic-speech-recognition":
            from huggingface_inference_toolkit.tasks.transformers.automatic_speech_recognition import (
                AutomaticSpeechRecognition,
                AutomaticSpeechRecognitionInput,
                AutomaticSpeechRecognitionOutput,
            )

            predictor = AutomaticSpeechRecognition(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_media_router(
                    predictor=predictor,
                    input_schema=AutomaticSpeechRecognitionInput,
                    output_schema=AutomaticSpeechRecognitionOutput,
                    accepted_mimetypes=[
                        "audio/flac",
                        "audio/xflac",
                        "audio/mpeg",
                        "audio/mp4",
                        "audio/ogg",
                        "audio/wav",
                        "audio/webm",
                    ],
                )
            )
        # transformers - image
        case "image-classification":
            from huggingface_inference_toolkit.tasks.transformers.image_classification import (
                ImageClassification,
                ImageClassificationInput,
                ImageClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ImageClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ImageClassificationInput,
                    output_schema=ImageClassificationOutput,
                    accepted_mimetypes=["image/jpeg", "image/png", "image/bmp", "image/webp"],  # type: ignore
                )
            )
        case "image-segmentation":
            from huggingface_inference_toolkit.tasks.transformers.image_segmentation import (
                ImageSegmentation,
                ImageSegmentationInput,
                ImageSegmentationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ImageSegmentation(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ImageSegmentationInput,
                    output_schema=ImageSegmentationOutput,
                    accepted_mimetypes=["image/jpeg", "image/png", "image/bmp", "image/webp"],
                )
            )
        case "object-detection":
            from huggingface_inference_toolkit.tasks.transformers.object_detection import (
                ObjectDetection,
                ObjectDetectionInput,
                ObjectDetectionOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ObjectDetection(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ObjectDetectionInput,
                    output_schema=ObjectDetectionOutput,
                    accepted_mimetypes=["image/jpeg", "image/png", "image/bmp", "image/webp"],
                )
            )
        # custom
        case "custom":
            if os.getenv("TRUST_REMOTE_CODE", None) is None or os.getenv("TRUST_REMOTE_CODE", None) in {
                0,
                "false",
                "False",
            }:
                raise RuntimeError(
                    f"Since `TRUST_REMOTE_CODE` (formerly known as `HF_TRUST_REMOTE_CODE`) is set to {os.getenv('TRUST_REMOTE_CODE', None)}, it means that the `custom` task cannot run, as it requires to pull and run custom code. Enabling it as `TRUST_REMOTE_CODE=(1, true, True)` is not recommended, unless you either developed the custom code or trust the developer / organization."
                )

            from huggingface_hub import snapshot_download

            from huggingface_inference_toolkit.tasks.custom import Custom

            # NOTE: if `model_id` is provided, then download the repository first (or just pull it from the cache
            # if already there); note that if `model_id` is not None here that means that `model_dir` is None and
            # the other way around
            if model_id is not None:
                model_dir = snapshot_download(repo_id=model_id, repo_type="model")

            try:
                app.include_router(router=custom_router(handler=Custom.load(model_dir)))  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Attempted to load a custom handler i.e. the class `{os.getenv('CUSTOM_HANDLER')}` from `{model_dir=}/{os.getenv('CUSTOM_HANDLER_FILE')}`, but either didn't find any or couldn't load it (as per \"{e}\")"
                )
        case _:
            raise ValueError(f"{task=} not supported!")

    logger.info(f"Loaded {model_id or model_dir=} with {task=} on {device=}.")

    # NOTE: some models as `microsoft/Magma-8B` may contain custom routers as those are not covered within the
    # default implementation, to solve that, we create those under `src/huggingface_inference_toolkit/models`
    # and add those on top of whatever the default router is.
    if model_id:
        model_file_name = model_id.replace("/", "--").lower()
        models_dir = Path(__file__).parent / "models"
        model_file_path = models_dir / f"{model_file_name}.py"
        if model_file_path.exists():
            logger.info(f"Provided {model_id=} has a custom model file at {model_file_path=}")
            try:
                import importlib.util

                spec = importlib.util.spec_from_file_location(model_file_name, model_file_path)
                if spec and spec.loader:
                    model_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(model_module)

                    if hasattr(model_module, "router"):
                        logger.info(f"Loading custom router for {model_id=}")
                        app.include_router(router=model_module.router)
                        logger.info(f"Loaded custom router for {model_id=}")
            except Exception as e:
                logger.warning(f"Failed to load custom router for {model_id}: {e}")

    logger.info("Available API routes:")
    groups = {"/docs": "/docs/oauth2-redirect", "/openapi.json": "/swagger.json", "/": "/predict"}
    logged = set()

    for route in app.routes:
        if hasattr(route, "methods") and hasattr(route, "path") and route.path not in logged:  # type: ignore
            methods = [m for m in sorted(route.methods) if m != "HEAD"]  # type: ignore
            for method in methods:
                if route.path in groups:  # type: ignore
                    logger.info(f"[{method:<4}] {route.path}, {groups[route.path]}")  # type: ignore
                    logged.update([route.path, groups[route.path]])  # type: ignore
                elif route.path not in groups.values():  # type: ignore
                    logger.info(f"[{method:<4}] {route.path}")  # type: ignore
                logged.add(route.path)  # type: ignore

    uvicorn.run(
        "huggingface_inference_toolkit.server:app",
        host=host,  # type: ignore
        port=port,  # type: ignore
        log_level=0,
        access_log=False,
        use_colors=True,
        workers=1,
    )
