import os
import time
from typing import List, Literal, Optional, Union

import uvicorn
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

from hf_serve.logging import logger
from hf_serve.middleware import (
    LoggingMiddleware,
    PrometheusMiddleware,
    RequestIdMiddleware,
)
from hf_serve.routers import (
    custom_router,
    health_router,
    metrics_router,
    predict_media_router,
    predict_router,
)
from hf_serve.server_utils import log_available_routes
from hf_serve.types import TaskTypes

app = FastAPI(title="Hugging Face Serve API")

# NOTE: FastAPI runs the middlewares in reverse order
app.add_middleware(middleware_class=PrometheusMiddleware, exclude_paths=["/health"])  # type: ignore
app.add_middleware(
    middleware_class=LoggingMiddleware,
    # NOTE: temporarily excluding it from the logging as otherwise Inference Endpoints API gets too verbose
    exclude_paths=["/health"],
    inference_paths=[
        "/",
        "/predict",
        "/predict-file",
        "/predict-form",
        "/predict-json",
        "/score",
        "/v1/chat/completions",
        "/v1/images/generations",
        "/v1/embeddings",
    ],
)
app.add_middleware(middleware_class=RequestIdMiddleware, exclude_paths=["/health"])

app.include_router(router=health_router)
app.include_router(router=metrics_router)


# NOTE: If not defined, then the FastAPI responses when validation via e.g. `payload: Payload = Body(...)`
# will just show a vague unreadable error, this way the error is a readable JSON with an actionable outcome
# Reference: https://fastapi.tiangolo.com/tutorial/handling-errors/#use-the-requestvalidationerror-body
# TODO: Given that the error is still not perfect, investigate on another potential way of handling those
# even if manually to keep those as simple as e.g. "Required 'X' but not provided in payload 'Y'"
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


def launch(
    model_id: Union[str, None],
    # NOTE: on Inference Endpoints the model is downloaded in advance into the `/repository` directory
    # meaning that the `model_id` will always be None there, and the `model_dir` will be used instead
    model_dir: Union[str, None],
    task: TaskTypes,
    # TODO: maybe we should include `npu` too as supported by `sentence-transformers`?
    device: Optional[Literal["auto", "balanced", "cuda", "cpu", "mps"]] = "auto",
    # TODO: maybe the best default is no default, but handling that separately based on the library as it seems
    # that `float32` is the go to for `sentence-transformers`, `float16` for `diffusers`, and `bfloat16` for
    # `transformers` with some models performing better on `float32` or `float16` too
    dtype: Optional[Literal["float32", "float16", "bfloat16", "float8", "int8", "int4"]] = "float16",
    accepted_mimetypes: Optional[List[str]] = None,
    max_file_size: Optional[int] = None,
    host: Optional[str] = "0.0.0.0",
    port: Optional[int] = 8080,
    cloud: Optional[Literal["azure"]] = None,
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

    logger.info(f"`hf-serve` starting for model {model_id or model_dir=} with {task=} on {device=}")

    match task:
        # openai-compatible
        case "image-text-to-text":
            from hf_serve.tasks.transformers.image_text_to_text import (
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
                from hf_serve.openai.routers import chat_completions_router, models_router
                from hf_serve.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,
                    tokenizer=predictor.pipeline.tokenizer,  # type: ignore
                )
                app.include_router(router=chat_completions_router(predictor=chat_completions))
                app.include_router(
                    router=models_router(model_id=chat_completions.model_id, timestamp=int(time.time()))  # type: ignore
                )
        case "text-generation" | "text2text-generation" | "conversational":
            from hf_serve.tasks.transformers.text_generation import (
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
                from hf_serve.openai.routers import chat_completions_router, models_router
                from hf_serve.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,
                    tokenizer=predictor.pipeline.tokenizer,  # type: ignore
                )
                app.include_router(router=chat_completions_router(predictor=chat_completions))
                app.include_router(
                    router=models_router(model_id=chat_completions.model_id, timestamp=int(time.time()))  # type: ignore
                )
        # diffusers
        case "text-to-image":
            import torch

            if dtype != "float32" and (
                device == "mps" or (device == "auto" and torch.backends.mps.is_available())
            ):
                raise RuntimeError("Support for `text-to-image` on MPS is unstable.")

            from hf_serve.routers import predict_image_router
            from hf_serve.tasks.diffusers.text_to_image import TextToImage, TextToImageInput

            predictor = TextToImage(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_image_router(
                    predictor=predictor,
                    input_schema=TextToImageInput,
                )
            )

            from hf_serve.openai.routers import images_generations_router, models_router
            from hf_serve.openai.tasks.images_generations import ImagesGenerations

            images_generations = ImagesGenerations(pipeline=predictor.pipeline)
            app.include_router(router=images_generations_router(predictor=images_generations))
            app.include_router(
                router=models_router(model_id=images_generations.model_id, timestamp=int(time.time()))  # type: ignore
            )
        # sentence-transformers
        case "sentence-similarity":
            from hf_serve.tasks.sentence_transformers.sentence_similarity import (
                SentenceSimilarity,
                SentenceSimilarityInput,
                SentenceSimilarityOutput,
            )

            predictor = SentenceSimilarity(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_router(
                    predictor=predictor,
                    input_schema=SentenceSimilarityInput,
                    output_schema=SentenceSimilarityOutput,
                )
            )

            from hf_serve.openai.routers import embeddings_router, models_router
            from hf_serve.openai.tasks.embeddings import Embeddings

            embeddings = Embeddings(pipeline=predictor.pipeline)
            app.include_router(router=embeddings_router(predictor=embeddings))
            app.include_router(
                router=models_router(model_id=embeddings.model_id, timestamp=int(time.time()))  # type: ignore
            )

            from hf_serve.compatibility.text_embeddings_inference import (
                router as text_embeddings_inference_router,
            )

            app.include_router(router=text_embeddings_inference_router)
        case "feature-extraction" | "sentence-embeddings" | "embeddings":
            from hf_serve.tasks.sentence_transformers.feature_extraction import (
                FeatureExtraction,
                FeatureExtractionInput,
                FeatureExtractionOutput,
            )

            predictor = FeatureExtraction(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_router(
                    predictor=predictor,
                    input_schema=FeatureExtractionInput,
                    output_schema=FeatureExtractionOutput,
                )
            )

            from hf_serve.openai.routers import embeddings_router, models_router
            from hf_serve.openai.tasks.embeddings import Embeddings

            embeddings = Embeddings(pipeline=predictor.pipeline)
            app.include_router(router=embeddings_router(predictor=embeddings))
            app.include_router(
                router=models_router(model_id=embeddings.model_id, timestamp=int(time.time()))  # type: ignore
            )

            from hf_serve.compatibility.text_embeddings_inference import (
                router as text_embeddings_inference_router,
            )

            app.include_router(router=text_embeddings_inference_router)
        case "text-ranking" | "sentence-ranking":
            from hf_serve.tasks.sentence_transformers.text_ranking import (
                TextRanking,
                TextRankingInput,
                TextRankingOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=TextRanking(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=TextRankingInput,  # type: ignore
                    output_schema=TextRankingOutput,  # type: ignore
                )
            )

            from hf_serve.compatibility.text_embeddings_inference import (
                router as text_embeddings_inference_router,
            )

            app.include_router(router=text_embeddings_inference_router)
        # transformers
        case "text-classification":
            from hf_serve.tasks.transformers.text_classification import (
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
            from hf_serve.tasks.transformers.fill_mask import (
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
            from hf_serve.tasks.transformers.question_answering import (
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
            from hf_serve.tasks.transformers.summarization import (
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
            from hf_serve.tasks.transformers.zero_shot_classification import (
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
            from hf_serve.tasks.transformers.token_classification import (
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
            from hf_serve.tasks.transformers.table_question_answering import (
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
            from hf_serve.tasks.transformers.translation import (
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
            from hf_serve.tasks.transformers.zero_shot_audio_classification import (
                ZeroShotAudioClassification,
                ZeroShotAudioClassificationFormInput,
                ZeroShotAudioClassificationInput,
                ZeroShotAudioClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ZeroShotAudioClassification(
                        model_id=model_id or model_dir,  # type: ignore
                        dtype=dtype,  # type: ignore
                        device=device,  # type: ignore
                    ),
                    input_schema=ZeroShotAudioClassificationInput,
                    input_form_schema=ZeroShotAudioClassificationFormInput,
                    output_schema=ZeroShotAudioClassificationOutput,
                    accepted_mimetypes=accepted_mimetypes or ["audio/*"],
                    max_file_size=max_file_size,
                )
            )
        case "audio-classification":
            from hf_serve.tasks.transformers.audio_classification import (
                AudioClassification,
                AudioClassificationFormInput,
                AudioClassificationInput,
                AudioClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=AudioClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=AudioClassificationInput,
                    input_form_schema=AudioClassificationFormInput,
                    output_schema=AudioClassificationOutput,
                    accepted_mimetypes=accepted_mimetypes or ["audio/*"],
                    max_file_size=max_file_size,
                )
            )
        case "automatic-speech-recognition":
            from hf_serve.tasks.transformers.automatic_speech_recognition import (
                AutomaticSpeechRecognition,
                AutomaticSpeechRecognitionFormInput,
                AutomaticSpeechRecognitionInput,
                AutomaticSpeechRecognitionOutput,
            )

            predictor = AutomaticSpeechRecognition(model_id=model_id or model_dir, dtype=dtype, device=device)  # type: ignore
            app.include_router(
                router=predict_media_router(
                    predictor=predictor,
                    input_schema=AutomaticSpeechRecognitionInput,
                    output_schema=AutomaticSpeechRecognitionOutput,
                    input_form_schema=AutomaticSpeechRecognitionFormInput,
                    accepted_mimetypes=accepted_mimetypes or ["audio/*"],
                    max_file_size=max_file_size,
                )
            )
        # transformers - image
        case "image-classification":
            from hf_serve.tasks.transformers.image_classification import (
                ImageClassification,
                ImageClassificationFormInput,
                ImageClassificationInput,
                ImageClassificationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ImageClassification(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ImageClassificationInput,
                    input_form_schema=ImageClassificationFormInput,
                    output_schema=ImageClassificationOutput,
                    accepted_mimetypes=accepted_mimetypes or ["image/*"],
                    max_file_size=max_file_size,
                )
            )
        case "image-segmentation":
            from hf_serve.tasks.transformers.image_segmentation import (
                ImageSegmentation,
                ImageSegmentationFormInput,
                ImageSegmentationInput,
                ImageSegmentationOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ImageSegmentation(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ImageSegmentationInput,
                    input_form_schema=ImageSegmentationFormInput,
                    output_schema=ImageSegmentationOutput,
                    accepted_mimetypes=accepted_mimetypes or ["image/*"],
                    max_file_size=max_file_size,
                )
            )
        case "object-detection":
            from hf_serve.tasks.transformers.object_detection import (
                ObjectDetection,
                ObjectDetectionFormInput,
                ObjectDetectionInput,
                ObjectDetectionOutput,
            )

            app.include_router(
                router=predict_media_router(
                    predictor=ObjectDetection(model_id=model_id or model_dir, dtype=dtype, device=device),  # type: ignore
                    input_schema=ObjectDetectionInput,
                    input_form_schema=ObjectDetectionFormInput,
                    output_schema=ObjectDetectionOutput,
                    accepted_mimetypes=accepted_mimetypes or ["image/*"],
                    max_file_size=max_file_size,
                )
            )
        case "visual-question-answering" | "vqa":
            from hf_serve.tasks.transformers.visual_question_answering import (
                VisualQuestionAnswering,
                VisualQuestionAnsweringInput,
                VisualQuestionAnsweringOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=VisualQuestionAnswering(
                        model_id=model_id or model_dir,  # type: ignore
                        dtype=dtype,  # type: ignore
                        device=device,  # type: ignore
                    ),
                    input_schema=VisualQuestionAnsweringInput,
                    output_schema=VisualQuestionAnsweringOutput,
                )
            )
        case "zero-shot-image-classification":
            from hf_serve.tasks.transformers.zero_shot_image_classification import (
                ZeroShotImageClassification,
                ZeroShotImageClassificationInput,
                ZeroShotImageClassificationOutput,
            )

            app.include_router(
                router=predict_router(
                    predictor=ZeroShotImageClassification(
                        model_id=model_id or model_dir,  # type: ignore
                        dtype=dtype,  # type: ignore
                        device=device,  # type: ignore
                    ),
                    input_schema=ZeroShotImageClassificationInput,
                    output_schema=ZeroShotImageClassificationOutput,
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

            from hf_serve.tasks.custom import Custom

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

    if cloud is not None and cloud == "azure":
        from hf_serve.compatibility.azure import router as azure_router

        app.include_router(router=azure_router)

    log_available_routes(app=app)

    uvicorn.run(
        "hf_serve.server:app",
        host=host,  # type: ignore
        port=port,  # type: ignore
        log_level=0,
        access_log=False,
        use_colors=True,
        workers=1,
    )
