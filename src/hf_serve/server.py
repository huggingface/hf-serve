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
from hf_serve.types.task import TaskTypes

app = FastAPI(title="Hugging Face Serve API")


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
    revision: Union[str, None],
    task: TaskTypes,
    # TODO: maybe we should include `npu` too as supported by `sentence-transformers`?
    device: Optional[Literal["auto", "balanced", "cuda", "cpu", "mps"]] = "auto",
    # TODO: maybe the best default is no default, but handling that separately based on the library as it seems
    # that `float32` is the go to for `sentence-transformers`, `float16` for `diffusers`, and `bfloat16` for
    # `transformers` with some models performing better on `float32` or `float16` too
    dtype: Optional[Literal["float32", "float16", "bfloat16", "float8", "int8", "int4"]] = None,
    trust_remote_code: bool = False,
    accepted_mimetypes: Optional[List[str]] = None,
    max_file_size: Optional[int] = None,
    host: Optional[str] = "0.0.0.0",
    port: Optional[int] = 8080,
    cloud: Optional[Literal["azure", "google"]] = None,
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

    # NOTE: Done this way to prevent for overriding the user-defined value for `--cloud` even if the `AIP_MODE`
    # environment variable is set
    if cloud is None and os.getenv("AIP_MODE") == "PREDICTION":
        logger.info(
            f"Given that the environment variable `API_MODE=PREDICTION`, the `--cloud` arg will be enforced to `google` if none is provided."
        )
        cloud = "google"

    if cloud is None or (cloud is not None and cloud != "google"):
        if any(key.startswith("AIP_") for key, _ in os.environ.items()):
            logger.warning(
                f"`--cloud` is {cloud}, but environment variables starting with `AIP_...` exist, indicating that the cloud provider is most likely Google Cloud, so bear that in mind and provide `--cloud google` if applicable."
            )

    # NOTE: FastAPI runs the middlewares in reverse order
    app.add_middleware(
        middleware_class=PrometheusMiddleware,
        exclude_paths=["/health"]
        if cloud is None or cloud != "google"
        else [os.getenv("AIP_HEALTH_ROUTE", "/health")],
    )
    app.add_middleware(
        middleware_class=LoggingMiddleware,
        # NOTE: temporarily excluding it from the logging as otherwise Inference Endpoints API gets too verbose
        exclude_paths=["/health"]
        if cloud is None or cloud != "google"
        else [os.getenv("AIP_HEALTH_ROUTE", "/health")],
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
    app.add_middleware(
        middleware_class=RequestIdMiddleware,
        exclude_paths=["/health"]
        if cloud is None or cloud != "google"
        else [os.getenv("AIP_HEALTH_ROUTE", "/health")],
    )

    if cloud is not None and cloud == "google":
        from hf_serve.compatibility.google.routers.health import router as google_health_router

        app.include_router(router=google_health_router)
    else:
        app.include_router(router=health_router)

    app.include_router(router=metrics_router)

    if trust_remote_code:
        logger.warning(
            f"You have set `trust_remote_code=True`, which is not recommended, meaning that you will run remote code (if applicable) for `{model_id or model_dir}`. Please make sure that you trust the model author or organization that has created the custom files before proceeding."
        )

    if model_dir:
        logger.info(f"`hf-serve` initializing `model_dir={model_dir}` on `device={device}` for `task={task}`.")
    else:
        logger.info(
            f"`hf-serve` initializing `model_id={model_id}` w/ `revision={revision or 'main'}` on `device={device}` for `task={task}`."
        )

    match task:
        # openai-compatible
        case "image-text-to-text":
            from hf_serve.tasks.transformers.image_text_to_text import (
                ImageTextToText,
                ImageTextToTextInput,
                ImageTextToTextOutput,
            )

            predictor = ImageTextToText(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.image_text_to_text import (
                    ImageTextToTextInputForGoogle,
                    ImageTextToTextOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=ImageTextToTextInputForGoogle,
                        output_schema=ImageTextToTextOutputForGoogle,
                        inner_input_schema=ImageTextToTextInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
                        input_schema=ImageTextToTextInput,
                        output_schema=ImageTextToTextOutput,
                    )
                )
            if predictor.pipeline.tokenizer is not None and (
                predictor.pipeline.tokenizer.chat_template is not None
                # or (
                #     hasattr(predictor.pipeline, "processor")
                #     and predictor.pipeline.processor is not None
                #     and hasattr(predictor.pipeline.processor, "chat_template")
                #     and predictor.pipeline.processor.chat_template is not None
                # )
            ):
                from hf_serve.openai.routers import chat_completions_router, models_router
                from hf_serve.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,  # type: ignore
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

            predictor = TextGeneration(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.text_generation import (
                    TextGenerationInputForGoogle,
                    TextGenerationOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TextGenerationInputForGoogle,
                        output_schema=TextGenerationOutputForGoogle,
                        inner_input_schema=TextGenerationInput,
                    )
                )
            else:
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
                    model=predictor.pipeline.model,  # type: ignore
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
                raise RuntimeError(
                    "Support for `text-to-image` on MPS in any dtype other than FP32 is unstable."
                )

            from hf_serve.routers import predict_image_router
            from hf_serve.tasks.diffusers.text_to_image import TextToImage, TextToImageInput

            predictor = TextToImage(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.diffusers.text_to_image import (
                    TextToImageInputForGoogle,
                    TextToImageOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TextToImageInputForGoogle,
                        output_schema=TextToImageOutputForGoogle,
                        inner_input_schema=TextToImageInput,
                    )
                )
            else:
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

            predictor = SentenceSimilarity(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,  # type: ignore
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.sentence_transformers.sentence_similarity import (
                    SentenceSimilarityInputForGoogle,
                    SentenceSimilarityOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=SentenceSimilarityInputForGoogle,
                        output_schema=SentenceSimilarityOutputForGoogle,
                        inner_input_schema=SentenceSimilarityInput,
                    )
                )
            else:
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

            predictor = FeatureExtraction(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,  # type: ignore
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.sentence_transformers.feature_extraction import (
                    FeatureExtractionInputForGoogle,
                    FeatureExtractionOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=FeatureExtractionInputForGoogle,
                        output_schema=FeatureExtractionOutputForGoogle,
                        inner_input_schema=FeatureExtractionInput,
                    )
                )
            else:
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

            predictor = TextRanking(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,  # type: ignore
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.sentence_transformers.text_ranking import (
                    TextRankingInputForGoogle,
                    TextRankingOutputForGoogle,
                )

                # NOTE: Here we need to patch the existing `TextRanking` predictor given that it's matching
                # on schema types, and given that the Google Cloud compatible schemas are custom, we also need
                # a custom implementation for the predictor
                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TextRankingInputForGoogle,  # type: ignore
                        output_schema=TextRankingOutputForGoogle,  # type: ignore
                        inner_input_schema=TextRankingInput,  # type: ignore
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
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

            predictor = TextClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.text_classification import (
                    TextClassificationInputForGoogle,
                    TextClassificationOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TextClassificationInputForGoogle,
                        output_schema=TextClassificationOutputForGoogle,
                        inner_input_schema=TextClassificationInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
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

            predictor = FillMask(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.fill_mask import (
                    FillMaskInputForGoogle,
                    FillMaskOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=FillMaskInputForGoogle,
                        output_schema=FillMaskOutputForGoogle,
                        inner_input_schema=FillMaskInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
                        input_schema=FillMaskInput,
                        output_schema=FillMaskOutput,
                    )
                )
        case "zero-shot-classification":
            from hf_serve.tasks.transformers.zero_shot_classification import (
                ZeroShotClassification,
                ZeroShotClassificationInput,
                ZeroShotClassificationOutput,
            )

            predictor = ZeroShotClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.zero_shot_classification import (
                    ZeroShotClassificationInputForGoogle,
                    ZeroShotClassificationOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=ZeroShotClassificationInputForGoogle,
                        output_schema=ZeroShotClassificationOutputForGoogle,
                        inner_input_schema=ZeroShotClassificationInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
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

            predictor = TokenClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.token_classification import (
                    TokenClassificationInputForGoogle,
                    TokenClassificationOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TokenClassificationInputForGoogle,
                        output_schema=TokenClassificationOutputForGoogle,
                        inner_input_schema=TokenClassificationInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
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

            predictor = TableQuestionAnswering(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.table_question_answering import (
                    TableQuestionAnsweringInputForGoogle,
                    TableQuestionAnsweringOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=TableQuestionAnsweringInputForGoogle,
                        output_schema=TableQuestionAnsweringOutputForGoogle,
                        inner_input_schema=TableQuestionAnsweringInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
                        input_schema=TableQuestionAnsweringInput,
                        output_schema=TableQuestionAnsweringOutput,
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

            predictor = ZeroShotAudioClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    raise RuntimeError(
                        f"Provided `{cloud=}` for `{task=}` but it's not yet supported on Google Cloud nor Vertex AI."
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=ZeroShotAudioClassificationInput,
                            output_schema=ZeroShotAudioClassificationOutput,
                        )
                    )
                case _:
                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
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

            predictor = AudioClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.audio_classification import (
                        AudioClassificationInputForGoogle,
                        AudioClassificationOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=AudioClassificationInputForGoogle,
                            output_schema=AudioClassificationOutputForGoogle,
                            inner_input_schema=AudioClassificationInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=AudioClassificationInput,
                            output_schema=AudioClassificationOutput,
                        )
                    )
                case _:
                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
                            input_schema=AudioClassificationInput,
                            input_form_schema=AudioClassificationFormInput,
                            output_schema=AudioClassificationOutput,
                            accepted_mimetypes=accepted_mimetypes or ["audio/*"],
                            max_file_size=max_file_size,
                        )
                    )
        case "asr" | "automatic-speech-recognition":
            from hf_serve.tasks.transformers.automatic_speech_recognition import (
                AutomaticSpeechRecognition,
                AutomaticSpeechRecognitionFormInput,
                AutomaticSpeechRecognitionInput,
                AutomaticSpeechRecognitionOutput,
            )

            predictor = AutomaticSpeechRecognition(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.automatic_speech_recognition import (
                        AutomaticSpeechRecognitionInputForGoogle,
                        AutomaticSpeechRecognitionOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=AutomaticSpeechRecognitionInputForGoogle,
                            output_schema=AutomaticSpeechRecognitionOutputForGoogle,
                            inner_input_schema=AutomaticSpeechRecognitionInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=AutomaticSpeechRecognitionInput,
                            output_schema=AutomaticSpeechRecognitionOutput,
                        )
                    )
                case _:
                    from hf_serve.tasks.transformers.automatic_speech_recognition import (
                        AutomaticSpeechRecognitionFormInput,
                    )

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

            predictor = ImageClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.image_classification import (
                        ImageClassificationInputForGoogle,
                        ImageClassificationOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=ImageClassificationInputForGoogle,
                            output_schema=ImageClassificationOutputForGoogle,
                            inner_input_schema=ImageClassificationInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=ImageClassificationInput,
                            output_schema=ImageClassificationOutput,
                        )
                    )
                case _:
                    from hf_serve.tasks.transformers.image_classification import (
                        ImageClassificationFormInput,
                    )

                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
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

            predictor = ImageSegmentation(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.image_segmentation import (
                        ImageSegmentationInputForGoogle,
                        ImageSegmentationOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=ImageSegmentationInputForGoogle,
                            output_schema=ImageSegmentationOutputForGoogle,
                            inner_input_schema=ImageSegmentationInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=ImageSegmentationInput,
                            output_schema=ImageSegmentationOutput,
                        )
                    )
                case _:
                    from hf_serve.tasks.transformers.image_segmentation import (
                        ImageSegmentationFormInput,
                    )

                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
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

            predictor = ObjectDetection(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.object_detection import (
                        ObjectDetectionInputForGoogle,
                        ObjectDetectionOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=ObjectDetectionInputForGoogle,
                            output_schema=ObjectDetectionOutputForGoogle,
                            inner_input_schema=ObjectDetectionInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=ObjectDetectionInput,
                            output_schema=ObjectDetectionOutput,
                        )
                    )
                case _:
                    from hf_serve.tasks.transformers.object_detection import ObjectDetectionFormInput

                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
                            input_schema=ObjectDetectionInput,
                            input_form_schema=ObjectDetectionFormInput,
                            output_schema=ObjectDetectionOutput,
                            accepted_mimetypes=accepted_mimetypes or ["image/*"],
                            max_file_size=max_file_size,
                        )
                    )
        case "zero-shot-image-classification":
            from hf_serve.tasks.transformers.zero_shot_image_classification import (
                ZeroShotImageClassification,
                ZeroShotImageClassificationInput,
                ZeroShotImageClassificationOutput,
            )

            predictor = ZeroShotImageClassification(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.zero_shot_image_classification import (
                    ZeroShotImageClassificationInputForGoogle,
                    ZeroShotImageClassificationOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=ZeroShotImageClassificationInputForGoogle,
                        output_schema=ZeroShotImageClassificationOutputForGoogle,
                        inner_input_schema=ZeroShotImageClassificationInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
                        input_schema=ZeroShotImageClassificationInput,
                        output_schema=ZeroShotImageClassificationOutput,
                    )
                )
        case "mask-generation":
            from hf_serve.tasks.transformers.mask_generation import (
                MaskGeneration,
                MaskGenerationInput,
                MaskGenerationOutput,
            )

            predictor = MaskGeneration(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            match cloud:
                case "google":
                    from hf_serve.compatibility.google.routers.predict import (
                        router as google_predict_router,
                    )
                    from hf_serve.compatibility.google.schemas.transformers.mask_generation import (
                        MaskGenerationInputForGoogle,
                        MaskGenerationOutputForGoogle,
                    )

                    app.include_router(
                        router=google_predict_router(
                            predictor=predictor,
                            input_schema=MaskGenerationInputForGoogle,
                            output_schema=MaskGenerationOutputForGoogle,
                            inner_input_schema=MaskGenerationInput,
                        )
                    )
                case "azure":
                    app.include_router(
                        router=predict_router(
                            predictor=predictor,
                            input_schema=MaskGenerationInput,
                            output_schema=MaskGenerationOutput,
                        )
                    )
                case _:
                    from hf_serve.tasks.transformers.mask_generation import MaskGenerationFormInput

                    app.include_router(
                        router=predict_media_router(
                            predictor=predictor,
                            input_schema=MaskGenerationInput,
                            input_form_schema=MaskGenerationFormInput,
                            output_schema=MaskGenerationOutput,
                            accepted_mimetypes=accepted_mimetypes or ["image/*"],
                            max_file_size=max_file_size,
                        )
                    )
        # NOTE: Support for `any-to-any` is limited and still experimental
        # TODO: Any to any usually also implies support for ASR when there's an audio processor
        case "any-to-any":
            from hf_serve.tasks.transformers.any_to_any import AnyToAny, AnyToAnyInput, AnyToAnyOutput

            predictor = AnyToAny(
                model_id=model_id or model_dir,  # type: ignore
                revision=revision,
                dtype=dtype,
                device=device,  # type: ignore
                trust_remote_code=trust_remote_code,
            )

            if cloud is not None and cloud == "google":
                from hf_serve.compatibility.google.routers.predict import (
                    router as google_predict_router,
                )
                from hf_serve.compatibility.google.schemas.transformers.any_to_any import (
                    AnyToAnyInputForGoogle,
                    AnyToAnyOutputForGoogle,
                )

                app.include_router(
                    router=google_predict_router(
                        predictor=predictor,
                        input_schema=AnyToAnyInputForGoogle,
                        output_schema=AnyToAnyOutputForGoogle,
                        inner_input_schema=AnyToAnyInput,
                    )
                )
            else:
                app.include_router(
                    router=predict_router(
                        predictor=predictor,
                        input_schema=AnyToAnyInput,
                        output_schema=AnyToAnyOutput,
                    )
                )
            if predictor.pipeline.tokenizer is not None and (
                predictor.pipeline.tokenizer.chat_template is not None
                or (
                    hasattr(predictor.pipeline, "processor")
                    and predictor.pipeline.processor is not None
                    and hasattr(predictor.pipeline.processor, "chat_template")
                    and predictor.pipeline.processor.chat_template is not None  # type: ignore
                )
            ):
                from hf_serve.openai.routers import chat_completions_router, models_router
                from hf_serve.openai.tasks.chat_completions import ChatCompletions

                chat_completions = ChatCompletions(
                    model=predictor.pipeline.model,  # type: ignore
                    tokenizer=predictor.pipeline.tokenizer,  # type: ignore
                    processor=predictor.pipeline.processor,  # type: ignore
                )
                app.include_router(router=chat_completions_router(predictor=chat_completions))
                app.include_router(
                    router=models_router(model_id=chat_completions.model_id, timestamp=int(time.time()))  # type: ignore
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
                model_dir = snapshot_download(repo_id=model_id, repo_type="model", revision=revision)

            try:
                app.include_router(router=custom_router(handler=Custom.load(model_dir)))  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    f"Attempted to load a custom handler i.e. the class `{os.getenv('CUSTOM_HANDLER')}` from `{model_dir=}/{os.getenv('CUSTOM_HANDLER_FILE')}`, but either didn't find any or couldn't load it (as per \"{e}\")"
                )
        case _:
            raise ValueError(f"{task=} not supported!")

    if cloud is not None and cloud == "azure":
        from hf_serve.compatibility.azure import router as azure_router

        app.include_router(router=azure_router)

    if model_dir:
        logger.info(f"Loaded `model_dir={model_dir}` on `device={device}` for `task={task}`.")
    else:
        logger.info(
            f"Loaded `model_id={model_id}` w/ `revision={revision or 'main'}` on `device={device}` for `task={task}`."
        )

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
