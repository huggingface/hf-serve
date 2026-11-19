import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hf_serve.compatibility.azure import router as azure_router
from hf_serve.compatibility.google.routers.predict import router as google_predict_router
from hf_serve.compatibility.google.schemas.transformers.any_to_any import (
    AnyToAnyInputForGoogle,
    AnyToAnyOutputForGoogle,
)
from hf_serve.compatibility.google.schemas.transformers.automatic_speech_recognition import (
    AutomaticSpeechRecognitionInputForGoogle,
    AutomaticSpeechRecognitionOutputForGoogle,
)
from hf_serve.compatibility.sagemaker import SageMakerRoutingMiddleware, post_paths
from hf_serve.routers.predict import router as predict_router
from hf_serve.routers.predict_media import media_router
from hf_serve.tasks.transformers.any_to_any import AnyToAny, AnyToAnyInput, AnyToAnyOutput
from hf_serve.tasks.transformers.automatic_speech_recognition import (
    AutomaticSpeechRecognition,
    AutomaticSpeechRecognitionFormInput,
    AutomaticSpeechRecognitionGenerationParameters,
    AutomaticSpeechRecognitionInput,
    AutomaticSpeechRecognitionOutput,
)


def blocked_parameters():
    for key in ("custom_generate", "trust_remote_code"):
        for value in ("org/custom-code", True, False, None):
            for wrap in (
                lambda data: data,
                lambda data: {"generate_kwargs": data},
                lambda data: {"generate_kwargs": {"generate_kwargs": data}},
                lambda data: {"options": [data]},
            ):
                yield key, wrap({key: value})


def any_to_any_predictor():
    predictor = AnyToAny.__new__(AnyToAny)
    predictor.pipeline = Mock(return_value=[{"generated_text": [{"content": "hello"}]}])
    return predictor


def asr_predictor():
    predictor = AutomaticSpeechRecognition.__new__(AutomaticSpeechRecognition)
    predictor.pipeline = Mock(return_value={"text": "hello"})
    predictor.chunk_length_s = 30
    return predictor


class GenerationParameterSchemaTests(unittest.TestCase):
    def test_remote_code_keys_are_rejected_by_native_and_google_schemas(self):
        for key, parameters in blocked_parameters():
            for schema, data in (
                (AnyToAnyInput, {"inputs": {"text": "hello"}, "parameters": parameters}),
                (AnyToAnyInputForGoogle, {"instances": ["hello"], "parameters": parameters}),
                (AutomaticSpeechRecognitionGenerationParameters, parameters),
                (
                    AutomaticSpeechRecognitionInput,
                    {"audio": "aGVsbG8=", "parameters": {"generation_parameters": parameters}},
                ),
                (
                    AutomaticSpeechRecognitionInputForGoogle,
                    {"instances": ["aGVsbG8="], "parameters": {"generation_parameters": parameters}},
                ),
            ):
                with self.subTest(schema=schema.__name__, parameters=parameters):
                    with self.assertRaisesRegex(ValidationError, key):
                        schema.model_validate(data)

    def test_model_specific_parameters_are_preserved(self):
        parameters = {
            "language": "swedish",
            "task": "transcribe",
            "return_timestamps": True,
            "condition_on_prev_tokens": False,
            "max_new_tokens": 32,
            "generate_kwargs": {"future_model_option": [1, {"temperature": 0.2}]},
        }
        for schema, data, get_parameters in (
            (AnyToAnyInput, {"inputs": {"text": "hello"}}, lambda model: model.parameters),
            (AnyToAnyInputForGoogle, {"instances": ["hello"]}, lambda model: model.parameters),
            (
                AutomaticSpeechRecognitionGenerationParameters,
                {},
                lambda model: model.model_dump(exclude_none=True),
            ),
        ):
            with self.subTest(schema=schema.__name__):
                payload = schema.model_validate(
                    parameters
                    if schema is AutomaticSpeechRecognitionGenerationParameters
                    else {**data, "parameters": parameters}
                )
                self.assertEqual(get_parameters(payload), parameters)

    def test_missing_and_empty_parameters_still_work(self):
        for parameters in (None, {}):
            AnyToAnyInput(inputs={"text": "hello"}, parameters=parameters)
            AutomaticSpeechRecognitionInput(inputs=b"audio", parameters=parameters)


class GenerationParameterInferenceTests(unittest.TestCase):
    def test_any_to_any_forwards_options_without_mutating_payload(self):
        from transformers import set_seed

        predictor = any_to_any_predictor()
        parameters = {"seed": 42, "generate_kwargs": {"max_new_tokens": 32, "future_option": True}}
        payload = AnyToAnyInput(inputs={"text": "hello", "audio": "audio.wav"}, parameters=parameters)
        original = copy.deepcopy(payload.model_dump())
        with patch("transformers.set_seed", spec=set_seed) as mocked_set_seed:
            self.assertEqual(predictor(payload).generated_text, "hello")
        mocked_set_seed.assert_called_once_with(42)
        self.assertEqual(
            predictor.pipeline.call_args.kwargs, {"generate_kwargs": parameters["generate_kwargs"]}
        )
        self.assertEqual(payload.model_dump(), original)

    def test_asr_forwards_whisper_options(self):
        predictor = asr_predictor()
        parameters = {"language": "swedish", "task": "transcribe", "condition_on_prev_tokens": False}
        payload = AutomaticSpeechRecognitionInput(
            inputs=b"audio", parameters={"return_timestamps": True, "generation_parameters": parameters}
        )
        with patch(
            "hf_serve.tasks.transformers.automatic_speech_recognition.AudioSegment.from_file",
            return_value=SimpleNamespace(duration_seconds=2),
        ):
            self.assertEqual(predictor(payload).text, "hello")
        predictor.pipeline.assert_called_once_with(
            b"audio", return_timestamps=True, chunk_length_s=30, batch_size=1, **parameters
        )

    def test_mutated_payloads_are_rejected_before_pipeline_or_audio_loading(self):
        for key, parameters in blocked_parameters():
            any_predictor = any_to_any_predictor()
            any_payload = AnyToAnyInput(inputs={"text": "hello"}, parameters={})
            any_payload.parameters.update(parameters)
            speech_predictor = asr_predictor()
            speech_payload = AutomaticSpeechRecognitionInput(
                inputs="audio.wav", parameters={"generation_parameters": {}}
            )
            # Extras are mutable too, so schema validation alone is insufficient.
            speech_payload.parameters.generation_parameters.model_extra.update(parameters)
            for predictor, payload in ((any_predictor, any_payload), (speech_predictor, speech_payload)):
                with self.subTest(task=type(predictor).__name__, parameters=parameters):
                    with patch(
                        "hf_serve.tasks.transformers.automatic_speech_recognition.Audio.deserialize"
                    ) as decode:
                        with self.assertRaisesRegex(ValueError, key):
                            predictor(payload)
                        decode.assert_not_called()
                        predictor.pipeline.assert_not_called()

    def test_unvalidated_payloads_are_rejected(self):
        predictor = any_to_any_predictor()
        payload = AnyToAnyInput.model_construct(
            inputs=AnyToAnyInput(inputs={"text": "hello"}).inputs,
            parameters={"generate_kwargs": {"custom_generate": "org/custom-code"}},
        )
        with self.assertRaisesRegex(ValueError, "custom_generate"):
            predictor(payload)
        predictor.pipeline.assert_not_called()


class GenerationParameterHTTPTests(unittest.TestCase):
    def test_google_any_to_any_rejects_remote_code_before_inference(self):
        predictor = any_to_any_predictor()
        app = FastAPI()
        app.include_router(
            google_predict_router(predictor, AnyToAnyInputForGoogle, AnyToAnyOutputForGoogle, AnyToAnyInput)
        )
        client = TestClient(app)
        for key, parameters in blocked_parameters():
            response = client.post("/predict", json={"instances": ["hello"], "parameters": parameters})
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn(key, response.text)
        predictor.pipeline.assert_not_called()

    def test_standard_azure_and_sagemaker_any_to_any_routes_reject_remote_code(self):
        predictor = any_to_any_predictor()
        app = FastAPI()
        with patch.object(predictor, "warmup"):
            app.include_router(predict_router(predictor, AnyToAnyInput, AnyToAnyOutput))
        app.include_router(azure_router)
        app.add_middleware(SageMakerRoutingMiddleware, invocation_paths=post_paths(app.routes))
        client = TestClient(app)
        for route in ("/", "/predict", "/score", "/invocations"):
            for key, parameters in blocked_parameters():
                with self.subTest(route=route, parameters=parameters):
                    response = client.post(route, json={"inputs": {"text": "hello"}, "parameters": parameters})
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn(key, response.text)
        predictor.pipeline.assert_not_called()
        for route in ("/predict", "/score", "/invocations"):
            response = client.post(
                route, json={"inputs": {"text": "hello"}, "parameters": {"future_option": True}}
            )
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(predictor.pipeline.call_args.kwargs, {"future_option": True})

    def test_asr_media_and_sagemaker_routes_reject_remote_code(self):
        predictor = asr_predictor()
        app = FastAPI()
        app.include_router(
            media_router(
                predictor,
                AutomaticSpeechRecognitionInput,
                AutomaticSpeechRecognitionFormInput,
                AutomaticSpeechRecognitionOutput,
                accepted_mimetypes=["audio/*"],
            )
        )
        app.add_middleware(SageMakerRoutingMiddleware, invocation_paths=post_paths(app.routes))
        client = TestClient(app)
        for route in ("/predict-json", "/invocations"):
            for key, parameters in blocked_parameters():
                with self.subTest(route=route, parameters=parameters):
                    response = client.post(
                        route, json={"inputs": "aGVsbG8=", "parameters": {"generation_parameters": parameters}}
                    )
                    self.assertEqual(response.status_code, 422, response.text)
                    self.assertIn(key, response.text)
        predictor.pipeline.assert_not_called()

    def test_google_asr_validation_and_whisper_options(self):
        predictor = asr_predictor()
        app = FastAPI()
        app.include_router(
            google_predict_router(
                predictor,
                AutomaticSpeechRecognitionInputForGoogle,
                AutomaticSpeechRecognitionOutputForGoogle,
                AutomaticSpeechRecognitionInput,
            )
        )
        client = TestClient(app)
        for key, parameters in blocked_parameters():
            response = client.post(
                "/predict",
                json={"instances": ["aGVsbG8="], "parameters": {"generation_parameters": parameters}},
            )
            self.assertEqual(response.status_code, 422, response.text)
            self.assertIn(key, response.text)
        predictor.pipeline.assert_not_called()
        with patch(
            "hf_serve.tasks.transformers.automatic_speech_recognition.AudioSegment.from_file",
            return_value=SimpleNamespace(duration_seconds=2),
        ):
            response = client.post(
                "/predict",
                json={
                    "instances": ["aGVsbG8="],
                    "parameters": {"generation_parameters": {"language": "swedish", "task": "transcribe"}},
                },
            )
        self.assertEqual(response.status_code, 200, response.text)
        call_kwargs = predictor.pipeline.call_args.kwargs
        self.assertEqual(call_kwargs["language"], "swedish")
        self.assertEqual(call_kwargs["task"], "transcribe")
        self.assertNotIn("generate_kwargs", call_kwargs)


if __name__ == "__main__":
    unittest.main()
