import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hf_serve.compatibility.google.routers.predict import router as google_predict_router
from hf_serve.compatibility.google.schemas.transformers.speaker_diarization import (
    SpeakerDiarizationInputForGoogle,
    SpeakerDiarizationOutputForGoogle,
)
from hf_serve.tasks.transformers.speaker_diarization import (
    SpeakerDiarizationInput,
    SpeakerDiarizationOutput,
    SpeakerSegment,
)


class FakePredictor:
    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        return SpeakerDiarizationOutput(
            segments=[SpeakerSegment.model_validate({"Start": 0.1, "End": 0.5, "Speaker": 0})]
        )


class GoogleSpeakerDiarizationTests(unittest.TestCase):
    def test_instances_share_parameters_and_return_predictions(self):
        predictor = FakePredictor()
        app = FastAPI()
        app.include_router(
            google_predict_router(
                predictor=predictor,
                input_schema=SpeakerDiarizationInputForGoogle,
                output_schema=SpeakerDiarizationOutputForGoogle,
                inner_input_schema=SpeakerDiarizationInput,
            )
        )

        response = TestClient(app).post(
            "/predict",
            json={
                "instances": ["audio-one", "audio-two"],
                "parameters": {"streaming_mode": "low_latency"},
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predictions": [
                    {"segments": [{"start": 0.1, "end": 0.5, "speaker": 0}]},
                    {"segments": [{"start": 0.1, "end": 0.5, "speaker": 0}]},
                ]
            },
        )
        self.assertEqual([payload.inputs for payload in predictor.payloads], ["audio-one", "audio-two"])
        self.assertTrue(all(payload.parameters.streaming_mode == "low_latency" for payload in predictor.payloads))


if __name__ == "__main__":
    unittest.main()
