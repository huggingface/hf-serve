"""Opt-in tests against published weights; no model downloads during normal test runs.

HF_SERVE_EMBEDDINGGEMMA2_MODEL=/path/to/model python -m unittest discover \
    -s tests -p test_embeddinggemma2_integration.py -v
"""

import base64
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from hf_serve.compatibility.google.routers.predict import router as google_router
from hf_serve.compatibility.google.schemas.sentence_transformers.feature_extraction import (
    FeatureExtractionInputForGoogle,
    FeatureExtractionOutputForGoogle,
)
from hf_serve.openai.routers.embeddings import router as embeddings_router
from hf_serve.openai.tasks.embeddings import Embeddings
from hf_serve.routers.predict import router as predict_router
from hf_serve.tasks.sentence_transformers.feature_extraction import (
    FeatureExtraction,
    FeatureExtractionInput,
    FeatureExtractionOutput,
)


@unittest.skipUnless(
    os.getenv("HF_SERVE_EMBEDDINGGEMMA2_MODEL"), "Set HF_SERVE_EMBEDDINGGEMMA2_MODEL to published weights"
)
class EmbeddingGemma2IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import av
        from scipy.io import wavfile

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(4)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.assets = Path(cls.temporary.name)
        cls.image = cls.assets / "image.png"
        Image.new("RGB", (128, 128), "red").save(cls.image)
        for seconds in (1, 12, 31):
            audio = (0.1 * np.sin(np.arange(16000 * seconds) * 2 * np.pi * 440 / 16000)).astype(np.float32)
            wavfile.write(cls.assets / f"audio-{seconds}.wav", 16000, audio)
        cls.video = cls.assets / "video.mp4"
        with av.open(str(cls.video), mode="w") as output:
            stream = output.add_stream("mpeg4", rate=1)
            stream.width = stream.height = 128
            stream.pix_fmt = "yuv420p"
            for color in ("red", "blue"):
                for packet in stream.encode(av.VideoFrame.from_image(Image.new("RGB", (128, 128), color))):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
        cls.predictor = FeatureExtraction(os.environ["HF_SERVE_EMBEDDINGGEMMA2_MODEL"], device="cpu")
        app = FastAPI()
        app.include_router(predict_router(cls.predictor, FeatureExtractionInput, FeatureExtractionOutput))
        app.include_router(embeddings_router(Embeddings(cls.predictor.pipeline)))
        cls.client = TestClient(app)
        cls.addClassCleanup(cls.client.close)
        google_app = FastAPI()
        google_app.include_router(
            google_router(
                cls.predictor,
                FeatureExtractionInputForGoogle,
                FeatureExtractionOutputForGoogle,
                FeatureExtractionInput,
            )
        )
        cls.google_client = TestClient(google_app)
        cls.addClassCleanup(cls.google_client.close)

    def assert_vectors(self, vectors, count, dimensions):
        array = np.asarray(vectors)
        self.assertEqual(array.shape, (count, dimensions))
        self.assertTrue(np.isfinite(array).all())
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5)

    def predict(self, inputs, **parameters):
        response = self.client.post("/predict", json={"inputs": inputs, **parameters})
        self.assertEqual(response.status_code, 200, response.text[:500])
        return response.json()["embeddings"]

    def test_text_prompts_and_all_supported_dimensions(self):
        for dimensions in (128, 256, 512, 768):
            with self.subTest(dimensions=dimensions):
                self.assert_vectors(
                    self.predict("Find a red square", prompt_name="SearchQuery", dimensions=dimensions),
                    1,
                    dimensions,
                )

    def test_all_media_modalities(self):
        for inputs in (
            {"image": str(self.image)},
            {"audio": str(self.assets / "audio-1.wav")},
            {"video": str(self.video)},
            {"image": [str(self.image), str(self.image)]},
        ):
            with self.subTest(inputs=inputs):
                self.assert_vectors(self.predict(inputs, dimensions=128), 1, 128)

    def test_interleaved_input_and_mixed_batch(self):
        interleaved = {
            "text": "Square <|image|> Sound <|audio|> Clip <|video|>",
            "image": str(self.image),
            "audio": str(self.assets / "audio-1.wav"),
            "video": str(self.video),
        }
        self.assert_vectors(self.predict(interleaved, dimensions=128), 1, 128)
        self.assert_vectors(self.predict(["A red square", {"image": str(self.image)}], dimensions=128), 2, 128)

    def test_openai_media_preprocessed_once_and_matches_predict(self):
        inputs = {"image": str(self.image)}
        expected = self.predict(inputs, dimensions=256)
        with patch.object(
            self.predictor.pipeline, "preprocess", wraps=self.predictor.pipeline.preprocess
        ) as preprocess:
            response = self.client.post(
                "/v1/embeddings",
                json={
                    "model": "google/embeddinggemma-2",
                    "input": inputs,
                    "dimensions": 256,
                    "encoding_format": "base64",
                },
            )
        self.assertEqual(response.status_code, 200, response.text[:500])
        self.assertEqual(preprocess.call_count, 1)
        body = response.json()
        vector = np.frombuffer(base64.b64decode(body["data"][0]["embedding"]), dtype=np.float32)
        self.assert_vectors([vector], 1, 256)
        np.testing.assert_allclose([vector], expected, atol=1e-5)
        self.assertGreater(body["usage"]["prompt_tokens"], 250)
        self.assertEqual(len(self.predictor.pipeline._forward_pre_hooks), 0)

    def test_google_payload(self):
        response = self.google_client.post(
            "/predict", json={"instances": [{"image": str(self.image)}], "parameters": {"dimensions": 128}}
        )
        self.assertEqual(response.status_code, 200, response.text[:500])
        self.assert_vectors(response.json()["predictions"][0]["embeddings"], 1, 128)

    def test_long_audio(self):
        for seconds in (12, 31):
            with self.subTest(seconds=seconds):
                self.assert_vectors(
                    self.predict({"audio": str(self.assets / f"audio-{seconds}.wav")}, dimensions=128), 1, 128
                )

    def test_invalid_requests(self):
        for payload in ({"inputs": {"image": []}}, {"inputs": "hello", "dimensions": 0}):
            self.assertEqual(self.client.post("/predict", json=payload).status_code, 422)
        response = self.client.post(
            "/v1/embeddings", json={"input": "hello", "model": "test", "prompt_name": "unknown"}
        )
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/predict", json={"inputs": {"image": str(self.assets / "missing.png")}})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
