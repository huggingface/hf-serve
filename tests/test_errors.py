import base64
import unittest
from typing import Literal

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException

from hf_serve.errors import MAX_VALUE_LENGTH, truncate_message
from hf_serve.routers.predict import router as predict_router
from hf_serve.serde.audio import Audio
from hf_serve.serde.image import Image
from hf_serve.server import truncated_http_exception_handler, validation_exception_handler
from hf_serve.tasks.predictor import Predictor

# NOTE: a FLAC header with invalid base64 padding, and a PNG header followed by data that isn't a valid image
AUDIO = "ZkxhQwAAACIQABAAAAhUABfvA+gA8AADV6AH4KsLAGLf55WnLE8wwDFnAwAAJAAAAAAA" * 1000 + "A"
IMAGE = "iVBORw0KGgo" + "A" * 60_000


class MediaInput(BaseModel):
    kind: Literal["audio", "image"]
    inputs: str

    model_config = ConfigDict(json_schema_extra={"examples": []})


class MediaOutput(BaseModel):
    size: int


class Media(Predictor[MediaInput, MediaOutput]):
    def __call__(self, payload: MediaInput) -> MediaOutput:
        if payload.kind == "audio":
            return MediaOutput(size=len(Audio.deserialize(payload.inputs)))
        return MediaOutput(size=Image.deserialize(payload.inputs).width)


class ErrorTruncationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        app = FastAPI()
        app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore
        app.add_exception_handler(StarletteHTTPException, truncated_http_exception_handler)  # type: ignore
        app.include_router(predict_router(Media(), MediaInput, MediaOutput))
        cls.client = TestClient(app)

    def assert_truncated(self, text: str, value: str) -> None:
        self.assertNotIn(value, text)
        self.assertIn(value[:64], text)
        self.assertIn(f"[TRUNCATED, {len(value):,} characters in total]", text)
        self.assertLess(len(text), 1_000)

    def test_invalid_base64_audio(self) -> None:
        response = self.client.post("/predict", json={"kind": "audio", "inputs": AUDIO})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assert_truncated(detail, AUDIO)
        self.assertTrue(
            detail.endswith("is neither a valid base64 string, public URL, nor a valid file system path")
        )

    def test_invalid_base64_image(self) -> None:
        for inputs, echoed in ((IMAGE, IMAGE), (f"data:image/png;base64,{IMAGE}", IMAGE)):
            with self.subTest(data_uri=inputs.startswith("data:")):
                response = self.client.post("/predict", json={"kind": "image", "inputs": inputs})
                self.assertEqual(response.status_code, 400)
                self.assert_truncated(response.json()["detail"], echoed)
                self.assertTrue(response.json()["detail"].startswith("Failed to deserialize image:"))

    def test_short_inputs_are_kept(self) -> None:
        inputs = base64.b64encode(b"not an image").decode()
        response = self.client.post("/predict", json={"kind": "image", "inputs": inputs})
        self.assertEqual(response.status_code, 400)
        self.assertIn(inputs, response.json()["detail"])
        self.assertNotIn("TRUNCATED", response.json()["detail"])

    def test_validation_error(self) -> None:
        response = self.client.post("/predict", json={"kind": "video", "inputs": AUDIO, "extra": "kept"})
        self.assertEqual(response.status_code, 422)
        body = response.json()
        self.assertEqual(body["body"]["kind"], "video")
        self.assertEqual(body["body"]["extra"], "kept")
        self.assert_truncated(body["body"]["inputs"], AUDIO)
        self.assertEqual([error["loc"] for error in body["detail"]], [["body", "kind"]])

        response = self.client.post("/predict", json={"kind": "audio", "inputs": [AUDIO]})
        self.assertEqual(response.status_code, 422)
        self.assert_truncated(response.text, AUDIO)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "inputs"])

    def test_truncate_message_boundaries(self) -> None:
        encoded = "Ab1" * (MAX_VALUE_LENGTH // 3) + "A" * (MAX_VALUE_LENGTH % 3)
        self.assertEqual(truncate_message(f"Input '{encoded}'"), f"Input '{encoded}'")
        self.assertIn("[TRUNCATED, 257 characters in total]", truncate_message(f"Input '{encoded}A'"))
        # NOTE: long paths, URLs, and sentences aren't mistaken for encoded data
        for message in (
            "/".join(["torchcodec"] * 40) + "/SingleStreamDecoder.cpp:123",
            "/".join(["Users", "runner", "work2"] * 30) + "/decoder.cpp",
            "https://huggingface.co/" + "/".join(["model"] * 60) + "?revision=main",
            " ".join(["word"] * 200),
        ):
            with self.subTest(message=message[:30]):
                self.assertEqual(truncate_message(message), message)


if __name__ == "__main__":
    unittest.main()
