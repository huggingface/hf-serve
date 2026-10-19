import base64
import io
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from hf_serve.openai.routers.images_generations import router as images_generations_router
from hf_serve.openai.tasks.images_generations import ImagesGenerations
from hf_serve.tasks.diffusers.text_to_image import (
    TextToImage,
    is_modular_pipeline,
    modular_external_components,
)


class FakeModularPipeline:
    def __init__(self, revised_prompt: str) -> None:
        self.revised_prompt = revised_prompt
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        images = [Image.new("RGB", (32, 16)) for _ in range(kwargs["num_images_per_prompt"])]
        return {"images": images, "prompt": self.revised_prompt}


def modular_text_to_image(pipeline: FakeModularPipeline) -> TextToImage:
    predictor = TextToImage.__new__(TextToImage)
    predictor.pipeline = pipeline
    predictor.is_modular = True
    return predictor


class IsModularPipelineTest(unittest.TestCase):
    def test_local_directory_layouts(self) -> None:
        cases = {
            ("modular_model_index.json",): True,
            ("model_index.json",): False,
            ("modular_model_index.json", "model_index.json"): False,
            (): False,
        }
        for files, expected in cases.items():
            with self.subTest(files=files), tempfile.TemporaryDirectory() as tmp:
                for name in files:
                    (Path(tmp) / name).write_text("{}")
                self.assertEqual(is_modular_pipeline(tmp), expected)


class ModularExternalComponentsTest(unittest.TestCase):
    def test_components_declared_in_another_repository(self) -> None:
        def spec(repository: str, subfolder: str) -> list:
            return [
                "transformers",
                "Model",
                {"pretrained_model_name_or_path": repository, "subfolder": subfolder},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            # NOTE: weights at the root of the copy are what makes `ModularPipeline.from_pretrained` wrongly
            # resolve the components without a `subfolder` to the local copy
            (Path(tmp) / "model.safetensors").write_bytes(b"")
            (Path(tmp) / "transformer").mkdir()
            (Path(tmp) / "modular_model_index.json").write_text(
                json.dumps(
                    {
                        "_class_name": "Pipeline",
                        "transformer": spec("org/model", "transformer"),
                        "text_encoder": spec("org/model", "text_encoder"),
                        "prompt_upsampler": spec("org/upsampler", ""),
                        "prompt_upsampler_processor": spec("org/upsampler", ""),
                    }
                )
            )

            external = modular_external_components(tmp)

        self.assertEqual(set(external), {"prompt_upsampler", "prompt_upsampler_processor"})
        self.assertEqual(external["prompt_upsampler"]["pretrained_model_name_or_path"], "org/upsampler")


class ModularImagesGenerationsTest(unittest.TestCase):
    def client(self, pipeline: FakeModularPipeline) -> TestClient:
        app = FastAPI()
        app.include_router(
            images_generations_router(ImagesGenerations(predictor=modular_text_to_image(pipeline)))
        )
        return TestClient(app)

    def test_returns_images_with_revised_prompt(self) -> None:
        pipeline = FakeModularPipeline(revised_prompt="A detailed caption of a fox")
        response = self.client(pipeline).post(
            "/v1/images/generations",
            json={"model": None, "prompt": "a fox", "n": 2, "response_format": "b64_json"},
        )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(len(data), 2)
        self.assertEqual({item["revised_prompt"] for item in data}, {"A detailed caption of a fox"})
        image = Image.open(io.BytesIO(base64.b64decode(data[0]["b64_json"])))
        self.assertEqual(image.size, (32, 16))
        self.assertEqual(pipeline.calls[0]["output"], ["images", "prompt"])

    def test_omits_revised_prompt_when_unchanged_and_defaults_n(self) -> None:
        pipeline = FakeModularPipeline(revised_prompt="a fox")
        response = self.client(pipeline).post(
            "/v1/images/generations", json={"model": None, "prompt": "a fox", "response_format": "b64_json"}
        )

        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()["data"]
        self.assertEqual(len(data), 1)
        self.assertNotIn("revised_prompt", data[0])
        self.assertEqual(pipeline.calls[0]["num_images_per_prompt"], 1)


if __name__ == "__main__":
    unittest.main()
