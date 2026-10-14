import base64
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import Mock, patch

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer

from hf_serve.compatibility.google.routers.predict import router as google_predict_router
from hf_serve.compatibility.google.schemas.sentence_transformers.feature_extraction import (
    FeatureExtractionInputForGoogle,
    FeatureExtractionOutputForGoogle,
)
from hf_serve.openai.routers.embeddings import router as embeddings_router
from hf_serve.openai.schemas.embeddings import EmbeddingsInput
from hf_serve.openai.tasks.embeddings import Embeddings
from hf_serve.routers.predict import router as predict_router
from hf_serve.tasks.sentence_transformers.feature_extraction import (
    FeatureExtraction,
    FeatureExtractionInput,
    FeatureExtractionOutput,
)
from hf_serve.tasks.sentence_transformers.loading import resolve_embedding_dtype


class RecordingPipeline(torch.nn.Module):
    """Exercise PyTorch's real forward hooks while keeping the encoder lightweight."""

    def __init__(self):
        super().__init__()
        self.default_prompt_name = None
        self.prompts = {"Document": "title: none | text: "}
        self.preprocess = Mock(side_effect=self._preprocess)
        self.encode = Mock(side_effect=self._encode)

    def _preprocess(self, inputs, **kwargs):
        return {"input_ids": torch.ones(len(inputs), 7), "attention_mask": torch.ones(len(inputs), 7)}

    def _encode(self, inputs, **kwargs):
        prompt_name = kwargs.get("prompt_name", self.default_prompt_name)
        prompt = self.prompts[prompt_name] if prompt_name is not None else None
        batch = inputs if isinstance(inputs, list) else [inputs]
        features = self.preprocess(batch, prompt=prompt)
        self(features)
        return self.encode.return_value

    def forward(self, features):
        return features


class EmbeddingSchemaTests(unittest.TestCase):
    def test_text_and_multimodal_inputs(self):
        for value in [
            "hello",
            ["hello", "world"],
            {"image": "photo.png"},
            {"audio": ["first.wav", "second.wav"]},
            {"video": "clip.mp4"},
            {"text": "Look <|image|>", "image": ["a.png", "b.png"]},
            [{"image": "a.png"}, {"image": "b.png"}],
        ]:
            with self.subTest(value=value):
                self.assertEqual(FeatureExtractionInput(inputs=value).sentences, value)
                self.assertEqual(EmbeddingsInput(input=value, model="test").input_, value)
                self.assertEqual(FeatureExtractionInputForGoogle(instances=[value]).instances, [value])

    def test_invalid_inputs(self):
        for value in [
            "",
            [],
            [""],
            {},
            {"image": []},
            {"image": ""},
            {"unknown": "test"},
            {"audio": [1, 2]},
            [{"image": ""}],
        ]:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    FeatureExtractionInput(inputs=value)
                with self.assertRaises(ValidationError):
                    EmbeddingsInput(input=value, model="test")

    def test_aliases_and_modality_order(self):
        value = {"image": "a.png", "text": "caption", "audio": "a.wav"}
        for key in ["inputs", "sentences"]:
            payload = FeatureExtractionInput.model_validate(
                {key: value, "parameters": {"prompt_name": "Document", "dimensions": 128}}
            )
            self.assertEqual(list(payload.model_dump()["sentences"]), list(value))
            self.assertEqual(payload.prompt_name, "Document")
            self.assertEqual(payload.dimensions, 128)

    def test_dimensions_must_be_positive(self):
        for dimensions in [0, -1]:
            with self.assertRaises(ValidationError):
                FeatureExtractionInput(inputs="hello", dimensions=dimensions)
            with self.assertRaises(ValidationError):
                EmbeddingsInput(input="hello", model="test", dimensions=dimensions)


class EmbeddingInferenceTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = RecordingPipeline()
        self.predictor = FeatureExtraction.__new__(FeatureExtraction)
        self.predictor.pipeline = self.pipeline

    def test_truncation_is_normalized_by_default(self):
        self.pipeline.encode.return_value = torch.tensor([0.6, 0.8])
        result = self.predictor(FeatureExtractionInput(inputs="hello", dimensions=2))
        self.assertEqual(len(result.embeddings), 1)
        self.pipeline.encode.assert_called_once_with(
            "hello", truncate_dim=2, normalize_embeddings=True, convert_to_tensor=True
        )

    def test_normalization_can_be_disabled(self):
        self.pipeline.encode.return_value = torch.ones(1, 2)
        self.predictor(FeatureExtractionInput(inputs={"image": "photo.png"}, normalize=False))
        self.assertFalse(self.pipeline.encode.call_args.kwargs["normalize_embeddings"])
        self.assertEqual(self.pipeline.encode.call_args.args[0], {"image": "photo.png"})

    def test_openai_multimodal_batch_prompt_usage_and_base64(self):
        vectors = np.array([[0.6, 0.8], [0.8, 0.6]], dtype=np.float32)
        self.pipeline.encode.return_value = vectors
        values = [{"text": "first"}, {"text": "second", "image": "a.png"}]
        result = Embeddings(self.pipeline)(
            EmbeddingsInput(
                input=values, model="test", prompt_name="Document", dimensions=2, encoding_format="base64"
            )
        )
        self.assertEqual(result.usage.prompt_tokens, 14)
        self.assertEqual(result.usage.total_tokens, 14)
        self.assertEqual([item.index for item in result.data], [0, 1])
        np.testing.assert_array_equal(
            np.frombuffer(base64.b64decode(result.data[0].embedding), dtype=np.float32), vectors[0]
        )
        self.pipeline.preprocess.assert_called_once_with(values, prompt="title: none | text: ")
        self.pipeline.encode.assert_called_once_with(
            values, prompt_name="Document", truncate_dim=2, convert_to_numpy=True, normalize_embeddings=True
        )

    def test_openai_single_text_vector(self):
        self.pipeline.encode.return_value = np.array([0.6, 0.8], dtype=np.float32)
        result = Embeddings(self.pipeline)(EmbeddingsInput(input="hello", model="test"))
        self.assertEqual(len(result.data), 1)
        self.assertEqual(len(result.data[0].embedding), 2)
        self.assertEqual(result.usage.prompt_tokens, 7)

    def test_openai_usage_includes_default_prompt(self):
        self.pipeline.default_prompt_name = "Document"
        self.pipeline.encode.return_value = np.ones(2, dtype=np.float32)
        Embeddings(self.pipeline)(EmbeddingsInput(input="hello", model="test"))
        self.pipeline.preprocess.assert_called_once_with(["hello"], prompt="title: none | text: ")

    def test_usage_excludes_padding(self):
        self.pipeline.preprocess.side_effect = None
        self.pipeline.preprocess.return_value = {
            "input_ids": torch.ones(2, 5),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]]),
        }
        self.pipeline.encode.return_value = np.ones((2, 2), dtype=np.float32)
        result = Embeddings(self.pipeline)(EmbeddingsInput(input=["long", "short"], model="test"))
        self.assertEqual(result.usage.prompt_tokens, 7)
        self.pipeline.preprocess.assert_called_once()

    def test_overlapping_requests_have_independent_usage(self):
        barrier = Barrier(2)

        def preprocess(inputs, **kwargs):
            barrier.wait(timeout=5)
            count = int(inputs[0]["image"])
            return {"input_ids": torch.ones(1, count), "attention_mask": torch.ones(1, count)}

        self.pipeline.preprocess.side_effect = preprocess
        self.pipeline.encode.return_value = np.ones(2, dtype=np.float32)
        predictor = Embeddings(self.pipeline)
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(
                executor.map(
                    lambda count: predictor(EmbeddingsInput(input={"image": str(count)}, model="test")), [3, 7]
                )
            )
        self.assertEqual([result.usage.prompt_tokens for result in results], [3, 7])
        self.assertEqual(len(self.pipeline._forward_pre_hooks), 0)

    def test_hooks_are_removed_when_encoding_fails(self):
        self.pipeline.encode.side_effect = ValueError("Inference failed")
        with self.assertRaisesRegex(ValueError, "Inference failed"):
            Embeddings(self.pipeline)(EmbeddingsInput(input="hello", model="test"))
        self.assertEqual(len(self.pipeline._forward_pre_hooks), 0)

    def test_hook_removal_during_another_requests_forward(self):
        first_ready, second_forward, first_finished = Event(), Event(), Event()

        def preprocess(inputs, **kwargs):
            count = int(inputs[0]["image"])
            if count == 3:
                first_ready.set()
                self.assertTrue(second_forward.wait(timeout=5))
            else:
                self.assertTrue(first_ready.wait(timeout=5))
            return {"input_ids": torch.ones(1, count), "attention_mask": torch.ones(1, count)}

        def pause_second_forward(_module, args):
            if args[0]["input_ids"].shape[-1] == 7:
                # PyTorch has already captured the hook list. Let the first request
                # finish and remove its hook before this forward visits that entry.
                second_forward.set()
                self.assertTrue(first_finished.wait(timeout=5))

        self.pipeline.preprocess.side_effect = preprocess
        self.pipeline.encode.return_value = np.ones(2, dtype=np.float32)
        predictor = Embeddings(self.pipeline)

        def request(count):
            try:
                return predictor(EmbeddingsInput(input={"image": str(count)}, model="test"))
            finally:
                if count == 3:
                    first_finished.set()

        handle = self.pipeline.register_forward_pre_hook(pause_second_forward)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(request, 3)
                second = executor.submit(request, 7)
                results = [first.result(), second.result()]
        finally:
            handle.remove()
        self.assertEqual([result.usage.prompt_tokens for result in results], [3, 7])
        self.assertEqual(len(self.pipeline._forward_pre_hooks), 0)

    def test_token_count_accumulates_multiple_forward_batches(self):
        def encode(inputs, **kwargs):
            self.pipeline({"input_ids": torch.ones(2, 3), "attention_mask": torch.ones(2, 3)})
            self.pipeline({"input_ids": torch.ones(1, 5), "attention_mask": torch.ones(1, 5)})
            return np.ones((3, 2), dtype=np.float32)

        self.pipeline.encode.side_effect = encode
        result = Embeddings(self.pipeline)(EmbeddingsInput(input=["a", "b", "c"], model="test"))
        self.assertEqual(result.usage.prompt_tokens, 11)

    def test_google_http_route(self):
        self.pipeline.encode.return_value = torch.tensor([[0.6, 0.8]])
        app = FastAPI()
        app.include_router(
            google_predict_router(
                self.predictor,
                FeatureExtractionInputForGoogle,
                FeatureExtractionOutputForGoogle,
                FeatureExtractionInput,
            )
        )
        response = TestClient(app).post(
            "/predict",
            json={
                "instances": [{"image": "a.png"}],
                "parameters": {"dimensions": 2, "normalize": True},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["predictions"][0]["embeddings"]), 1)
        self.pipeline.encode.assert_called_once_with(
            {"image": "a.png"}, truncate_dim=2, normalize_embeddings=True, convert_to_tensor=True
        )

    def test_http_routes(self):
        self.pipeline.encode.return_value = torch.tensor([[0.6, 0.8]])
        app = FastAPI()
        with patch.object(self.predictor, "warmup"):
            app.include_router(predict_router(self.predictor, FeatureExtractionInput, FeatureExtractionOutput))
        app.include_router(embeddings_router(Embeddings(self.pipeline)))
        client = TestClient(app)
        response = client.post("/predict", json={"inputs": {"audio": "a.wav"}})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["embeddings"]), 1)
        self.pipeline.encode.return_value = np.array([0.6, 0.8], dtype=np.float32)
        response = client.post("/v1/embeddings", json={"input": {"image": "a.png"}, "model": "test"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"][0]["object"], "embedding")
        self.assertEqual(client.post("/predict", json={"inputs": {"image": []}}).status_code, 422)


class EmbeddingDtypeTests(unittest.TestCase):
    def test_embeddinggemma_precision(self):
        with (
            patch(
                "transformers.PretrainedConfig.get_config_dict",
                return_value=({"model_type": "embedding_gemma2"}, {}),
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            for device, expected in [
                ("cpu", "float32"),
                ("mps", "float32"),
                ("cuda", "bfloat16"),
                ("auto", "bfloat16"),
            ]:
                self.assertEqual(resolve_embedding_dtype("test", None, None, device), expected)
            self.assertEqual(resolve_embedding_dtype("test", None, "float32", "cuda"), "float32")
            with self.assertRaisesRegex(ValueError, "does not support float16"):
                resolve_embedding_dtype("test", None, "float16", "cuda")

    def test_cuda_without_bfloat16(self):
        with (
            patch(
                "transformers.PretrainedConfig.get_config_dict",
                return_value=({"model_type": "embedding_gemma2"}, {}),
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.is_bf16_supported", return_value=False),
        ):
            self.assertEqual(resolve_embedding_dtype("test", None, None, "cuda"), "float32")

    def test_other_models_keep_existing_dtype(self):
        with patch(
            "transformers.PretrainedConfig.get_config_dict", return_value=({"model_type": "bert"}, {})
        ) as config:
            self.assertEqual(resolve_embedding_dtype("test", "revision", None, "cpu"), "auto")
            config.assert_called_once_with("test", revision="revision")
            self.assertEqual(resolve_embedding_dtype("test", None, "float16", "cpu"), "float16")

    def test_models_with_config_in_module_subdirectory(self):
        with patch("transformers.PretrainedConfig.get_config_dict", side_effect=OSError("No root config.json")):
            self.assertEqual(resolve_embedding_dtype("test", None, None, "cpu"), "auto")

    def test_local_embeddinggemma_saved_in_module_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "0_Transformer").mkdir()
            (root / "0_Transformer" / "config.json").write_text(json.dumps({"model_type": "embedding_gemma2"}))
            (root / "modules.json").write_text(json.dumps([{"path": "0_Transformer"}]))
            self.assertEqual(resolve_embedding_dtype(directory, None, None, "cpu"), "float32")
            with self.assertRaisesRegex(ValueError, "does not support float16"):
                resolve_embedding_dtype(directory, None, "float16", "cpu")

    def test_non_transformer_configs_are_ignored_before_backbone(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "1_Pooling").mkdir()
            (root / "1_Pooling" / "config.json").write_text(json.dumps({"word_embedding_dimension": 4}))
            (root / "0_Transformer").mkdir()
            (root / "0_Transformer" / "config.json").write_text(json.dumps({"model_type": "embedding_gemma2"}))
            (root / "modules.json").write_text(json.dumps([{"path": "1_Pooling"}, {"path": "0_Transformer"}]))
            self.assertEqual(resolve_embedding_dtype(directory, None, None, "cpu"), "float32")


class NativeEmbeddingRegressionTests(unittest.TestCase):
    def test_saved_word_embeddings_and_pooling_still_load(self):
        word_embeddings = WordEmbeddings(
            WhitespaceTokenizer(["hello", "world"], stop_words=[]), torch.ones(2, 4)
        )
        pipeline = SentenceTransformer(modules=[word_embeddings, Pooling(4)], device="cpu")
        with tempfile.TemporaryDirectory() as directory:
            pipeline.save_pretrained(directory)
            predictor = FeatureExtraction(directory, device="cpu")
            result = predictor(FeatureExtractionInput(inputs="hello world"))
            np.testing.assert_allclose(result.embeddings, [[0.5, 0.5, 0.5, 0.5]])

    def test_native_encode_decodes_media_once_and_counts_prompt(self):
        class ImageWordEmbeddings(WordEmbeddings):
            modalities = ["text", "image"]

            def preprocess(self, inputs, prompt=None, **kwargs):
                texts = []
                for item in inputs:
                    with Image.open(item["image"]) as image:
                        image.load()
                    texts.append("hello world")
                return super().preprocess(texts, prompt=prompt, **kwargs)

        word_embeddings = ImageWordEmbeddings(
            WhitespaceTokenizer(["hello", "world", "prefix"], stop_words=[]), torch.ones(3, 4)
        )
        pipeline = SentenceTransformer(
            modules=[word_embeddings, Pooling(4)], device="cpu", prompts={"query": "prefix "}
        )
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "sample.png"
            Image.new("RGB", (8, 8), "red").save(image_path)
            with patch.object(Image, "open", wraps=Image.open) as decode:
                result = Embeddings(pipeline)(
                    EmbeddingsInput(
                        input={"image": str(image_path)}, model="test", prompt_name="query", dimensions=2
                    )
                )
            self.assertEqual(decode.call_count, 1)
            self.assertEqual(result.usage.prompt_tokens, 3)
            np.testing.assert_allclose(result.data[0].embedding, [2**-0.5, 2**-0.5], atol=1e-6)
            self.assertEqual(len(pipeline._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
