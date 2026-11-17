import types
import unittest
from unittest import mock

import torch

from hf_serve.cli import parser
from hf_serve.devices import (
    CUDA_ATTN_IMPLEMENTATION,
    default_attn_implementation,
    is_neuron_available,
    resolve_device,
)
from hf_serve.tasks.diffusers.text_to_image import resolve_modular_device


def neuron_module(available: bool) -> types.SimpleNamespace:
    return types.SimpleNamespace(is_available=lambda: available)


class IsNeuronAvailableTest(unittest.TestCase):
    def test_without_torch_neuronx(self) -> None:
        with mock.patch.object(torch, "neuron", None, create=True):
            self.assertFalse(is_neuron_available())

    def test_with_torch_neuronx(self) -> None:
        for available in (True, False):
            with (
                self.subTest(available=available),
                mock.patch.object(torch, "neuron", neuron_module(available), create=True),
            ):
                self.assertEqual(is_neuron_available(), available)


class ResolveDeviceTest(unittest.TestCase):
    def test_resolve_device(self) -> None:
        cases = [
            # (device, cuda_available, mps_available, neuron_available, expected)
            ("auto", False, False, True, "neuron"),
            ("auto", False, False, False, "auto"),
            # NOTE: `device_map="auto"` keeps handling CUDA and MPS
            ("auto", True, False, True, "auto"),
            ("auto", False, True, True, "auto"),
            ("cpu", False, False, True, "cpu"),
            ("balanced", False, False, True, "balanced"),
            ("neuron", False, False, True, "neuron"),
        ]
        for device, cuda, mps, neuron, expected in cases:
            with (
                self.subTest(device=device, cuda=cuda, mps=mps, neuron=neuron),
                mock.patch("torch.cuda.is_available", return_value=cuda),
                mock.patch("torch.backends.mps.is_available", return_value=mps),
                mock.patch("hf_serve.devices.is_neuron_available", return_value=neuron),
            ):
                self.assertEqual(resolve_device(device), expected)

    def test_resolve_modular_device(self) -> None:
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch("hf_serve.tasks.diffusers.text_to_image.is_neuron_available", return_value=True),
        ):
            self.assertEqual(resolve_modular_device("auto"), "neuron")
            self.assertEqual(resolve_modular_device("cpu"), "cpu")


class DefaultAttnImplementationTest(unittest.TestCase):
    def test_default_attn_implementation(self) -> None:
        cases = [
            # (device, cuda_available, mps_available, expected)
            ("auto", True, False, CUDA_ATTN_IMPLEMENTATION),
            ("cuda", True, False, CUDA_ATTN_IMPLEMENTATION),
            ("balanced", True, False, CUDA_ATTN_IMPLEMENTATION),
            ("mps", False, True, "eager"),
            ("auto", False, True, "eager"),
            # NOTE: the Hub kernel is CUDA only, so other devices fall back to PyTorch SDPA
            ("neuron", False, False, "sdpa"),
            ("cpu", True, False, "sdpa"),
            ("auto", False, False, "sdpa"),
        ]
        for device, cuda, mps, expected in cases:
            with (
                self.subTest(device=device, cuda=cuda, mps=mps),
                mock.patch("torch.cuda.is_available", return_value=cuda),
                mock.patch("torch.backends.mps.is_available", return_value=mps),
            ):
                self.assertEqual(default_attn_implementation(device), expected)


class CliDeviceTest(unittest.TestCase):
    def test_accepts_neuron(self) -> None:
        args = parser.parse_args(["--model-id", "org/model", "--task", "text-generation", "--device", "neuron"])
        self.assertEqual(args.device, "neuron")


class TextGenerationKwargsTest(unittest.TestCase):
    def test_pipeline_receives_flat_generation_parameters(self) -> None:
        from hf_serve.tasks.transformers.text_generation import TextGeneration, TextGenerationInput

        pipeline = mock.Mock(return_value=[{"generated_text": "Paris"}], device=torch.device("cpu"))
        with (
            mock.patch("transformers.pipeline", return_value=pipeline) as pipeline_factory,
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch("torch.mps.is_available", return_value=False),
        ):
            predictor = TextGeneration(model_id="org/model", device="neuron")
            output = predictor(
                TextGenerationInput.model_validate(
                    {"inputs": "The capital of France is", "parameters": {"max_new_tokens": 4, "stop": ["."]}}
                )
            )

        factory_kwargs = pipeline_factory.call_args.kwargs
        self.assertEqual(factory_kwargs["device"], "neuron")
        self.assertNotIn("attn_implementation", factory_kwargs)
        self.assertEqual(factory_kwargs["model_kwargs"], {"attn_implementation": "sdpa"})

        call_kwargs = pipeline.call_args.kwargs
        self.assertNotIn("generate_kwargs", call_kwargs)
        self.assertEqual(call_kwargs["max_new_tokens"], 4)
        self.assertEqual(call_kwargs["stop_strings"], ["."])
        self.assertIs(call_kwargs["tokenizer"], pipeline.tokenizer)
        self.assertEqual(output.generated_text, "Paris")

    def test_stop_strings_are_matched_on_cpu_on_neuron(self) -> None:
        from hf_serve.tasks.transformers.text_generation import TextGeneration, TextGenerationInput

        pipeline = mock.Mock(
            return_value=[{"generated_text": "Paris"}], device=types.SimpleNamespace(type="neuron")
        )
        criteria_cls = mock.Mock()
        with (
            mock.patch("transformers.pipeline", return_value=pipeline),
            mock.patch(
                "hf_serve.tasks.transformers.stopping_criteria.cpu_stop_string_criteria_cls",
                return_value=criteria_cls,
            ),
            mock.patch("torch.mps.is_available", return_value=False),
        ):
            predictor = TextGeneration(model_id="org/model", device="neuron")
            predictor(TextGenerationInput.model_validate({"inputs": "Hi", "parameters": {"stop": ["."]}}))

        call_kwargs = pipeline.call_args.kwargs
        self.assertNotIn("stop_strings", call_kwargs)
        self.assertNotIn("tokenizer", call_kwargs)
        self.assertEqual(list(call_kwargs["stopping_criteria"]), [criteria_cls.return_value])
        criteria_cls.assert_called_once_with(tokenizer=pipeline.tokenizer, stop_strings=["."])

    def test_image_text_to_text_stop_strings_are_matched_on_cpu_on_neuron(self) -> None:
        from hf_serve.tasks.transformers.image_text_to_text import ImageTextToText, ImageTextToTextInput

        pipeline = mock.Mock(
            return_value=[{"generated_text": "A cat"}], device=types.SimpleNamespace(type="neuron")
        )
        criteria_cls = mock.Mock()
        with (
            mock.patch("transformers.pipeline", return_value=pipeline),
            mock.patch(
                "hf_serve.tasks.transformers.stopping_criteria.cpu_stop_string_criteria_cls",
                return_value=criteria_cls,
            ),
            mock.patch("torch.mps.is_available", return_value=False),
        ):
            predictor = ImageTextToText(model_id="org/model", device="neuron")
            predictor(
                ImageTextToTextInput.model_validate(
                    {
                        "inputs": {"image": "https://example.com/cat.png", "text": "What is this?"},
                        "parameters": {"max_new_tokens": 4, "temperature": 0.5, "stop": ["."]},
                    }
                )
            )

        call_kwargs = pipeline.call_args.kwargs
        self.assertEqual(call_kwargs["images"], "https://example.com/cat.png")
        self.assertEqual(call_kwargs["max_new_tokens"], 4)
        self.assertNotIn("temperature", call_kwargs)
        generate_kwargs = call_kwargs["generate_kwargs"]
        self.assertEqual(generate_kwargs["temperature"], 0.5)
        self.assertNotIn("stop_strings", generate_kwargs)
        self.assertEqual(list(generate_kwargs["stopping_criteria"]), [criteria_cls.return_value])
        criteria_cls.assert_called_once_with(tokenizer=pipeline.processor.tokenizer, stop_strings=["."])

    def test_cpu_stop_string_criteria_runs_on_cpu(self) -> None:
        from transformers import StopStringCriteria

        from hf_serve.tasks.transformers.stopping_criteria import cpu_stop_string_criteria_cls

        seen = {}

        def fake_call(self, input_ids, scores, **kwargs):
            seen["input_ids"] = input_ids
            return torch.tensor([True, False])

        with (
            mock.patch.object(StopStringCriteria, "__init__", return_value=None),
            mock.patch.object(StopStringCriteria, "__call__", fake_call),
        ):
            criteria = cpu_stop_string_criteria_cls()(tokenizer=None, stop_strings=["."])
            criteria.maximum_token_len = 3
            input_ids = torch.arange(10).reshape(2, 5)
            result = criteria(input_ids, None)

        self.assertEqual(seen["input_ids"].device.type, "cpu")
        self.assertTrue(torch.equal(seen["input_ids"], input_ids[:, -3:]))
        self.assertEqual(result.device, input_ids.device)
        self.assertEqual(result.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
