import os
from typing import List, Optional, Tuple

import torch  # NOTE: `torch` import cannot be lazy since it's used on both `__init__` and `__call__`
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel, field_validator

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor

MODULAR_MODEL_INDEX = "modular_model_index.json"
MODEL_INDEX = "model_index.json"


def is_modular_pipeline(model_id: str, revision: Optional[str] = None) -> bool:
    """Whether `model_id` only ships a `diffusers.ModularPipeline`, i.e. a `modular_model_index.json`
    without a `model_index.json`, which `AutoPipelineForText2Image` cannot load."""
    if os.path.isdir(model_id):
        return os.path.isfile(os.path.join(model_id, MODULAR_MODEL_INDEX)) and not os.path.isfile(
            os.path.join(model_id, MODEL_INDEX)
        )

    from huggingface_hub import file_exists

    return file_exists(model_id, MODULAR_MODEL_INDEX, revision=revision) and not file_exists(
        model_id, MODEL_INDEX, revision=revision
    )


def resolve_modular_device(device: str) -> str:
    # NOTE: `ModularPipeline` has no `device_map`, so `auto` and `balanced` resolve to a single device
    if device not in {"auto", "balanced"}:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class TextToImageParameters(BaseModel):
    negative_prompt: Optional[str] = Field(default=None)
    width: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("width", AliasPath("target_size", "width")),
    )
    height: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("height", AliasPath("target_size", "height")),
    )
    num_inference_steps: Optional[int] = Field(default=None)
    guidance_scale: Optional[float] = Field(default=None)
    num_images_per_prompt: int = Field(default=1)
    seed: Optional[int] = Field(default=None)

    @field_validator("num_images_per_prompt")
    @classmethod
    def validate_num_images_per_prompt(cls, v: int) -> int:
        if v != 1:
            logger.warning(
                f"num_images_per_prompt={v} provided, but only num_images_per_prompt=1 is supported. "
                "Setting num_images_per_prompt=1 instead. Any other value won't have any effect."
            )
            return 1
        return v


class TextToImageInput(BaseModel):
    # NOTE: if we plan on adding full support, even if still compatible with the Inference API we should most
    # likely add and handle the `prompt` and `prompt_2` as inputs, as well as the `negative_prompt` and
    # `negative_prompt_2` parameters
    inputs: str
    parameters: Optional[TextToImageParameters] = Field(default=None)

    # NOTE: these examples are prepared in a way so that those appear in the Swagger API docs
    # with compatibility for Inference Endpoints, but any of the aliases above can be used instead
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "a photo of an astronaut riding a horse on mars",
                    "parameters": {
                        "target_size": {"width": 64, "height": 64},
                        "num_inference_steps": 1,
                        "seed": 42,
                    },
                }
            ]
        }
    )


class TextToImageOutput(RootModel):
    root: ImageType

    # NOTE: No serialization to bytes here, given that the Hugging Face API expects the `Accept` header to reply
    # with the image rather than the bytes
    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextToImage(Predictor[TextToImageInput, TextToImageOutput]):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "balanced",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        self.is_modular = is_modular_pipeline(model_id, revision=revision)
        if self.is_modular:
            from diffusers import ModularPipeline  # type: ignore

            # NOTE: components whose files are within a local `model_id` (e.g. `/repository` on Inference
            # Endpoints) are loaded from there, any other component (e.g. a prompt upsampler living in a
            # separate repository) is loaded from the Hub as declared in `modular_model_index.json`
            self.pipeline = ModularPipeline.from_pretrained(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            if dtype is not None:
                self.pipeline.load_components(dtype=getattr(torch, dtype))
            else:
                self.pipeline.load_components()
            self.pipeline.to(resolve_modular_device(device))
            return

        if device == "auto":
            logger.warning(
                f"{device=} is set, but on `diffusers` only `device_map='balanced'` is supported at the moment,"
                " meaning that the different pipeline components will be distributed among the available devices."
                " Alternatively, you can directly specify the device to use instead being either 'cuda', 'mps' or 'cpu'."
            )
            device = "balanced"

        from diffusers import AutoPipelineForText2Image  # type: ignore

        # TODO: maybe add some pre-download checks to prevent downloading all the files but then stumbling
        # upon e.g. `OSError: TencentARC/flux-mini does not appear to have a file named model_index.json.`
        # when loading the model
        # NOTE: it appears that the `model_id` on Inference Endpoints pre-downloads the files, meaning that in
        # /opt/huggingface/model all the contents for the given model should be downloaded and available
        # meaning that e.g. the fix for `diffusers` should be applied there
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=getattr(torch, dtype) if dtype is not None else None,
            device=device if device != "balanced" else None,
            device_map=device,
            # NOTE: these are disabled to prevent generating black images
            safety_checker=None,
            requires_safety_checker=False,
            trust_remote_code=trust_remote_code,
        )

        # NOTE: ValueError: It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`.
        # if device == "cuda" and torch.cuda.is_available():
        #     self.pipeline.enable_model_cpu_offload()

        if device == "mps" and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)
            if (torch.mps.driver_allocated_memory() / (1024**3)) < 64:
                self.pipeline.enable_attention_slicing()

    def generate(self, prompt: str, **kwargs) -> Tuple[List[ImageType], Optional[str]]:
        """Runs the pipeline and returns the generated images, plus the prompt the image model actually
        received when a `ModularPipeline` rewrote it (e.g. via a prompt upsampler), otherwise `None`."""
        if not self.is_modular:
            return self.pipeline(prompt=prompt, **kwargs)[0], None

        output = self.pipeline(prompt=prompt, **kwargs, output=["images", "prompt"])
        revised_prompt = output.get("prompt")
        if isinstance(revised_prompt, list):
            revised_prompt = revised_prompt[0] if revised_prompt else None
        return output["images"], revised_prompt if revised_prompt and revised_prompt != prompt else None

    def __call__(self, payload: TextToImageInput) -> TextToImageOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_defaults=True, exclude_none=True)

        # NOTE: given that `seed` is not natively supported, we need to set the `seed` within a `torch.Generator`
        # in advance and provide that to the `DiffusionPipeline.__call__`
        if seed := parameters.pop("seed", None):
            parameters["generator"] = torch.Generator().manual_seed(int(seed))  # type: ignore

        # NOTE: `num_images_per_prompt=1` because the `TextToImage` task returns an image without
        # a JSON, meaning that only a single image is supported
        images, _ = self.generate(prompt=payload.inputs, **parameters, num_images_per_prompt=1)
        return TextToImageOutput(root=images[0])
