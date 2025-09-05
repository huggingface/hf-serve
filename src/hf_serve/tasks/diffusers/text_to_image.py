from typing import Optional

import torch  # NOTE: `torch` import cannot be lazy since it's used on both `__init__` and `__call__`
from PIL.Image import Image as PILImage
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, field_validator

from hf_serve.logging import logger
from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


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
                        "target_size": {"width": 512, "height": 512},
                        "num_inference_steps": 1,
                        "seed": 42,
                    },
                }
            ]
        }
    )


class TextToImageOutput(BaseModel):
    # NOTE: the output just contains `image` and not e.g. `images` since only one image can be generated
    # at a time at the moment
    image: PILImage

    model_config = ConfigDict(
        json_encoders={PILImage: Image.serialize},  # type: ignore
        arbitrary_types_allowed=True,
    )


# TODO: missing AIP_MODE handling i.e. input contains `instances` and output contains `predictions`
class TextToImage(Predictor[TextToImageInput, TextToImageOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        if device == "auto":
            logger.warning(
                f"{device=} is set, but on `diffusers` only `device_map='balanced'` is supported at the moment,"
                " meaning that the different pipeline components will be distributed among the available devices."
                " Alternatively, you can directly specify the devide to use instead being either 'cuda', 'mps' or 'cpu'."
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
            torch_dtype=getattr(torch, dtype),
            device=device if device != "balanced" else None,
            device_map=device if device == "balanced" else None,
            # NOTE: these are disabled to prevent generating black images
            safety_checker=None,
            requires_safety_checker=False,
        )

        # NOTE: ValueError: It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`.
        if device != "balanced":
            if device == "cuda" and torch.cuda.is_available():
                self.pipeline.enable_model_cpu_offload()
            elif device == "mps" and torch.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.set_per_process_memory_fraction(0.9)
                if (torch.mps.driver_allocated_memory() / (1024**3)) < 64:
                    self.pipeline.enable_attention_slicing()

    def __call__(self, payload: TextToImageInput) -> TextToImageOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_defaults=True, exclude_none=True)

        # NOTE: given that `seed` is not natively supported, we need to set the `seed` within a `torch.Generator`
        # in advance and provide that to the `DiffusionPipeline.__call__`
        if seed := parameters.pop("seed", None):
            parameters["generator"] = torch.Generator().manual_seed(int(seed))  # type: ignore

        images = self.pipeline(prompt=payload.inputs, **parameters)[0]
        return TextToImageOutput(image=images[0])
