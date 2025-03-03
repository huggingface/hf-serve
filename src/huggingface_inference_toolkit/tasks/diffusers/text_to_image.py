import base64
from io import BytesIO
from typing import Optional

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.tasks.predictor import Predictor


class TextToImageInput(BaseModel):
    prompt: str = Field(
        validation_alias=AliasChoices("prompt", AliasPath("inputs"), AliasPath("inputs", "prompt"))
    )
    width: Optional[int] = Field(
        None,
        validation_alias=AliasChoices(
            "width", AliasPath("parameters", "width"), AliasPath("parameters", "target_size", "width")
        ),
    )
    height: Optional[int] = Field(
        None,
        validation_alias=AliasChoices(
            "height", AliasPath("parameters", "height"), AliasPath("parameters", "target_size", "height")
        ),
    )
    guidance_scale: Optional[float] = Field(
        7.5,
        validation_alias=AliasChoices("guidance_scale", AliasPath("parameters", "guidance_scale")),
    )
    negative_prompt: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("negative_prompt", AliasPath("parameters", "negative_prompt")),
    )
    num_inference_steps: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("num_inference_steps", AliasPath("parameters", "num_inference_steps")),
    )
    seed: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("seed", AliasPath("parameters", "seed")),
    )
    # TODO: unsure about how the scheduler is provided / used
    # scheduler: Optional[str] = Field()

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
    image: str


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
        # NOTE: ValueError: It seems like you have activated a device mapping strategy on the pipeline so calling `enable_model_cpu_offload() isn't allowed. You can call `reset_device_map()` first and then call `enable_model_cpu_offload()`.
        device_kwargs = {"device": device} if device not in {"balanced"} else {"device_map": device}
        self.pipeline = AutoPipelineForText2Image.from_pretrained(
            model_id,
            torch_dtype=getattr(torch, dtype),
            **device_kwargs,
        )

        if device == "cuda" and torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        elif device == "mps" and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)
            if (torch.mps.driver_allocated_memory() / (1024**3)) < 64:
                self.pipeline.enable_attention_slicing()

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        # TODO: better validation and more meaningful errors on warmup
        self(TextToImageInput(**TextToImageInput.model_config["json_schema_extra"]["examples"][0]))  # type: ignore

    def __call__(self, payload: TextToImageInput) -> TextToImageOutput:
        payload_dump = payload.model_dump(exclude_defaults=True)

        # TODO: explore if can be integrated within the schema itself
        if "seed" in payload_dump:
            payload_dump["generator"] = torch.Generator().manual_seed(int(payload_dump["seed"]))
            payload_dump.pop("seed")

        # TODO: add custom error to inform the user about either pipeline for i/o formatting failures
        out = self.pipeline(**payload_dump)
        image = out.images[0]  # type: ignore
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image = base64.b64encode(buffered.getvalue()).decode()

        return TextToImageOutput(image=image)
