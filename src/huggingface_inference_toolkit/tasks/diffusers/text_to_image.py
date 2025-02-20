import base64
from io import BytesIO
from typing import List

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, Field

from huggingface_inference_toolkit.tasks.predictor import Predictor


class TextToImageInput(BaseModel):
    prompt: str = Field(
        validation_alias=AliasChoices("prompt", AliasPath("inputs"), AliasPath("inputs", "prompt"))
    )
    width: int = Field(256, validation_alias=AliasChoices("width", AliasPath("parameters", "width")))
    height: int = Field(256, validation_alias=AliasChoices("height", AliasPath("parameters", "height")))
    # TODO: add missing


class TextToImageOutput(BaseModel):
    images: List[str]


# TODO: missing AIP_MODE handling i.e. input contains `instances` and output contains `predictions`
class TextToImage(Predictor[TextToImageInput, TextToImageOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

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
            device=device if device not in {"balanced"} else None,
            device_map=device if device in {"balanced"} else None,
        )

        if torch.cuda.is_available():
            self.pipeline.enable_model_cpu_offload()
        elif torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)
            if (torch.mps.driver_allocated_memory() / (1024**3)) < 64:
                self.pipeline.enable_attention_slicing()

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        _ = self.pipeline("a photo of an astronaut riding a horse on mars", num_inference_steps=1)  # type: ignore

    def __call__(self, payload: TextToImageInput) -> TextToImageOutput:
        images = self.pipeline(**payload.model_dump()).images  # type: ignore
        buffered_images = []
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            buffered_images.append(base64.b64encode(buffered.getvalue()).decode())
        return TextToImageOutput(images=buffered_images)
