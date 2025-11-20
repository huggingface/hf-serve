from typing import List, Optional, Union

from PIL import Image as ImageModule
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.types import FileForm


# TODO(alvarobartt): Add `parameters` from https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.MaskGenerationPipeline.__call__
class MaskGenerationParameters(BaseModel): ...


class MaskGenerationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[MaskGenerationParameters] = Field(default=None)

    # model_config = ConfigDict(
    #     json_schema_extra={
    #         "examples": [
    #             {
    #                 "inputs": "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg"
    #             }
    #         ]
    #     },
    # )


class MaskGenerationFormInput(BaseModel):
    file: FileForm
    # TODO(alvarobartt): Add `parameters` from https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/pipelines#transformers.MaskGenerationPipeline.__call__

    model_config = ConfigDict(extra="forbid")


class MaskGenerationOutputValue(BaseModel):
    mask: ImageType
    score: Optional[float] = Field(default=None)

    model_config = ConfigDict(
        json_encoders={ImageType: Image.serialize},
        arbitrary_types_allowed=True,
    )


class MaskGenerationOutput(BaseModel):
    results: List[MaskGenerationOutputValue]


class MaskGeneration(Predictor[MaskGenerationInput, MaskGenerationOutput]):
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "auto") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.mask_generation import MaskGenerationPipeline

        # NOTE: Some models don't come with default `device_map` distribution, so let's stick
        # to standard device assignment in the meantime
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        # NOTE: On MPS it might fail during the forward pass with the following error (unrelated to the `--dtype` arg)
        # TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64. Please use float32 instead.
        self.pipeline: MaskGenerationPipeline = pipeline(
            task="mask-generation",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

        if device == "mps" and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: MaskGenerationInput) -> MaskGenerationOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)

        output = self.pipeline(Image.deserialize(payload.inputs), **parameters)

        return MaskGenerationOutput(
            results=[
                MaskGenerationOutputValue(
                    mask=ImageModule.fromarray(mask.cpu().numpy().astype("uint8") * 255), score=score
                )
                for (mask, score) in zip(output["masks"], output["scores"])
            ]
        )
