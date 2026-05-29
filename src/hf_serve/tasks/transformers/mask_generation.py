from typing import List, Optional, Union

from PIL import Image as ImageModule
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.types.form import FileForm, FloatForm, IntForm


class MaskGenerationParameters(BaseModel):
    mask_threshold: Optional[float] = Field(default=0.0)
    pred_iou_thresh: Optional[float] = Field(default=0.88)
    stability_score_thresh: Optional[float] = Field(default=0.95)
    stability_score_offset: Optional[int] = Field(default=1)
    crops_nms_thresh: Optional[float] = Field(default=0.7)
    crops_n_layers: Optional[int] = Field(default=0)
    crop_overlap_ratio: Optional[float] = Field(default=512 / 1500)
    crop_n_points_downscale_factor: Optional[int] = Field(default=1)
    timeout: Optional[float] = Field(default=None)


class MaskGenerationInput(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[MaskGenerationParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
                    "parameters": {"points_per_batch": 64},
                }
            ]
        },
    )


class MaskGenerationFormInput(BaseModel):
    file: FileForm

    mask_threshold: Optional[FloatForm] = Field(default=0.0)
    pred_iou_thresh: Optional[FloatForm] = Field(default=0.88)
    stability_score_thresh: Optional[FloatForm] = Field(default=0.95)
    stability_score_offset: Optional[IntForm] = Field(default=1)
    crops_nms_thresh: Optional[FloatForm] = Field(default=0.7)
    crops_n_layers: Optional[IntForm] = Field(default=0)
    crop_overlap_ratio: Optional[FloatForm] = Field(default=512 / 1500)
    crop_n_points_downscale_factor: Optional[IntForm] = Field(default=1)
    timeout: Optional[FloatForm] = Field(default=None)

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
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
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
            trust_remote_code=trust_remote_code,
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
