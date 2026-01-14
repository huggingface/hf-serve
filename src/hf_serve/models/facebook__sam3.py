from typing import Optional, Union

import torch  # NOTE: `torch` import cannot be lazy since it's used on both `__init__` and `__call__`
from PIL import Image as ImageModule
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.logging import logger
from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor
from hf_serve.tasks.transformers.mask_generation import (
    MaskGenerationOutput,
    MaskGenerationOutputValue,
    MaskGenerationParameters,
)


class FacebookSAM3Parameters(MaskGenerationParameters):
    # mask_threshold: Optional[float] = Field(default=0.0)
    # pred_iou_thresh: Optional[float] = Field(default=0.88)
    # stability_score_thresh: Optional[float] = Field(default=0.95)
    # stability_score_offset: Optional[int] = Field(default=1)
    # crops_nms_thresh: Optional[float] = Field(default=0.7)
    # crops_n_layers: Optional[int] = Field(default=0)
    # crop_overlap_ratio: Optional[float] = Field(default=512 / 1500)
    # crop_n_points_downscale_factor: Optional[int] = Field(default=1)
    # timeout: Optional[float] = Field(default=None)

    prompt: Optional[str] = Field(default=None)


class FacebookSAM3Input(BaseModel):
    inputs: Union[str, bytes] = Field(validation_alias=AliasChoices("inputs", "image"))
    parameters: Optional[FacebookSAM3Parameters] = Field(default=None, json_schema_extra={"overridden": True})

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
                    "parameters": {"prompt": "truck", "points_per_batch": 64},
                }
            ]
        },
    )


FacebookSAM3Output = MaskGenerationOutput


class FacebookSAM3(Predictor[FacebookSAM3Input, FacebookSAM3Output]):
    def __init__(
        self,
        model_id: str = "facebook/sam3",
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        import torch
        from transformers import Sam3Model, Sam3Processor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.model = Sam3Model.from_pretrained(
            model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
            trust_remote_code=trust_remote_code,
        ).to(device)

        self.processor = Sam3Processor.from_pretrained(
            model_id, revision=revision, trust_remote_code=trust_remote_code
        )

        if device == "mps" and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: FacebookSAM3Input) -> FacebookSAM3Output:
        if payload.parameters:
            if timeout := payload.parameters.timeout:
                logger.warning(
                    f"`{{..., 'parameters': {{..., {timeout=}}} has been provided, but it will be ignored and won't have any effect via `hf-serve`."
                )

        inputs = self.processor(
            images=Image.deserialize(payload.inputs),
            text=payload.parameters.prompt
            if payload.parameters and hasattr(payload.parameters, "prompt")
            else None,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        output = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # ?
            mask_threshold=payload.parameters.mask_threshold if payload.parameters else None,
            # pred_iou_thresh: Optional[float] = Field(default=0.88)
            # stability_score_thresh: Optional[float] = Field(default=0.95)
            # stability_score_offset: Optional[int] = Field(default=1)
            # crops_nms_thresh: Optional[float] = Field(default=0.7)
            # crops_n_layers: Optional[int] = Field(default=0)
            # crop_overlap_ratio: Optional[float] = Field(default=512 / 1500)
            # crop_n_points_downscale_factor: Optional[int] = Field(default=1)
            # timeout: Optional[float] = Field(default=None)
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        # Results contain:
        # - masks: Binary masks resized to original image size
        # - boxes: Bounding boxes in absolute pixel coordinates (xyxy format)
        # - scores: Confidence scores

        return FacebookSAM3Output(
            results=[
                MaskGenerationOutputValue(
                    mask=ImageModule.fromarray(mask.cpu().numpy().astype("uint8") * 255), score=score
                )
                for (mask, score) in zip(output["masks"], output["scores"])
            ]
        )
