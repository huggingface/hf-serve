from typing import List, Optional, Tuple, Union

import torch  # NOTE: `torch` import cannot be lazy since it's used on both `__init__` and `__call__`
from PIL import Image as ImageModule
from PIL.Image import Image as ImageType
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from hf_serve.serde import Image
from hf_serve.tasks.predictor import Predictor


class Sam3Parameters(BaseModel):
    mask_threshold: Optional[float] = Field(default=0.5)


class Sam3Inputs(BaseModel):
    text: str
    image: Union[str, bytes]


class Sam3Input(BaseModel):
    inputs: Sam3Inputs
    parameters: Optional[Sam3Parameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "image": "https://huggingface.co/datasets/hf-internal-testing/sam2-fixtures/resolve/main/truck.jpg",
                        "text": "prompt",
                    },
                    "parameters": {"mask_threshold": 0.5},
                }
            ]
        },
    )


class Sam3OutputValue(BaseModel):
    mask: ImageType
    score: Optional[float] = Field(default=None)
    box: Optional[Tuple[float, float, float, float]] = Field(default=None)

    model_config = ConfigDict(
        json_encoders={ImageType: Image.serialize},
        arbitrary_types_allowed=True,
    )


class Sam3Output(BaseModel):
    results: List[Sam3OutputValue]


class Sam3(Predictor[Sam3Input, Sam3Output]):
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
            revision=revision or "main",
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            trust_remote_code=trust_remote_code,
        )
        self.model.to(device)  # type: ignore

        self.processor = Sam3Processor.from_pretrained(
            model_id, revision=revision or "main", trust_remote_code=trust_remote_code
        )

        if device == "mps" and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: Sam3Input) -> Sam3Output:
        inputs = self.processor(
            images=Image.deserialize(payload.inputs.image),
            text=payload.inputs.text,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        output = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,  # ?
            mask_threshold=payload.parameters.mask_threshold
            if payload.parameters and payload.parameters.mask_threshold is not None
            else 0.0,
            target_sizes=inputs.get("original_sizes").tolist(),  # type: ignore
        )[0]

        return Sam3Output(
            results=[
                Sam3OutputValue(
                    mask=ImageModule.fromarray(mask.astype("uint8") * 255),
                    score=score,
                    box=box,
                )
                for (mask, score, box) in zip(
                    output["masks"].cpu().numpy(), output["scores"], output["boxes"].cpu().numpy()
                )
            ]
        )
