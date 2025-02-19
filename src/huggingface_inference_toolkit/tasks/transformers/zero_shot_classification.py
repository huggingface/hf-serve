from typing import List, Optional

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor

class ZeroShotClassificationParameters(BaseModel):
    candidate_labels: List[str]
    hypothesis_template: Optional[str] = None
    multi_label: Optional[bool] = None

class ZeroShotClassificationInput(BaseModel):
    sequences: str = Field(
        validation_alias=AliasChoices("sequences", AliasPath("text"), AliasPath("sequences", "inputs")),
    )
    parameters: Optional[ZeroShotClassificationParameters] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sequences": "I have a problem with my iphone that needs to be resolved asap!!",
                    "parameters": {
                        "candidate_labels":["urgent", "not urgent", "phone", "tablet", "computer"],
                    },
                }
            ]
        }
    )


class ZeroShotClassificationOutputValue(BaseModel):
    sequence: str
    labels: List[str]
    scores: List[float]


class ZeroShotClassificationOutput(RootModel):
    root: ZeroShotClassificationOutputValue


class ZeroShotClassification(Predictor[ZeroShotClassificationInput, ZeroShotClassificationOutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="zero-shot-classification",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = ZeroShotClassificationInput(**ZeroShotClassificationInput.model_json_schema().get("examples")[0])
        _ = self(warmup_input)

    def __call__(self, input: ZeroShotClassificationInput) -> ZeroShotClassificationOutput:
        payload = input.model_dump(exclude_none=True)
        
        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(**payload)  # type: ignore
        return ZeroShotClassificationOutput(root=pipeline_results)

