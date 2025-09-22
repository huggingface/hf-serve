from typing import List, Optional

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, RootModel

from hf_serve.tasks.predictor import Predictor


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
                        "candidate_labels": ["urgent", "not urgent", "phone", "tablet", "computer"],
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
    def __init__(self, model_id: str, dtype: Optional[str] = None, device: str = "balanced") -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.zero_shot_classification import ZeroShotClassificationPipeline

        # NOTE: Apparently some (not all) models don't support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline: ZeroShotClassificationPipeline = pipeline(
            task="zero-shot-classification",
            model=model_id,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: ZeroShotClassificationInput) -> ZeroShotClassificationOutput:
        payload = payload.model_dump(exclude_none=True)

        if "parameters" in payload:
            parameters = payload.pop("parameters") or {}
            payload.update(parameters)

        pipeline_results = self.pipeline(**payload)  # type: ignore
        return ZeroShotClassificationOutput(root=pipeline_results)
