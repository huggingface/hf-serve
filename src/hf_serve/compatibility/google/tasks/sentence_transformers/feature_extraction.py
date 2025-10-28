from typing import Annotated, List, Literal, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.predictor import Predictor
from hf_serve.tasks.sentence_transformers.feature_extraction import (
    FeatureExtraction,
    FeatureExtractionInput,
    FeatureExtractionOutput,
)


class VertexParameters(BaseModel):
    normalize: bool = Field(default=True)
    dimensions: Optional[int] = Field(default=None)
    prompt_name: Optional[str] = Field(default=None)
    truncate: bool = Field(default=False)
    truncation_direction: Literal["left", "right"] = Field(default="right")


class VertexInput(BaseModel):
    instances: Annotated[List[Union[str, List[str]]], Len(min_length=1)]
    parameters: Optional[VertexParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": ["What is Deep Learning?"],
                    "parameters": {"normalize": True},
                },
            ]
        }
    )


class VertexOutput(BaseModel):
    predictions: List[FeatureExtractionOutput]


class VertexPredictor(Predictor[VertexInput, VertexOutput]):
    def __init__(self, predictor: FeatureExtraction) -> None:
        self.predictor = predictor

    def __call__(self, payload: VertexInput) -> VertexOutput:
        predictions = []
        for instance in payload.instances:
            input_payload = FeatureExtractionInput(
                sentences=instance,
                **{} if payload.parameters is None else payload.parameters.model_dump(exclude_defaults=True),
            )
            predictions.append(self.predictor(payload=input_payload))
        return VertexOutput(predictions=predictions)
