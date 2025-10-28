from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.diffusers.text_to_image import (
    TextToImageInput,
    TextToImageOutput,
    TextToImageParameters,
)
from hf_serve.tasks.predictor import Predictor


class VertexInput(BaseModel):
    instances: List[str]
    parameters: Optional[TextToImageParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": ["a photo of an astronaut riding a horse on mars"],
                    "parameters": {
                        "target_size": {"width": 64, "height": 64},
                        "num_inference_steps": 1,
                        "seed": 42,
                    },
                }
            ]
        }
    )


class VertexOutput(BaseModel):
    predictions: List[TextToImageOutput]


class VertexPredictor(Predictor[VertexInput, VertexOutput]):
    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def __call__(self, payload: VertexInput) -> VertexOutput:
        predictions = []
        for instance in payload.instances:
            predictions.append(self.predictor(TextToImageInput(inputs=instance, parameters=payload.parameters)))
        return VertexOutput(predictions=predictions)
