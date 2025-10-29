from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.predictor import Predictor
from hf_serve.tasks.sentence_transformers.sentence_similarity import (
    SentenceSimilarity,
    SentenceSimilarityInput,
    SentenceSimilarityInputs,
    SentenceSimilarityOutput,
    SentenceSimilarityParameters,
)


class VertexInput(BaseModel):
    instances: Annotated[List[SentenceSimilarityInputs], Len(min_length=1)]
    parameters: Optional[SentenceSimilarityParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": [
                        {
                            "source_sentence": "I'm very happy",
                            "sentences": ["I'm filled with happiness", "I'm happy"],
                        }
                    ],
                    "parameters": None,
                },
            ]
        }
    )


class VertexOutput(BaseModel):
    predictions: List[SentenceSimilarityOutput]


class VertexPredictor(Predictor[VertexInput, VertexOutput]):
    def __init__(self, predictor: SentenceSimilarity) -> None:
        self.predictor = predictor

    def __call__(self, payload: VertexInput) -> VertexOutput:
        predictions = []
        for instance in payload.instances:
            input_payload = SentenceSimilarityInput(inputs=instance, parameters=None)  # type: ignore
            predictions.append(self.predictor(payload=input_payload))
        return VertexOutput(predictions=predictions)
