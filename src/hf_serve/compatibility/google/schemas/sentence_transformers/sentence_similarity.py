from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.sentence_transformers.sentence_similarity import (
    SentenceSimilarityInputs,
    SentenceSimilarityOutput,
    SentenceSimilarityParameters,
)


class SentenceSimilarityGoogleInput(BaseModel):
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


class SentenceSimilarityGoogleOutput(BaseModel):
    predictions: List[SentenceSimilarityOutput]
