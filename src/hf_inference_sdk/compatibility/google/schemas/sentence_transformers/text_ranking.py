from typing import Annotated, List, Literal, Optional

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_inference_sdk.tasks.sentence_transformers.text_ranking import TextRankingOutput


class TextRankingInput(BaseModel):
    query: str
    documents: List[str]


class TextRankingParameters(BaseModel):
    return_documents: bool = Field(default=False)
    raw_scores: bool = Field(default=False)
    truncate: bool = Field(default=False)
    truncation_direction: Literal["left", "right"] = Field(default="right")


class TextRankingInputForGoogle(BaseModel):
    instances: Annotated[List[TextRankingInput], Len(min_length=1)]
    parameters: Optional[TextRankingParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": [
                        {
                            "query": "What is Deep Learning?",
                            "texts": ["Deep Learning is...", "Deep Learning is not ..."],
                        }
                    ],
                    "parameters": {"return_documents": True},
                },
            ]
        }
    )


class TextRankingOutputForGoogle(BaseModel):
    predictions: List[TextRankingOutput]
