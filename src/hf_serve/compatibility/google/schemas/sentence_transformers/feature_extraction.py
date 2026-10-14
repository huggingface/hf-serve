from typing import Annotated, List, Literal, Optional

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.sentence_transformers.feature_extraction import FeatureExtractionOutput
from hf_serve.tasks.sentence_transformers.inputs import EmbeddingInputs


class FeatureExtractionParameters(BaseModel):
    normalize: bool = Field(default=True)
    dimensions: Optional[int] = Field(default=None, gt=0)
    prompt_name: Optional[str] = Field(default=None)
    truncate: bool = Field(default=False)
    truncation_direction: Literal["left", "right"] = Field(default="right")


class FeatureExtractionInputForGoogle(BaseModel):
    instances: Annotated[List[EmbeddingInputs], Len(min_length=1)]
    parameters: Optional[FeatureExtractionParameters] = Field(default=None)

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


class FeatureExtractionOutputForGoogle(BaseModel):
    predictions: List[FeatureExtractionOutput]
