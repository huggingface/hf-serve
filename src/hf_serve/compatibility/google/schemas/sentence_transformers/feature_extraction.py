from typing import Annotated, List, Literal, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field

from hf_serve.tasks.sentence_transformers.feature_extraction import FeatureExtractionOutput


class FeatureExtractionParameters(BaseModel):
    normalize: bool = Field(default=True)
    dimensions: Optional[int] = Field(default=None)
    prompt_name: Optional[str] = Field(default=None)
    truncate: bool = Field(default=False)
    truncation_direction: Literal["left", "right"] = Field(default="right")


class FeatureExtractionGoogleInput(BaseModel):
    instances: Annotated[List[Union[str, List[str]]], Len(min_length=1)]
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


class FeatureExtractionGoogleOutput(BaseModel):
    predictions: List[FeatureExtractionOutput]
