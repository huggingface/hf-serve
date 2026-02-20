from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.image_classification import (
    ImageClassificationOutput,
    ImageClassificationParameters,
)


class ImageClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[ImageClassificationParameters] = Field(default=None)


class ImageClassificationOutputForGoogle(BaseModel):
    predictions: List[ImageClassificationOutput]
