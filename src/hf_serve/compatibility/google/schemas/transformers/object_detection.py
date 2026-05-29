from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.object_detection import (
    ObjectDetectionOutput,
    ObjectDetectionParameters,
)


class ObjectDetectionInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[ObjectDetectionParameters] = Field(default=None)


class ObjectDetectionOutputForGoogle(BaseModel):
    predictions: List[ObjectDetectionOutput]
