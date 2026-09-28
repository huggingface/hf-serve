from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.zero_shot_object_detection import (
    ZeroShotObjectDetectionOutput,
    ZeroShotObjectDetectionParameters,
)


class ZeroShotObjectDetectionInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[ZeroShotObjectDetectionParameters] = Field(default=None)


class ZeroShotObjectDetectionOutputForGoogle(BaseModel):
    predictions: List[ZeroShotObjectDetectionOutput]
