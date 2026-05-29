from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.image_segmentation import (
    ImageSegmentationOutput,
    ImageSegmentationParameters,
)


class ImageSegmentationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[ImageSegmentationParameters] = Field(default=None)


class ImageSegmentationOutputForGoogle(BaseModel):
    predictions: List[ImageSegmentationOutput]
