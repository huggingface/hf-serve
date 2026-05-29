from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.zero_shot_image_classification import (
    ZeroShotImageClassificationOutput,
    ZeroShotImageClassificationParameters,
)


class ZeroShotImageClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[ZeroShotImageClassificationParameters] = Field(default=None)


class ZeroShotImageClassificationOutputForGoogle(BaseModel):
    predictions: List[ZeroShotImageClassificationOutput]
