from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.image_text_to_text import (
    ImageTextToTextInputs,
    ImageTextToTextOutput,
    ImageTextToTextParameters,
)


class ImageTextToTextInputForGoogle(BaseModel):
    instances: Annotated[List[ImageTextToTextInputs], Len(min_length=1)]
    parameters: Optional[ImageTextToTextParameters] = Field(default=None)


class ImageTextToTextOutputForGoogle(BaseModel):
    predictions: List[ImageTextToTextOutput]
