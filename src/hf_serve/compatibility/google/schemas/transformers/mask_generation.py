from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.mask_generation import MaskGenerationOutput, MaskGenerationParameters


class MaskGenerationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[MaskGenerationParameters] = Field(default=None)


class MaskGenerationOutputForGoogle(BaseModel):
    predictions: List[MaskGenerationOutput]
