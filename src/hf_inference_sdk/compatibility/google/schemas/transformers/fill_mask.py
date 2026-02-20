from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.fill_mask import FillMaskOutput, FillMaskParameters


class FillMaskInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[FillMaskParameters] = Field(default=None)


class FillMaskOutputForGoogle(BaseModel):
    predictions: List[FillMaskOutput]
