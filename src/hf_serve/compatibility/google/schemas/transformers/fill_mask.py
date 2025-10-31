from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.fill_mask import FillMaskOutput, FillMaskParameters


class FillMaskInputOnGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[FillMaskParameters] = Field(default=None)


class FillMaskOutputOnGoogle(BaseModel):
    predictions: List[FillMaskOutput]
