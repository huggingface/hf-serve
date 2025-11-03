from typing import Annotated, List, Literal, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.token_classification import TokenClassificationOutput


class TokenClassificationParameters:
    aggregation_strategy: Optional[Literal["none", "simple", "first", "average", "max"]] = Field(default=None)
    ignore_labels: Optional[List[str]] = Field(default=None)
    stride: Optional[int] = Field(default=None)


class TokenClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[TokenClassificationParameters] = Field(default=None)


class TokenClassificationOutputForGoogle(BaseModel):
    predictions: List[TokenClassificationOutput]
