from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.token_classification import (
    TokenClassificationOutput,
    TokenClassificationParameters,
)


class TokenClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[TokenClassificationParameters] = Field(default=None)


class TokenClassificationOutputForGoogle(BaseModel):
    predictions: List[TokenClassificationOutput]
