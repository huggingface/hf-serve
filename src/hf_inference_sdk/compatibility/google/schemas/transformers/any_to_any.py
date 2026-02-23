from typing import Annotated, Any, Dict, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.any_to_any import AnyToAnyOutput


class AnyToAnyInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[Dict[str, Any]] = Field(default=None)


class AnyToAnyOutputForGoogle(BaseModel):
    predictions: List[AnyToAnyOutput]
