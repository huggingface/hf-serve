from typing import Annotated, List, Optional

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.summarization import SummarizationOutput, SummarizationParameters


class SummarizationInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[SummarizationParameters] = Field(default=None)


class SummarizationOutputForGoogle(BaseModel):
    predictions: List[SummarizationOutput]
