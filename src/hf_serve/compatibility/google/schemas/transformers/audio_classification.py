from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.audio_classification import (
    AudioClassificationOutput,
    AudioClassificationParameters,
)


class AudioClassificationInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[AudioClassificationParameters] = Field(default=None)


class AudioClassificationOutputForGoogle(BaseModel):
    predictions: List[AudioClassificationOutput]
