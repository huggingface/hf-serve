from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_serve.tasks.transformers.voice_activity_detection import (
    VoiceActivityDetectionOutput,
    VoiceActivityDetectionParameters,
)


class VoiceActivityDetectionInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[VoiceActivityDetectionParameters] = Field(default=None)


class VoiceActivityDetectionOutputForGoogle(BaseModel):
    predictions: List[VoiceActivityDetectionOutput]
