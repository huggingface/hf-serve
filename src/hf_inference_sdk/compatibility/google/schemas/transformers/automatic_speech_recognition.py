from typing import Annotated, List, Optional, Union

from annotated_types import Len
from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.transformers.automatic_speech_recognition import (
    AutomaticSpeechRecognitionOutput,
    AutomaticSpeechRecognitionParameters,
)


class AutomaticSpeechRecognitionInputForGoogle(BaseModel):
    instances: Annotated[List[Union[str, bytes]], Len(min_length=1)]
    parameters: Optional[AutomaticSpeechRecognitionParameters] = Field(default=None)


class AutomaticSpeechRecognitionOutputForGoogle(BaseModel):
    predictions: List[AutomaticSpeechRecognitionOutput]
