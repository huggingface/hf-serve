from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class Predictor(ABC, Generic[InputType, OutputType]):
    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(self, input: InputType) -> OutputType: ...
