from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Union

from pydantic import BaseModel

# NOTE: here to handle both the standard type and the `anyOf` syntax if multiple I/O schemas are valid
InputType = TypeVar("InputType", bound=Union[BaseModel, Union[BaseModel, ...]])  # type: ignore
OutputType = TypeVar("OutputType", bound=Union[BaseModel, Union[BaseModel, ...]])  # type: ignore


class Predictor(ABC, Generic[InputType, OutputType]):
    def __init__(self) -> None: ...

    @abstractmethod
    def __call__(self, payload: InputType) -> OutputType: ...
