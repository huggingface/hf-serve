from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class EmbeddingsInput(BaseModel):
    # NOTE: Using `alias="input"` because `input` is a reserved keyword. Note that when dumping the schema into
    # a JSON, you should use `.model_dump(by_alias=True)` so that the dump uses the alias rather than `input_`
    input_: Union[str, List[str]] = Field(..., alias="input")
    model: str
    # NOTE: Dimensions is only supported if the embedding model was trained with Matryoshka Representation
    # Learning (MRL), otherwise it will use the default dimensions it was trained for, not allowing truncation
    # to lower dimensions
    dimensions: Optional[int] = Field(default=None)
    encoding_format: Literal["float", "base64"] = Field(default="float")
    user: Optional[str] = Field(default=None)


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingOutput(BaseModel):
    object_: Literal["embedding"] = Field(default="embedding", alias="object")
    embedding: Union[List[float], str]
    index: int


class EmbeddingsOutput(BaseModel):
    object_: Literal["list"] = Field(default="list", alias="object")
    data: List[EmbeddingOutput]
    model: str
    usage: Usage
