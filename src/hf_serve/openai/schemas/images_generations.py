from typing import Annotated, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ImagesGenerationsInput(BaseModel):
    prompt: str
    background: Optional[Literal["transparent", "opaque", "auto"]] = Field(default=None)
    model: Optional[str]
    moderation: Optional[Literal["low", "auto"]] = Field(default=None)
    n: Optional[Annotated[int, Field(..., ge=1, le=10)]] = Field(default=None)
    output_compression: Optional[Annotated[int, Field(..., ge=0, le=100)]] = Field(default=None)
    output_format: Literal["png", "jpeg", "webp"] = Field(default="png")
    partial_images: Optional[Annotated[int, Field(..., ge=0, le=3)]] = Field(default=None)
    quality: Optional[Literal["auto", "high", "medium", "low", "hd", "standard"]] = Field(default=None)
    response_format: Optional[Literal["url", "b64_json"]] = Field(default="url")
    size: Optional[str] = Field(default="auto")
    stream: Optional[bool] = Field(default=False)
    style: Optional[Literal["vivid", "natural"]] = Field(default=None)
    user: Optional[str] = Field(default=None)


class Details(BaseModel):
    image_tokens: int
    text_tokens: int


class Usage(BaseModel):
    input_tokens: int
    input_tokens_details: Details
    output_tokens: int
    total_tokens: int


class ImagesGenerationsOutput(BaseModel):
    background: Optional[Literal["transparent", "opaque"]] = Field(default=None)
    created: int
    data: List[Dict[Literal["b64_json", "url", "revised_prompt"], str]]
    output_format: Literal["png", "jpeg", "webp"]
    quality: Optional[Literal["low", "medium", "high"]] = Field(default=None)
    size: str
    usage: Optional[Usage] = Field(default=None)
