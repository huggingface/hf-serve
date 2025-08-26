from typing import Union

from PIL import Image as PILImage
from pydantic import BaseModel, field_validator

from huggingface_inference_toolkit.serde import Image


class ImageInput(BaseModel):
    inputs: Union[str, bytes]

    @field_validator("inputs", mode="after")
    @classmethod
    def deserialize_inputs(cls, v) -> PILImage.Image:
        return Image.deserialize(v)
