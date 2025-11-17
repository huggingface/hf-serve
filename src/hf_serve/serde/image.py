import base64
from io import BytesIO
from typing import Literal, Union

from PIL import Image as ImageModule
from PIL.Image import Image as ImageType


class Image:
    @staticmethod
    def deserialize(image: Union[str, bytes]) -> ImageType:
        try:
            if isinstance(image, bytes):
                return ImageModule.open(BytesIO(image))
            elif isinstance(image, str):
                # TODO: given that `load_image` won't specifically handle that the image URL if provided
                # is invalid, we need to first download if an URL and handle that separately, then forward
                # it to `load_image`
                from transformers.image_utils import load_image

                return load_image(image)
        except Exception as e:
            raise ValueError(f"Failed to deserialize image: {e}")

    @staticmethod
    def serialize(image: ImageType, image_format: Literal["png", "jpeg", "webp"] = "png") -> str:
        buffer = BytesIO()
        image.save(buffer, **{"format": image_format})
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
