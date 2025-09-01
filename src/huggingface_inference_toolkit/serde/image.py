import base64
from io import BytesIO
from typing import Union

from PIL import Image as ImageModule
from PIL.Image import Image as ImageType


class Image:
    @staticmethod
    def deserialize(image: Union[str, bytes]) -> ImageType:
        try:
            if isinstance(image, bytes):
                return ImageModule.open(BytesIO(image))
            elif isinstance(image, str):
                from transformers.image_utils import load_image

                return load_image(image)
        except Exception as e:
            raise ValueError(f"Failed to deserialize image: {e}")

    @staticmethod
    def serialize(image: ImageType) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
