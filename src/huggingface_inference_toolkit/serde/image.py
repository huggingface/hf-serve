import base64
from io import BytesIO
from typing import Union

from PIL import Image as ImageModule
from PIL.Image import Image as ImageType
from transformers.image_utils import load_image


class Image:
    @staticmethod
    def deserialize(image_input: Union[str, bytes]) -> ImageType:
        try:
            if isinstance(image_input, bytes):
                return ImageModule.open(BytesIO(image_input))
            elif isinstance(image_input, str):
                return load_image(image_input)
        except Exception as e:
            raise ValueError(f"Failed to deserialize image: {e}")

    @staticmethod
    def serialize(image: ImageType) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
