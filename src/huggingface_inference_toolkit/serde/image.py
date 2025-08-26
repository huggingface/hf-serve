import base64
from io import BytesIO
from typing import Union

from PIL import Image as PILImage
from transformers.image_utils import load_image


class Image:
    @staticmethod
    def deserialize(image_input: Union[str, bytes]) -> PILImage.Image:
        try:
            if isinstance(image_input, bytes):
                return PILImage.open(BytesIO(image_input))
            elif isinstance(image_input, str):
                img = load_image(image_input)
                return img
            else:
                raise ValueError("Unsupported image input type")
        except Exception as e:
            raise ValueError(f"Failed to deserialize image: {e}")

    @staticmethod
    def serialize(image: PILImage) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
