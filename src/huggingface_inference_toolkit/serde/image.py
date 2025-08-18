import base64
from io import BytesIO
import requests
from typing import Union

from PIL import Image as PILImage


class Image:
    @staticmethod
    def deserialize(image_input: Union[str, bytes]) -> PILImage.Image:
        try:
            if isinstance(image_input, bytes):
                return PILImage.open(BytesIO(image_input))
            elif isinstance(image_input, str):
                if image_input.startswith(("http://", "https://")):
                    response = requests.get(image_input)
                    response.raise_for_status()
                    return PILImage.open(BytesIO(response.content))
                elif "." in image_input.split("/")[-1]:
                    # Assume it's a file path
                    return PILImage.open(image_input)
                else:
                    return PILImage.open(BytesIO(base64.b64decode(image_input)))
            else:
                raise ValueError("Unsupported image input type")
        except Exception as e:
            raise ValueError(f"Failed to deserialize image: {e}")

    @staticmethod
    def serialize(image: PILImage) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
