import base64
from io import BytesIO

from PIL.Image import Image as PILImage


class Image:
    @staticmethod
    def serialize(image: PILImage) -> str:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
