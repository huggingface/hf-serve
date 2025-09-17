import base64
import binascii
from pathlib import Path

import requests


class Audio:
    @staticmethod
    def deserialize(audio: str) -> bytes:
        if audio.startswith(("http://", "https://")):
            response = requests.get(audio)
            response.raise_for_status()
            return response.content

        file_path = Path(audio)
        if file_path.is_file():
            with open(file_path, "rb") as f:
                return f.read()

        try:
            return base64.b64decode(audio)
        except binascii.Error:
            raise ValueError(
                f"Input '{audio}' is neither a valid base64 string, public URL, nor a valid file system path"
            )
