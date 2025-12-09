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

        # NOTE: This is just a pre-validation to not try to deserialize `audio` in base64 as a `Path`, to prevent
        # the deserialization from failing or taking longer than anticipated
        if len(audio) < 260 and ("/" in audio or "\\" in audio or "." in audio):
            file_path = Path(audio)
            try:
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        return f.read()
            except OSError:
                # NOTE: If the provided `audio` is a `str` with a long `base64` encoding, when trying to check
                # if the `Path` is a file, it will fail with `OSError: [Errno 63] File name too long: ...`, to
                # prevent that we continue without re-raising the exception to still deserialize from `base64`
                pass

        try:
            return base64.b64decode(audio)
        except binascii.Error:
            raise ValueError(
                f"Input '{audio}' is neither a valid base64 string, public URL, nor a valid file system path"
            )
