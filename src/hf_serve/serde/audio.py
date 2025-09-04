import base64
import binascii


class Audio:
    @staticmethod
    def deserialize(audio: str) -> bytes:
        """Deserialize base64-encoded string to raw audio bytes."""
        try:
            return base64.b64decode(audio)
        except binascii.Error as e:
            raise ValueError(f"Invalid base64 string: {e}") from e
