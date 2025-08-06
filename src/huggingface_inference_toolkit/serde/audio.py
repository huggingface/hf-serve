import base64


class Audio:
    @staticmethod
    def deserialize(audio_base64: str) -> bytes:
        """Deserialize base64-encoded string to raw audio bytes."""
        try:
            dec = base64.b64decode(audio_base64)
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid base64 string: {e}") from e

        return dec
