import re
from typing import Any

# NOTE: requests may contain large values e.g. base64-encoded images or audios, which error messages from both
# `hf-serve` and the underlying libraries (e.g. `transformers.image_utils.load_image`) tend to echo back in full,
# so those are truncated before being returned to the client
MAX_VALUE_LENGTH = 256
TRUNCATED_PREFIX_LENGTH = 64

# NOTE: an unbroken run of base64 / base64url characters this long is encoded data rather than a word, a path,
# or an URL, as those contain separators such as whitespace, dots, colons, or question marks; additionally,
# encoded binary data mixes lowercase, uppercase, and digits, and only has a "/" every 64 characters on average,
# which tells it apart from e.g. a long bare path
_ENCODED_RUN = re.compile(rf"[A-Za-z0-9+/=_-]{{{MAX_VALUE_LENGTH + 1},}}")


def _truncate(value: str) -> str:
    return f"{value[:TRUNCATED_PREFIX_LENGTH]}...[TRUNCATED, {len(value):,} characters in total]"


def _truncate_encoded(match: re.Match) -> str:
    value = match.group()
    is_encoded = (
        any(char.islower() for char in value)
        and any(char.isupper() for char in value)
        and any(char.isdigit() for char in value)
        and value.count("/") <= len(value) / 16
    )
    return _truncate(value) if is_encoded else value


def truncate_message(message: str) -> str:
    """Truncates the encoded data (e.g. base64 strings) within a free-form error message."""
    return _ENCODED_RUN.sub(_truncate_encoded, message)


def truncate_values(value: Any) -> Any:
    """Recursively truncates the strings longer than `MAX_VALUE_LENGTH` within a JSON-like value."""
    if isinstance(value, str):
        return _truncate(value) if len(value) > MAX_VALUE_LENGTH else value
    if isinstance(value, bytes):
        return truncate_values(value.decode("utf-8", errors="replace"))
    if isinstance(value, dict):
        return {key: truncate_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [truncate_values(item) for item in value]
    return value
