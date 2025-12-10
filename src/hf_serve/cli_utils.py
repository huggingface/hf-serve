from typing import Optional


def strtobool(value: Optional[str]) -> bool:
    return True if value is not None and value.lower() in {"y", "yes", "t", "true", "on", "1"} else False
