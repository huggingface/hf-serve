from collections.abc import Mapping
from typing import Annotated, Any, Dict

from pydantic import AfterValidator

_REMOTE_CODE_PARAMETERS = frozenset({"custom_generate", "trust_remote_code"})


def validate_generation_parameters(parameters: Any) -> Any:
    """Reject request-level code loading options without restricting model-specific parameters."""
    pending = [parameters]
    while pending:
        value = pending.pop()
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key in _REMOTE_CODE_PARAMETERS:
                    raise ValueError(f"`{key}` is not allowed in request generation parameters")
                pending.append(nested)
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
    return parameters


GenerationParameters = Annotated[Dict[str, Any], AfterValidator(validate_generation_parameters)]
