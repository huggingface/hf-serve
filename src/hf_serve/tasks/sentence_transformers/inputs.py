from typing import Annotated, Dict, List, Literal, Union

from pydantic import AfterValidator, Field


def validate_multimodal_input(value: Dict[str, Union[str, List[str]]]) -> Dict[str, Union[str, List[str]]]:
    if not value:
        raise ValueError("A multimodal input must contain at least one modality")
    return value


NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
MultimodalInput = Annotated[
    Dict[
        Literal["text", "image", "audio", "video"],
        Union[NonEmptyString, Annotated[List[NonEmptyString], Field(min_length=1)]],
    ],
    AfterValidator(validate_multimodal_input),
]
EmbeddingInput = Union[NonEmptyString, MultimodalInput]
EmbeddingInputs = Union[EmbeddingInput, Annotated[List[EmbeddingInput], Field(min_length=1)]]
