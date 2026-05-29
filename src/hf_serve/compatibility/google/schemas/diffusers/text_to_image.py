from typing import Annotated, List, Optional, Self

from annotated_types import Len
from pydantic import BaseModel, ConfigDict, Field, FieldSerializationInfo, field_serializer

from hf_serve.serde.image import Image
from hf_serve.tasks.diffusers.text_to_image import TextToImageOutput, TextToImageParameters


class TextToImageInputForGoogle(BaseModel):
    instances: Annotated[List[str], Len(min_length=1)]
    parameters: Optional[TextToImageParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "instances": ["a photo of an astronaut riding a horse on mars"],
                    "parameters": {
                        "target_size": {"width": 64, "height": 64},
                        "num_inference_steps": 1,
                        "seed": 42,
                    },
                }
            ]
        }
    )


class TextToImageOutputForGoogle(BaseModel):
    predictions: List[TextToImageOutput]

    # NOTE: Given that the `TextToImageOutput` is a `pydantic.RootModel` with a `PIL.Image.Image` that's not
    # serialized by default given that the `/predict` route for `text-to-image` should return the image on the
    # Inference API and not the JSON payload; the output needs to be serialized for Vertex AI via a
    # `field_serializer` for `predictions` instead.
    @field_serializer("predictions", when_used="json")
    def serialize_images(self: Self, value: List[TextToImageOutput], _: FieldSerializationInfo) -> List[str]:
        return [Image.serialize(image.root) for image in value]
