import os
from functools import lru_cache
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from PIL.Image import Image as PILImage

from hf_serve.logging import logger
from hf_serve.openai.schemas.images_generations import (
    ImagesGenerationsInput,
    ImagesGenerationsOutput,
)
from hf_serve.serde import Image

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline


class ImagesGenerations:
    def __init__(self, pipeline: "DiffusionPipeline") -> None:
        super().__init__()

        self.pipeline = pipeline

    @property
    @lru_cache(maxsize=1)
    def model_id(self) -> Union[str, None]:
        return (
            self.pipeline.config._name_or_path  # type: ignore
            if self.pipeline.config is not None and not Path(self.pipeline.config._name_or_path).exists()  # type: ignore
            else os.getenv("MODEL_ID", os.getenv("MODEL_DIR"))
        )

    def __call__(
        self, payload: ImagesGenerationsInput, request_id: Optional[str] = None
    ) -> ImagesGenerationsOutput:
        # NOTE: Some of the parameters are only supported within OpenAI, so reporting those to let the user know
        # that those cannot and won't be used for the Diffusers-based implementation.
        # NOTE: It's placed here instead of as a model-validator under `ImagesGenerationsInputs` as otherwise we
        # cannot propagate the `request_id` meaning the logging message is meaningless
        for parameter in {
            "background",
            "moderation",
            "output_compression",
            "partial_images",
            "quality",
            "style",
            "user",
        }:
            if hasattr(payload, parameter) and getattr(payload, parameter) is not None:
                logger.debug(
                    f"`[{request_id}] {parameter}={getattr(payload, parameter)}` was provided, but it's not supported, so it will be ignored."
                )
                setattr(payload, parameter, None)

        if hasattr(payload, "stream") and getattr(payload, "stream") is True:
            message = "[{request_id}] `stream=True` was provided, but it's not supported. Please make sure you set it to `False`."
            logger.error(message)
            raise ValueError(message)

        if hasattr(payload, "response_format") and getattr(payload, "response_format") == "url":
            message = "[{request_id}] `response_format='url'` is not supported, only `response_format='b64_json'` is supported."
            logger.error(message)
            raise ValueError(message)

        payload_json: Dict[str, Any] = {"prompt": payload.prompt, "num_images_per_prompt": payload.n}

        if payload.size and payload.size != "auto":
            try:
                height_str, width_str = payload.size.split("x")
                payload_json["height"] = int(height_str)
                payload_json["width"] = int(width_str)
            except Exception as e:
                message = f"[{request_id}] Provided `{payload.size=}` is not valid (parsing failed with `{e}`), it should either be an x-separated string with the heigth and width as integers as e.g. `1024x1024`, or `auto` to rely on the model default."
                logger.error(message)
                raise ValueError(message)

        images: List[PILImage] = self.pipeline(**payload_json)[0]  # type: ignore

        return ImagesGenerationsOutput(
            background=None,
            created=int(time()),
            data=[{"b64_json": Image.serialize(image, format=payload.output_format)} for image in images],
            output_format=payload.output_format,
            quality=None,
            size=f"{images[0].height}x{images[0].width}",
            usage=None,
        )
