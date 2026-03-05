from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from hf_inference_sdk.tasks.predictor import Predictor


class AnyToAnyInputs(BaseModel):
    text: str
    images: Optional[Union[str, List[str]]] = Field(default=None)
    videos: Optional[Union[str, List[str]]] = Field(default=None)
    audio: Optional[Union[str, List[str]]] = Field(default=None)


class AnyToAnyInput(BaseModel):
    inputs: AnyToAnyInputs
    parameters: Optional[Dict[str, Any]] = Field(default=None)


class AnyToAnyOutput(BaseModel):
    generated_text: str
    # NOTE: It's missing `generated_audio` and `generated_image`, but given that we don't support such use-cases
    # yet, only `generated_text` is defined
    # from hf_inference_sdk.serde.image import ImageType
    # generated_image: Optional[ImageType] = Field(default=None)
    # from hf_inference_sdk.serde.audio import AudioType  # TODO: Missing `serialize` method
    # generated_audio: Optional[AudioType]


class AnyToAny(Predictor[AnyToAnyInput, AnyToAnyOutput]):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[str] = None,
        device: str = "auto",
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        import torch
        from transformers import pipeline
        from transformers.pipelines.any_to_any import AnyToAnyPipeline  # type: ignore

        self.pipeline: AnyToAnyPipeline = pipeline(  # type: ignore
            task="any-to-any",  # type: ignore
            model=model_id,
            revision=revision,
            dtype=getattr(torch, dtype) if dtype is not None else "auto",
            device=device if device != "auto" else None,
            device_map=device if device == "auto" else None,
            trust_remote_code=trust_remote_code,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

    def __call__(self, payload: AnyToAnyInput) -> AnyToAnyOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters

        if seed := parameters.pop("seed", None):
            from transformers import set_seed

            set_seed(seed)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": payload.inputs.text},
                ],
            }
        ]
        if isinstance(payload.inputs.audio, list):
            for audio in payload.inputs.audio:
                messages[-1]["content"].append({"type": "audio", "path": audio})
        else:
            messages[-1]["content"].append({"type": "audio", "path": payload.inputs.audio})

        # TODO: Add support for images and videos

        output = self.pipeline(messages, **parameters)  # type: ignore
        return AnyToAnyOutput(generated_text=output[0]["generated_text"][-1]["content"])
