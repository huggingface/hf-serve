from typing import List, Literal, Optional, Tuple, Union

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from huggingface_inference_toolkit.tasks.predictor import Predictor


class ASRInput(BaseModel):
    inputs: str = Field(
        validation_alias=AliasChoices("inputs", AliasPath("input_features")),
    )
    return_timestamps: Optional[bool] = Field(
        False,
        validation_alias=AliasChoices("return_timestamps", AliasPath("parameters", "return_timestamps")),
    )
    do_sample: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices(
            "do_sample",
            AliasPath(
                "parameters",
                "generate_kwargs",
                "do_sample",
            ),
        ),
    )
    early_stopping: Optional[Union[bool, Literal["never"]]] = Field(
        None,
        validation_alias=AliasChoices(
            "early_stopping",
            AliasPath(
                "parameters",
                "generate_kwargs",
                "early_stopping",
            ),
        ),
    )
    epsilon_cutoff: Optional[float] = Field(
        None,
        validation_alias=AliasChoices(
            "epsilon_cutoff", AliasPath("parameters", "generate_kwargs", "epsilon_cutoff")
        ),
    )
    eta_cutoff: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("eta_cutoff", AliasPath("parameters", "generate_kwargs", "eta_cutoff")),
    )
    max_length: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("max_length", AliasPath("parameters", "generate_kwargs", "max_length")),
    )
    max_new_tokens: Optional[int] = Field(
        None,
        validation_alias=AliasChoices(
            "max_new_tokens", AliasPath("parameters", "generate_kwargs", "max_new_tokens")
        ),
    )
    min_length: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("min_length", AliasPath("parameters", "generate_kwargs", "min_length")),
    )
    min_new_tokens: Optional[int] = Field(
        None,
        validation_alias=AliasChoices(
            "min_new_tokens", AliasPath("parameters", "generate_kwargs", "min_new_tokens")
        ),
    )
    num_beam_groups: Optional[int] = Field(
        None,
        validation_alias=AliasChoices(
            "num_beam_groups", AliasPath("parameters", "generate_kwargs", "num_beam_groups")
        ),
    )
    num_beams: Optional[int] = Field(
        None,
        validation_alias=AliasChoices("num_beams", AliasPath("parameters", "generate_kwargs", "num_beams")),
    )
    penalty_alpha: Optional[float] = Field(
        None,
        validation_alias=AliasChoices(
            "penalty_alpha", AliasPath("parameters", "generate_kwargs", "penalty_alpha")
        ),
    )
    temperature: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("temperature", AliasPath("parameters", "generate_kwargs", "temperature")),
    )
    top_k: Optional[int] = Field(
        None, validation_alias=AliasChoices("top_k", AliasPath("parameters", "generate_kwargs", "top_k"))
    )
    top_p: Optional[float] = Field(
        None, validation_alias=AliasChoices("top_p", AliasPath("parameters", "generate_kwargs", "top_p"))
    )
    typical_p: Optional[float] = Field(
        None,
        validation_alias=AliasChoices("typical_p", AliasPath("parameters", "generate_kwargs", "typical_p")),
    )
    use_cache: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices("use_cache", AliasPath("parameters", "generate_kwargs", "use_cache")),
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                    "parameters": {
                        "return_timestamps": "true",
                    },
                }
            ]
        }
    )


class ASROutputChunk(BaseModel):
    text: str = Field(
        ..., validation_alias=AliasChoices("text", AliasPath("AutomaticSpeechRecognitionOutputChunk", "text"))
    )
    timestamp: Optional[Tuple[float, float]] = Field(
        None,
        validation_alias=AliasChoices(
            "timestamp", AliasPath("AutomaticSpeechRecognitionOutputChunk", "timestamp")
        ),
    )


class ASROutput(BaseModel):
    text: str = Field(
        ..., validation_alias=AliasChoices("text", AliasPath("AutomaticSpeechRecognitionOutput", "text"))
    )
    chunks: Optional[List[ASROutputChunk]] = Field(
        None, validation_alias=AliasChoices("chunks", AliasPath("AutomaticSpeechRecognitionOutput", "chunks"))
    )


class ASR(Predictor[ASRInput, ASROutput]):
    def __init__(self, model_id: str, dtype: str = "float16", device: str = "balanced") -> None:
        super().__init__()

        from transformers import pipeline as transformers_pipeline  # type: ignore

        # apparently some (not all) the models do not support the `device_map=auto` so we should probably
        # either add a check or just default to CUDA instead
        if device == "auto":
            # e.g. DistilBertForSequenceClassification won't support it
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu"

        self.pipeline = transformers_pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=getattr(torch, dtype),
            device=device if device not in {"auto"} else None,
            device_map=device if device in {"auto"} else None,
        )

        if torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.set_per_process_memory_fraction(0.9)

        # first-time "warmup" pass to ensure that the model is ready to start serving requets
        warmup_input = ASRInput(**ASRInput.model_json_schema().get("examples")[0])
        self(warmup_input)

    def __call__(self, input: ASRInput) -> ASROutput:
        payload = input.model_dump(exclude_none=True)
        pipeline_results = self.pipeline(**payload)
        return ASROutput(**pipeline_results)  # type: ignore
