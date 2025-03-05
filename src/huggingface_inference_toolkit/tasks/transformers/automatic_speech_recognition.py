from typing import Annotated, List, Literal, Optional, Union

import torch
from pydantic import AliasChoices, AliasPath, BaseModel, BeforeValidator, ConfigDict, Field, RootModel

from huggingface_inference_toolkit.tasks.predictor import Predictor


class ASRInput(BaseModel):
    inputs: str = Field()

    return_timestamps: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices(
            "return_timestamps", AliasPath("AutomaticSpeechRecognitionParameters", "return_timestamps")
        ),
    )
    do_sample: Optional[bool] = Field(
        None,
        validation_alias=AliasChoices(
            "do_sample",
            AliasPath(
                "AutomaticSpeechRecognitionParameters",
                "AutomaticSpeechRecognitionGenerationParameters",
                "do_sample",
            ),
        ),
    )
    early_stopping: Optional[Union[bool, Literal["never"]]] = Field(
        None,
        validation_alias=AliasChoices(
            "early_stopping",
            AliasPath(
                "AutomaticSpeechRecognitionParameters",
                "AutomaticSpeechRecognitionGenerationParameters",
                "early_stopping",
            ),
        ),
    )

    # under AutomaticSpeechRecognitionParameters and AutomaticSpeechRecognitionGenerationParameters
    # do_sample: Optional[bool] = None
    # early_stopping: Optional[Union[bool, AutomaticSpeechRecognitionEarlyStoppingEnum]] = None
    # epsilon_cutoff: Optional[float] = None
    # eta_cutoff: Optional[float] = None
    # max_length: Optional[int] = None
    # max_new_tokens: Optional[int] = None
    # min_length: Optional[int] = None
    # min_new_tokens: Optional[int] = None
    # num_beam_groups: Optional[int] = None
    # num_beams: Optional[int] = None
    # penalty_alpha: Optional[float] = None
    # temperature: Optional[float] = None
    # top_k: Optional[int] = None
    # top_p: Optional[float] = None
    # typical_p: Optional[float] = None
    # use_cache: Optional[bool] = None

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/1.flac",
                }
            ]
        }
    )


class ASROutputValue(BaseModel):
    text: str

    # these are under the AutomaticSpeechRecognitionOutputChunk
    # text: Optional[str] = Field(
    #     None,
    #     validation_alias=AliasChoices("text", AliasPath("AutomaticSpeechRecognitionOutputChunk", "text"))
    # )
    # timestamp: List[float] = Field(
    #     None,
    #     validation_alias=AliasChoices("timestamp", AliasPath("AutomaticSpeechRecognitionOutputChunk", "timestamp"))
    # )


class ASROutput(RootModel):
    root: Annotated[
        List[ASROutputValue],
        BeforeValidator(lambda value: [value] if not isinstance(value, list) else value),
    ]


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

        print(warmup_input)

        self(warmup_input)

    def __call__(self, input: ASRInput) -> ASROutput:
        payload = input.model_dump(exclude_none=True)

        print(payload)

        pipeline_results = self.pipeline(**payload)

        print(pipeline_results)

        return ASROutput(root=pipeline_results)  # type: ignore
