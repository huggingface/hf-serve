from typing import Any, List, Literal, Optional, Self, Type, Union

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, field_validator, model_validator

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


# NOTE: `FeatureExtraction` won't define the input payload as `inputs` and `parameters` to keep parity with the
# Text Embeddings Inference (TEI) counterpart which is the default on Inference Endpoints API, whilst still adding
# the required aliases to prevent from being disruptive with the rest of the input payloads along `hf-serve`
# Reference: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/embed
class FeatureExtractionInput(BaseModel):
    sentences: Union[str, List[str]] = Field(
        validation_alias=AliasChoices("sentences", "inputs", AliasPath("inputs", "sentences"))
    )

    normalize: bool = Field(
        default=True,
        validation_alias=AliasChoices("normalize", AliasPath("parameters", "normalize")),
    )
    dimensions: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("dimensions", AliasPath("parameters", "dimensions")),
    )
    prompt_name: Optional[str] = Field(
        default=None, validation_alias=AliasChoices("prompt_name", AliasPath("parameters", "prompt_name"))
    )

    # NOTE: Both `truncate` and `truncation_direction` are not allowed / supported on Sentence Transformers, and
    # given that those are there only due to compatibility with Text Embeddings Inference (TEI) those will be
    # excluded with a warning that those won't have any effect on the underlying `encode` method
    truncate: Optional[bool] = Field(default=None, exclude=True)
    truncation_direction: Optional[Literal["left", "Left", "right", "Right"]] = Field(
        default=None, exclude=True
    )

    @field_validator("sentences", mode="after")
    @classmethod
    def validate_sentences(cls: Type[Self], v: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(v, str):
            if len(v) < 1:
                raise ValueError("When `sentences` is provided as a string, it must not be empty")
        elif isinstance(v, list):
            if len(v) == 0:
                raise ValueError("When `sentences` is provided as a list, it must not be empty")
            if not all(isinstance(s, str) and len(s) > 0 for s in v):
                raise ValueError(
                    "When `sentences` is provided as a list, all the items must be non-empty strings"
                )
        return v

    @model_validator(mode="before")
    @classmethod
    def warn_unsupported_fields(cls: Type[Self], values: Any) -> Any:
        # NOTE: Here using `any` is fine since we don't want to loop over both values, meaning that if `truncate`
        # is provided we show the warning without waiting for the confirmation on whether `truncation_direction`
        # is there or not, as we don't care as long as 1 is indeed provided. Also no need for removing those as
        # we're using `exclude=True` already
        if any(values.get(k, None) is not None for k in {"truncate", "truncation_direction"}):
            logger.warning(
                "Neither `truncate` nor `truncation_direction` are supported fields for `SentenceTransformer.encode`, hence those will be ignored."
            )
        return values

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": "What is Deep Learning?",
                    "parameters": {"normalize": True},
                },
                {
                    "sentences": "What is Deep Learning?",
                    "normalize": True,
                },
            ]
        }
    )


class FeatureExtractionOutput(BaseModel):
    # NOTE: for this to be aligned with Text Embeddings Inference (TEI) it should rather be `root` i.e.
    # given that's a single output it shouldn't contain a key "embeddings"
    embeddings: List[List[float]]


class FeatureExtraction(Predictor[FeatureExtractionInput, FeatureExtractionOutput]):
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None,
        device: Optional[Literal["auto", "cpu", "cuda", "mps"]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
        trust_remote_code: bool = False,
    ) -> None:
        super().__init__()

        import torch
        from sentence_transformers import SentenceTransformer

        if (
            device == "mps" or (device == "auto" and torch.backends.mps.is_available())
        ) and not attn_implementation:
            logger.warning(
                "Device is set to `mps` (or `auto` with MPS being available), so setting `attn_implementation='eager'` by default to prevent potential SDPA-related issues as per https://github.com/UKPLab/sentence-transformers/issues/3498."
            )
            attn_implementation = "eager"

        model_kwargs = {
            # NOTE: `torch_dtype` to be deprecated in favour of `dtype` as Transformers will be PyTorch-only
            # and Sentence Transformers raises a warning starting on 5.1.0
            "dtype": dtype or "auto",
            # TODO: use `flash_attention_2` depending on compute capability and whether it's installed or not
            # NOTE: Default to `eager` instead of `sdpa`, even if `sdpa` tends to be supported and more
            # performant, there are still some models that won't support it e.g. `sentence-transformers/all-mpnet-base-v2`
            "attn_implementation": attn_implementation or "eager",
        }
        if device == "auto":
            model_kwargs["device_map"] = device

        # TODO: print the initialization arguments in advance to let the user know and provide enough transparency
        # as the idea is to make sure that the logging messages are useful and actionable
        # TODO: add support for `SparseEncoder` models
        self.pipeline = SentenceTransformer(
            model_id,
            device=device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if device != "auto"
            else None,
            backend=backend,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )

    def __call__(self, payload: FeatureExtractionInput) -> FeatureExtractionOutput:
        # NOTE: Exclude `sentences` because it's the input i.e. not a parameter
        parameters = payload.model_dump(exclude={"sentences"}, exclude_none=True, exclude_defaults=True)

        if dimensions := parameters.pop("dimensions", None):
            parameters["truncate_dim"] = dimensions
        if normalize_embeddings := parameters.pop("normalize", None):
            parameters["normalize_embeddings"] = normalize_embeddings
        parameters["convert_to_tensor"] = True

        embeddings = self.pipeline.encode(payload.sentences, **parameters)

        # NOTE: if embeddings doesn't contain the batch dimension i.e., the provided `inputs` only contains a
        # single sequence instead of a batch, then we add the batch dimension to make sure the outputs are
        # consistent
        if embeddings.ndim < 2:
            embeddings = embeddings.unsqueeze(dim=0)

        return FeatureExtractionOutput(embeddings=embeddings.tolist())
