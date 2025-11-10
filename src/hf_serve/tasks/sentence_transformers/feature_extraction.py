from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

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
    truncate: bool = Field(
        default=False, validation_alias=AliasChoices("truncate", AliasPath("parameters", "truncate"))
    )
    truncation_direction: Literal["left", "right"] = Field(
        default="right",
        validation_alias=AliasChoices("truncation_direction", AliasPath("parameters", "truncation_direction")),
    )

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
            "attn_implementation": attn_implementation or "sdpa",
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
        )

    def __call__(self, payload: FeatureExtractionInput) -> FeatureExtractionOutput:
        payload_json = payload.model_dump(exclude_none=True, exclude_defaults=True)
        embeddings = self.pipeline.encode(**payload_json, convert_to_tensor=True)

        # NOTE: if embeddings doesn't contain the batch dimension i.e., the provided `inputs` only contains a
        # single sequence instead of a batch, then we add the batch dimension to make sure the outputs are
        # consistent
        if embeddings.ndim < 2:
            embeddings = embeddings.unsqueeze(dim=0)

        return FeatureExtractionOutput(embeddings=embeddings.tolist())
