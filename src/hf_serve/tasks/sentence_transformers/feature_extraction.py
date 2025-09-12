from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


# NOTE: won't define e.g. `FeatureExtractionParameters` per se to keep parity with
# Text Embeddings Inference (TEI)
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
        validation_alias=AliasChoices("normalize", AliasPath("parameters", "normalize")),
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
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        device: Optional[Literal["cpu", "cuda", "mps"]] = None,
        backend: Optional[Literal["torch", "onnx", "openvino"]] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
    ) -> None:
        super().__init__()

        from sentence_transformers import SentenceTransformer

        if device == "mps" and not attn_implementation:
            logger.warning(
                "Device is set to `mps`, so setting `attn_implementation='eager'` by default to prevent potential SDPA-related issues as per https://github.com/UKPLab/sentence-transformers/issues/3498."
            )
            attn_implementation = "eager"

        # TODO: print the initialization arguments in advance to let the user know and provide enough transparency
        # as the idea is to make sure that the logging messages are useful and actionable
        self.pipeline = SentenceTransformer(
            model_id,
            device=device,
            backend=backend or "torch",  # type: ignore
            model_kwargs={
                # NOTE: `torch_dtype` to be deprecated in favour of `dtype` as Transformers will be PyTorch-only
                # and Sentence Transformers raises a warning starting on 5.1.0
                "torch_dtype": dtype or "float32",
                # TODO: use `flash_attention_2` depending on compute capability and whether it's installed or not
                "attn_implementation": attn_implementation or "sdpa",
            },
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
