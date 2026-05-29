from typing import Annotated, Any, List, Literal, Optional, Self, Type

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


class SentenceSimilarityInputs(BaseModel):
    source_sentence: Annotated[str, Field(strict=True, min_length=1)]
    sentences: Annotated[List[str], Field(strict=True, min_length=1)]


class SentenceSimilarityParameters(BaseModel):
    prompt_name: Optional[str] = Field(default=None)

    # NOTE: Both `truncate` and `truncation_direction` are not allowed / supported on Sentence Transformers, and
    # given that those are there only due to compatibility with Text Embeddings Inference (TEI) those will be
    # excluded with a warning that those won't have any effect on the underlying `encode` method
    truncate: Optional[bool] = Field(default=None, exclude=True)
    truncation_direction: Optional[Literal["left", "Left", "right", "Right"]] = Field(
        default=None, exclude=True
    )

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


class SentenceSimilarityInput(BaseModel):
    inputs: SentenceSimilarityInputs
    parameters: Optional[SentenceSimilarityParameters] = Field(default=None)

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "inputs": {
                        "source_sentence": "I'm very happy",
                        "sentences": ["I'm filled with happiness", "I'm happy"],
                    },
                    "parameters": {"truncate": True, "truncation_direction": "left", "prompt_name": None},
                },
            ]
        }
    )


class SentenceSimilarityOutput(BaseModel):
    # NOTE: This is most likely wrong and should just return `List[float]` instead, but in order to keep
    # parity with the former Inference API we're keeping it this way for the moment
    similarities: List[List[float]]


class SentenceSimilarity(Predictor[SentenceSimilarityInput, SentenceSimilarityOutput]):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None,
        device: Optional[Literal["auto", "cpu", "cuda", "mps"]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
        # NOTE: specific for sentence similarity
        # TODO: given that some tasks come with specific arguments, eventually rewrite `hf-serve` so that the
        # CLI interface is `hf-serve <TASK> --model-id ...` rather than `hf-serve --model-id ... --task ...`
        similarity_fn_name: Literal["cosine", "dot", "euclidean", "manhattan"] = "cosine",
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

        self.pipeline = SentenceTransformer(
            model_id,
            revision=revision,
            device=device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if device != "auto"
            else None,
            backend=backend,
            model_kwargs=model_kwargs,
            similarity_fn_name=similarity_fn_name,
            trust_remote_code=trust_remote_code,
        )

    def __call__(self, payload: SentenceSimilarityInput) -> SentenceSimilarityOutput:
        parameters = {}
        if payload.parameters:
            parameters = payload.parameters.model_dump(exclude_none=True)
        parameters["convert_to_tensor"] = True

        source_embedding = self.pipeline.encode(payload.inputs.source_sentence, **parameters)
        sentence_embeddings = self.pipeline.encode(payload.inputs.sentences, **parameters)

        similarities = self.pipeline.similarity(source_embedding, sentence_embeddings).tolist()
        return SentenceSimilarityOutput(similarities=similarities)
