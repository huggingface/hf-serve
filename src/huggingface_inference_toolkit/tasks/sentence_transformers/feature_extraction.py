from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, Field

from huggingface_inference_toolkit.logging import logger
from huggingface_inference_toolkit.tasks.predictor import Predictor


class FeatureExtractionInput(BaseModel):
    sentences: Union[str, List[str]] = Field(
        validation_alias=AliasChoices("sentence", AliasPath("inputs", "sentence"))
    )
    normalize: Optional[bool] = Field(
        default=False,
        validation_alias=AliasChoices("normalize", AliasPath("normalize_embeddings")),
    )


class FeatureExtractionOutput(BaseModel):
    embeddings: List[float]


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

        self.pipeline = SentenceTransformer(
            model_id,
            device=device,
            backend=backend or "torch",  # type: ignore
            model_kwargs={
                "torch_dtype": dtype or "float32",
                # TODO: use `flash_attention_2` depending on compute capability and whether it's installed or not
                "attn_implementation": attn_implementation or "sdpa",
            },
        )

    def __call__(self, payload: FeatureExtractionInput) -> FeatureExtractionOutput:
        return FeatureExtractionOutput(
            embeddings=self.pipeline.encode(payload.sentences, convert_to_tensor=True).tolist(),
        )
