from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, Field
from sentence_transformers import SentenceTransformer

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
        device: Optional[Literal["cpu", "cuda", "mps", "npu"]] = None,
        backend: Optional[Literal["torch", "onnx", "openvino"]] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "sdpa",
    ) -> None:
        super().__init__()

        self.pipeline = SentenceTransformer(
            model_id,
            device=device,
            backend=backend or "torch",  # type: ignore
            model_kwargs={
                "torch_dtype": dtype or "float32",
                "attn_implementation": attn_implementation or "sdpa",
            },
        )

    def __call__(self, payload: FeatureExtractionInput) -> FeatureExtractionOutput:
        return FeatureExtractionOutput(
            embeddings=self.pipeline.encode(payload.sentences, convert_to_tensor=True).tolist(),
        )
