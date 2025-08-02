import os
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, Field
from sentence_transformers import SentenceTransformer

from huggingface_inference_toolkit.tasks.predictor import Predictor


class SentenceEmbeddingsInput(BaseModel):
    sentences: Union[str, List[str]] = Field(
        validation_alias=AliasChoices("sentence", AliasPath("inputs", "sentence"))
    )
    normalize: Optional[bool] = Field(
        default=False,
        validation_alias=AliasChoices("normalize", AliasPath("normalize_embeddings")),
    )


class SentenceEmbeddingsOutput(BaseModel):
    embeddings: List[float]


class SentenceEmbeddings(Predictor[SentenceEmbeddingsInput, SentenceEmbeddingsOutput]):
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
    
    @property
    def model_id(self) -> Union[str, None]:
        return (
            self.pipeline.tokenizer.name_or_path
            if not Path(self.pipeline.tokenizer.name_or_path).exists()
            else os.getenv("MODEL_ID")
        )

    def __call__(self, payload: SentenceEmbeddingsInput) -> SentenceEmbeddingsOutput:
        return SentenceEmbeddingsOutput(
            embeddings=self.pipeline.encode(payload.sentences, convert_to_tensor=True).tolist(),
        )

