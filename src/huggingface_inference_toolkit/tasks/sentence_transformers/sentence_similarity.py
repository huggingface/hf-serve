import os
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, Field
from sentence_transformers import SentenceTransformer

from huggingface_inference_toolkit.tasks.predictor import Predictor


class SentenceSimilarityInput(BaseModel):
    source_sentence: str = Field(
        validation_alias=AliasChoices(
            "source_sentence", AliasPath("inputs", "source_sentence"), AliasPath("sentence")
        )
    )
    sentences: List[str] = Field(
        validation_alias=AliasChoices(
            "sentences", AliasPath("inputs", "sentences"), AliasPath("target_sentences")
        )
    )


class SentenceSimilarityOutput(BaseModel):
    similarities: List[List[float]]


class SentenceSimilarity(Predictor[SentenceSimilarityInput, SentenceSimilarityOutput]):
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        device: Optional[Literal["cpu", "cuda", "mps", "npu"]] = None,
        backend: Optional[Literal["torch", "onnx", "openvino"]] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "sdpa",
        # NOTE: specific for sentence similarity
        similarity_fn_name: Optional[Literal["cosine", "dot", "euclidean", "manhattan"]] = "cosine",
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
            similarity_fn_name=similarity_fn_name or "cosine",
        )

    @property
    def model_id(self) -> Union[str, None]:
        return (
            self.pipeline.tokenizer.name_or_path
            if not Path(self.pipeline.tokenizer.name_or_path).exists()
            else os.getenv("MODEL_ID")
        )

    def __call__(self, payload: SentenceSimilarityInput) -> SentenceSimilarityOutput:
        source_sentence_emb = self.pipeline.encode(payload.source_sentence, convert_to_tensor=True)
        sentence_embs = self.pipeline.encode(payload.sentences, convert_to_tensor=True)
        return SentenceSimilarityOutput(
            similarities=self.pipeline.similarity(source_sentence_emb, sentence_embs).tolist()
        )
