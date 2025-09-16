from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


class SentenceSimilarityInputs(BaseModel):
    source_sentence: str
    sentences: List[str]


class SentenceSimilarityParameters(BaseModel): ...


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
                    "parameters": None,
                },
            ]
        }
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
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
        # NOTE: specific for sentence similarity
        # TODO: given that some tasks come with specific arguments, eventually rewrite `hf-serve` so that the
        # CLI interface is `hf-serve <TASK> --model-id ...` rather than `hf-serve --model-id ... --task ...`
        similarity_fn_name: Optional[Literal["cosine", "dot", "euclidean", "manhattan"]] = "cosine",
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
                # NOTE: `torch_dtype` to be deprecated in favour of `dtype` as Transformers will be PyTorch-only
                # and Sentence Transformers raises a warning starting on 5.1.0
                "dtype": dtype or "float32",
                # TODO: use `flash_attention_2` depending on compute capability and whether it's installed or not
                "attn_implementation": attn_implementation or "sdpa",
            },
            similarity_fn_name=similarity_fn_name or "cosine",
        )

    def __call__(self, payload: SentenceSimilarityInput) -> SentenceSimilarityOutput:
        source_embedding = self.pipeline.encode(payload.inputs.source_sentence, convert_to_tensor=True)
        sentence_embeddings = self.pipeline.encode(payload.inputs.sentences, convert_to_tensor=True)

        similarities = self.pipeline.similarity(source_embedding, sentence_embeddings).tolist()
        return SentenceSimilarityOutput(similarities=similarities)
