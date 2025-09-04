from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


# TODO: it should accept inputs as a list of inputs to match the current Inference API
class PredictInput(BaseModel):
    sentences: Union[Tuple[str, str], List[str], List[List[str]], List[Tuple[str, str]]]


class PredictOutput(BaseModel):
    scores: List[float]


# TODO: it should accept inputs as a list of inputs to match the current Inference API
class RankInput(BaseModel):
    query: str
    texts: List[str]
    return_documents: Optional[bool] = False


class Score(BaseModel):
    index: int
    score: float
    text: str


class RankOutput(BaseModel):
    scores: List[Score]


# NOTE: most likely not ideal, but we should support both scenarios, so needs to be like this in the meantime
TextRankingInput = Union[PredictInput, RankInput]
TextRankingOutput = Union[PredictOutput, RankOutput]


class TextRanking(Predictor[TextRankingInput, TextRankingOutput]):
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        device: Optional[Literal["cpu", "cuda", "mps", "npu"]] = None,
        backend: Optional[Literal["torch", "onnx", "openvino"]] = "torch",
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = None,
    ) -> None:
        super().__init__()

        import torch
        from sentence_transformers import CrossEncoder

        if device == "mps" and not attn_implementation:
            logger.warning(
                "Device is set to `mps`, so setting `attn_implementation='eager'` by default to prevent potential SDPA-related issues as per https://github.com/UKPLab/sentence-transformers/issues/3498."
            )
            attn_implementation = "eager"

        self.pipeline = CrossEncoder(
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
            activation_fn=torch.nn.Sigmoid(),
        )

    def __call__(self, payload: TextRankingInput) -> TextRankingOutput:
        match payload:
            case PredictInput():
                scores = self.pipeline.predict(payload.sentences, convert_to_tensor=True).tolist()
                return PredictOutput(scores=scores)
            case RankInput():
                scores = self.pipeline.rank(
                    payload.query,
                    payload.texts,
                    return_documents=payload.return_documents,  # type: ignore
                )
                # TODO: can the remapping of `corpus_id` be handled within the Pydantic schema instead?
                # NOTE: here we rename "corpus_id" key to "index" for all scores to match the `/rerank` endpoint
                # in Text Embeddings Inference (TEI)
                # Reference: https://huggingface.github.io/text-embeddings-inference/
                for score in scores:
                    score["index"] = score.pop("corpus_id")  # type: ignore
                return RankOutput(scores=scores)  # type: ignore
