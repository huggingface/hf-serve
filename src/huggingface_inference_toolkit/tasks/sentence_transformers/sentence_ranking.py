from typing import List, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

from huggingface_inference_toolkit.tasks.predictor import Predictor


class PredictInput(BaseModel):
    sentences: Union[Tuple[str, str], List[str], List[List[str]], List[Tuple[str, str]]]


class PredictOutput(BaseModel):
    scores: List[List[float]]


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
SentenceRankingInput = Union[PredictInput, RankInput]
SentenceRankingOutput = Union[PredictOutput, RankOutput]


class SentenceRanking(Predictor[SentenceRankingInput, SentenceRankingOutput]):
    def __init__(
        self,
        model_id: str,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = "float32",
        device: Optional[Literal["cpu", "cuda", "mps", "npu"]] = None,
        attn_implementation: Optional[Literal["eager", "sdpa", "flash_attention_2"]] = "eager",
    ) -> None:
        # TODO: maybe add support for every argument (?)
        super().__init__()

        self.pipeline = CrossEncoder(
            model_id,
            device=device,
            automodel_args={
                "torch_dtype": dtype or "float32",
                "attn_implementation": attn_implementation or "eager",
            },
            default_activation_function=torch.nn.Sigmoid(),
        )

    # TODO: rename `input` to `payload` as `input` is a reserved keyword
    def __call__(self, input: SentenceRankingInput) -> SentenceRankingOutput:
        match input:
            case PredictInput():
                scores = self.pipeline.predict(input.sentences, convert_to_tensor=True).tolist()
                return PredictOutput(scores=scores)
            case RankInput():
                scores = self.pipeline.rank(input.query, input.texts, return_documents=input.return_documents)  # type: ignore
                # NOTE: here we rename "corpus_id" key to "index" for all scores to match TEI
                for score in scores:
                    score["index"] = score.pop("corpus_id")  # type: ignore
                return RankOutput(scores=scores)  # type: ignore
