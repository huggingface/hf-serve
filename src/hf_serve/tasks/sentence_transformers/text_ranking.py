from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field

from hf_serve.logging import logger
from hf_serve.tasks.predictor import Predictor


# TODO: it should accept inputs as a list of inputs to match the current Inference API
class PredictInput(BaseModel):
    # NOTE: this originally allows tuples too, but that's invalid in JSON, hence removed
    sentences: Union[List[str], List[List[str]]]

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "sentences": ["What is Deep Learning?", "What is not Deep Learning?"],
                },
                {
                    "sentences": [["What is Deep Learning?", "What is not Deep Learning?"]],
                },
            ]
        }
    )


class PredictOutput(BaseModel):
    scores: List[float]


# TODO: it should accept inputs as a list of inputs to match the current Inference API
# TODO: given that the requests can increase in size we should try to truncate all the str fields
# to e.g. 500 characters when logging those as e.g. `query: ...[TRUNCATED]`
class RankInput(BaseModel):
    query: str = Field(validation_alias=AliasChoices("query", AliasPath("inputs", "query")))
    texts: List[str] = Field(
        validation_alias=AliasChoices(
            "texts", "documents", AliasPath("inputs", "texts"), AliasPath("inputs", "documents")
        )
    )

    return_documents: bool = Field(
        default=False,
        validation_alias=AliasChoices(
            "return_documents", "return_text", AliasPath("parameters", "return_documents")
        ),
    )
    # NOTE: the parameters below are defined in Text Embeddings Inference (TEI) but unsupported natively within
    # the `CrossEncoder.rank` method, hence are currently useless
    raw_scores: bool = Field(default=False, validation_alias=AliasPath("parameters", "raw_scores"))
    truncate: bool = Field(default=False, validation_alias=AliasPath("parameters", "truncate"))
    truncation_direction: Literal["left", "right"] = Field(
        default="right", validation_alias=AliasPath("parameters", "truncation_direction")
    )

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "query": "What is Deep Learning?",
                    "texts": ["Deep Learning is...", "Deep Learning is not ..."],
                    "parameters": {"return_documents": True},
                },
            ]
        }
    )


class Score(BaseModel):
    # NOTE: here we rename "corpus_id" key to "index" for all scores to match the `/rerank` endpoint
    # in Text Embeddings Inference (TEI)
    # Reference: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    index: int = Field(validation_alias=AliasChoices("index", "corpus_id"))
    score: float
    # NOTE: `text` is optional as it won't be provided is `return_documents=False`
    text: Optional[str] = Field(default=None)


class RankOutput(BaseModel):
    scores: List[Score]


# NOTE: most likely not ideal, but we should support both scenarios, so needs to be like this in the meantime
TextRankingInput = Union[PredictInput, RankInput]
TextRankingOutput = Union[PredictOutput, RankOutput]


class TextRanking(Predictor[TextRankingInput, TextRankingOutput]):
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
        from sentence_transformers import CrossEncoder

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

        self.pipeline = CrossEncoder(
            model_id,
            device=device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if device != "auto"
            else None,
            backend=backend,
            model_kwargs=model_kwargs,
            activation_fn=torch.nn.Sigmoid(),
        )

    def __call__(self, payload: TextRankingInput) -> TextRankingOutput:
        match payload:
            case PredictInput():
                scores = self.pipeline.predict(sentences=payload.sentences, convert_to_tensor=True)
                if scores.ndim < 2:
                    scores = scores.unsqueeze(dim=0)
                return PredictOutput(scores=scores.tolist())

            case RankInput():
                scores = self.pipeline.rank(
                    query=payload.query,
                    documents=payload.texts,
                    return_documents=payload.return_documents,
                )
                return RankOutput(scores=scores)  # type: ignore
