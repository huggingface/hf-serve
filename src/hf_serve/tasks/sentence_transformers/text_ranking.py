from typing import List, Literal, Optional, Union

from pydantic import AliasChoices, AliasPath, BaseModel, ConfigDict, Field, field_validator

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
    # NOTE: the parameters below follow the `/rerank` endpoint in Text Embeddings Inference (TEI), and can be
    # provided either at the top level (as in TEI) or within `parameters`
    # Reference: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    raw_scores: bool = Field(
        default=False, validation_alias=AliasChoices("raw_scores", AliasPath("parameters", "raw_scores"))
    )
    # NOTE: `None` truncates, matching TEI where `truncate` falls back to `--auto-truncate` (enabled by default),
    # whilst `False` rejects inputs longer than the model's maximum sequence length
    truncate: Optional[bool] = Field(
        default=None, validation_alias=AliasChoices("truncate", AliasPath("parameters", "truncate"))
    )
    truncation_direction: Literal["left", "right"] = Field(
        default="right",
        validation_alias=AliasChoices("truncation_direction", AliasPath("parameters", "truncation_direction")),
    )

    @field_validator("truncation_direction", mode="before")
    @classmethod
    def lowercase_truncation_direction(cls, value: object) -> object:
        # NOTE: TEI accepts both "Left"/"Right" (as per its OpenAPI schema) and "left"/"right"
        return value.lower() if isinstance(value, str) else value

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
        revision: Optional[str] = None,
        dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None,
        device: Optional[Literal["auto", "cpu", "cuda", "mps"]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
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
            # NOTE: Default to `eager` instead of `sdpa`, even if `sdpa` tends to be supported and more
            # performant, there are still some models that won't support it e.g. `sentence-transformers/all-mpnet-base-v2`
            "attn_implementation": attn_implementation or "eager",
        }
        if device == "auto":
            model_kwargs["device_map"] = device

        self.pipeline = CrossEncoder(
            model_id,
            revision=revision,
            device=device
            or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if device != "auto"
            else None,
            backend=backend,
            model_kwargs=model_kwargs,
            activation_fn=torch.nn.Sigmoid(),
            trust_remote_code=trust_remote_code,
        )
        # NOTE: used instead of the default `Sigmoid` activation when `raw_scores=True`, to return the logits
        self.identity = torch.nn.Identity()

    def __call__(self, payload: TextRankingInput) -> TextRankingOutput:
        match payload:
            case PredictInput():
                scores = self.pipeline.predict(inputs=payload.sentences, convert_to_tensor=True)
                if scores.ndim < 2:
                    scores = scores.unsqueeze(dim=0)
                return PredictOutput(scores=scores.tolist())

            case RankInput():
                tokenizer = self.pipeline.tokenizer
                if payload.truncate is False and payload.texts:
                    max_length = self.pipeline.max_seq_length
                    encodings = tokenizer([payload.query] * len(payload.texts), payload.texts)
                    length = max(len(input_ids) for input_ids in encodings["input_ids"])
                    if length > max_length:
                        raise ValueError(f"`inputs` must have less than {max_length} tokens. Given: {length}")

                # NOTE: the tokenizer is shared, so the truncation side is restored after ranking; this is safe as
                # the routes run the predictor synchronously within the event loop, so requests don't overlap
                truncation_side = tokenizer.truncation_side
                tokenizer.truncation_side = payload.truncation_direction
                try:
                    scores = self.pipeline.rank(
                        query=payload.query,
                        documents=payload.texts,
                        return_documents=payload.return_documents,
                        activation_fn=self.identity if payload.raw_scores else None,
                    )
                finally:
                    tokenizer.truncation_side = truncation_side
                return RankOutput(scores=scores)  # type: ignore
