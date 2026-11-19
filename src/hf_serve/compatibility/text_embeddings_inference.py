from typing import List

from fastapi import APIRouter, Body, Request
from fastapi.responses import JSONResponse

from hf_serve.logging import logger
from hf_serve.tasks.sentence_transformers.text_ranking import RankInput, Score, TextRanking

router = APIRouter()


# https://github.com/huggingface/text-embeddings-inference/blob/ebb63dfa7121705f1999a06d8e222581a5221c00/router/src/http/server.rs#L1834
@router.get("/api-doc/openapi.json")
async def openapi_json(request: Request) -> JSONResponse:
    return request.app.openapi()


def rerank_router(predictor: TextRanking) -> APIRouter:
    rerank_router = APIRouter()

    # NOTE: mirrors TEI's `/rerank`, which returns a bare list of ranks (omitting `text` unless `return_text=True`)
    # and reports errors as `{"error": ..., "error_type": ...}`
    # Reference: https://huggingface.github.io/text-embeddings-inference/#/Text%20Embeddings%20Inference/rerank
    @rerank_router.post("/rerank", response_model=List[Score], response_model_exclude_none=True)
    async def rerank(request: Request, payload: RankInput = Body(...)) -> List[Score]:
        request_id = getattr(request.state, "request_id", None)
        try:
            return predictor(payload=payload).scores  # type: ignore
        except ValueError as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            return JSONResponse(  # type: ignore
                status_code=422, content={"error": f"Input validation error: {e}", "error_type": "Validation"}
            )
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            return JSONResponse(status_code=424, content={"error": str(e), "error_type": "Backend"})  # type: ignore

    return rerank_router
