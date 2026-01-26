from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException, Request
from PIL.Image import Image as ImageType
from pydantic import ConfigDict, RootModel, ValidationError

from hf_inference_sdk.logging import logger
from hf_inference_sdk.serde import Image


def router(handler: Any) -> APIRouter:
    router = APIRouter()

    class ArbitraryResponse(RootModel):
        root: Any

        model_config = ConfigDict(
            json_encoders={ImageType: Image.serialize},
            arbitrary_types_allowed=True,
        )

    # NOTE: for Inference Endpoints we also need to route to / for the /predict route, as
    # that's the endpoint being hit within the Inference API widgets
    @router.post("/")
    @router.post("/predict")
    async def predict(request: Request, payload: Dict[str, Any] = Body(...)) -> ArbitraryResponse:
        request_id = getattr(request.state, "request_id", None)

        try:
            logger.info(f"[{request_id}] Received request with: {payload}")
            return ArbitraryResponse(root=handler(payload))
        except (ValueError, ValidationError) as e:
            logger.error(f"[{request_id}] Failed validating I/O with: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"[{request_id}] Failed running inference with: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    return router
