from typing import Any, Dict

import PIL
from fastapi import APIRouter, HTTPException
from pydantic import ConfigDict, RootModel

from hf_serve.serde import Image


def router(handler: Any) -> APIRouter:
    router = APIRouter()

    class ArbitraryResponse(RootModel):
        root: Any
        model_config = ConfigDict(
            json_encoders={PIL.Image.Image: Image.serialize},  # type: ignore
            arbitrary_types_allowed=True,
        )

    # NOTE: for Inference Endpoints we also need to route to / for the /predict route, as
    # that's the endpoint being hit within the Inference API widgets
    @router.post("/")
    @router.post("/predict")
    async def predict(data: Dict[str, Any]) -> ArbitraryResponse:
        try:
            return ArbitraryResponse(root=handler(data))
        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, serde errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
