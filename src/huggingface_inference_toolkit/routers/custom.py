from typing import Any, Dict

from fastapi import APIRouter, HTTPException


def router(handler: Any) -> APIRouter:
    router = APIRouter()

    # NOTE: for Inference Endpoints we also need to route to / for the /predict route, as
    # that's the endpoint being hit within the Inference API widgets
    @router.post("/")
    @router.post("/predict")
    async def predict(data: Dict[str, Any]) -> Any:
        try:
            return handler(data)
        # TODO(alvarobartt): create better custom exceptions and handle those here with different
        # error codes for I/O validation errors, ser/de errors, or pipeline errors
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
