import unittest

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.testclient import TestClient

from hf_serve.compatibility.sagemaker import SageMakerRoutingMiddleware, post_paths


def create_app(media_routes: bool) -> FastAPI:
    app = FastAPI()

    @app.post("/predict")
    async def predict(request: Request):
        if media_routes:
            return RedirectResponse("/predict-json")
        return {"route": request.url.path}

    if media_routes:

        @app.post("/predict-json")
        @app.post("/predict-form")
        @app.post("/predict-file")
        async def predict_media(request: Request):
            return {"route": request.url.path, "body": (await request.body()).decode("utf-8")}

    app.add_middleware(SageMakerRoutingMiddleware, invocation_paths=post_paths(app.routes))
    return app


class SageMakerMediaRoutingTests(unittest.TestCase):
    def test_default_json_goes_directly_to_predict_json(self):
        client = TestClient(create_app(media_routes=True), follow_redirects=False)

        response = client.post("/invocations", json={"inputs": "audio"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"route": "/predict-json", "body": '{"inputs":"audio"}'})
        self.assertEqual(response.history, [])

    def test_default_form_and_file_use_matching_media_routes(self):
        client = TestClient(create_app(media_routes=True), follow_redirects=False)

        form = client.post(
            "/invocations",
            content=b"form-data",
            headers={"Content-Type": "multipart/form-data; boundary=example"},
        )
        file = client.post("/invocations", content=b"audio", headers={"Content-Type": "audio/wav"})

        self.assertEqual(form.json(), {"route": "/predict-form", "body": "form-data"})
        self.assertEqual(file.json(), {"route": "/predict-file", "body": "audio"})

    def test_explicit_route_takes_precedence(self):
        client = TestClient(create_app(media_routes=True), follow_redirects=False)

        response = client.post(
            "/invocations",
            json={"inputs": "audio"},
            headers={"X-Amzn-SageMaker-Custom-Attributes": "route=predict-file"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["route"], "/predict-file")

    def test_non_media_task_keeps_predict_default(self):
        client = TestClient(create_app(media_routes=False), follow_redirects=False)

        response = client.post("/invocations", json={"inputs": "text"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"route": "/predict"})


if __name__ == "__main__":
    unittest.main()
