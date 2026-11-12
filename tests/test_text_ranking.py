import math
import tempfile
import unittest

import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from hf_serve.compatibility.text_embeddings_inference import rerank_router
from hf_serve.routers.predict import router as predict_router
from hf_serve.tasks.sentence_transformers.text_ranking import TextRanking, TextRankingInput, TextRankingOutput

WORDS = ["deep", "learning", "is", "a", "subfield", "of", "machine", "cheese", "tasty", "what"]
QUERY = "what is deep learning"
TEXTS = ["cheese is tasty", "deep learning is a subfield of machine learning"]


def build_cross_encoder(directory: str, num_labels: int = 1) -> None:
    # NOTE: a tiny randomly initialized BERT cross-encoder, so that the tests run offline and deterministically
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    BertTokenizer(
        vocab={token: index for index, token in enumerate(tokens)}, model_max_length=32
    ).save_pretrained(directory)
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(tokens),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
        num_labels=num_labels,
    )
    BertForSequenceClassification(config).save_pretrained(directory)


class TextRankingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        build_cross_encoder(cls.tmp.name)
        predictor = TextRanking(model_id=cls.tmp.name, device="cpu")
        app = FastAPI()
        app.include_router(predict_router(predictor, TextRankingInput, TextRankingOutput))
        app.include_router(rerank_router(predictor))
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def rerank(self, **body):
        return self.client.post("/rerank", json={"query": QUERY, "texts": TEXTS, **body})

    def test_rerank_returns_tei_ranks(self) -> None:
        response = self.rerank()
        self.assertEqual(response.status_code, 200, response.text)
        ranks = response.json()
        self.assertEqual(sorted(rank["index"] for rank in ranks), [0, 1])
        self.assertEqual(
            [rank["score"] for rank in ranks], sorted((rank["score"] for rank in ranks), reverse=True)
        )
        self.assertTrue(all(set(rank) == {"index", "score"} for rank in ranks))

        with_text = self.rerank(return_text=True).json()
        self.assertEqual({rank["index"]: rank["text"] for rank in with_text}, dict(enumerate(TEXTS)))

    def test_raw_scores_return_logits(self) -> None:
        for raw_scores in ({"raw_scores": True}, {"parameters": {"raw_scores": True}}):
            with self.subTest(raw_scores=raw_scores):
                scores = {rank["index"]: rank["score"] for rank in self.rerank().json()}
                logits = {rank["index"]: rank["score"] for rank in self.rerank(**raw_scores).json()}
                for index, logit in logits.items():
                    self.assertAlmostEqual(1 / (1 + math.exp(-logit)), scores[index], places=5)
                self.assertNotEqual(logits, scores)

    def test_truncate_false_rejects_long_inputs(self) -> None:
        long_text = " ".join(["deep"] * 64)
        response = self.client.post("/rerank", json={"query": QUERY, "texts": [long_text], "truncate": False})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error_type"], "Validation")
        self.assertIn("must have less than 32 tokens", response.json()["error"])

        for truncate in ({}, {"truncate": True}):
            with self.subTest(truncate=truncate):
                body = {"query": QUERY, "texts": [long_text], **truncate}
                self.assertEqual(self.client.post("/rerank", json=body).status_code, 200)

        # NOTE: inputs within the limit are accepted even when truncation is disabled
        self.assertEqual(self.rerank(truncate=False).status_code, 200)

    def test_truncation_direction(self) -> None:
        text = " ".join(["cheese tasty"] * 10 + ["deep learning machine"] * 10)
        scores = {}
        for direction in ("Left", "left", "Right", "right"):
            response = self.client.post(
                "/rerank",
                json={"query": QUERY, "texts": [text], "raw_scores": True, "truncation_direction": direction},
            )
            self.assertEqual(response.status_code, 200, response.text)
            scores[direction] = response.json()[0]["score"]
        self.assertEqual(scores["Left"], scores["left"])
        self.assertEqual(scores["Right"], scores["right"])
        self.assertNotEqual(scores["left"], scores["right"])
        # NOTE: the tokenizer truncation side is restored after the request
        self.assertEqual(self.rerank(raw_scores=True).json(), self.rerank(raw_scores=True).json())
        self.assertEqual(
            self.client.post("/rerank", json={"query": QUERY, "texts": [text], "raw_scores": True}).json()[0][
                "score"
            ],
            scores["right"],
        )

    def test_predict_route_keeps_wrapped_scores(self) -> None:
        response = self.client.post("/predict", json={"query": QUERY, "texts": TEXTS, "raw_scores": True})
        self.assertEqual(response.status_code, 200, response.text)
        expected = self.rerank(raw_scores=True).json()
        self.assertEqual(response.json(), {"scores": [{**rank, "text": None} for rank in expected]})


class SentencePairScoresTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.clients = {}
        for num_labels in (1, 3):
            directory = f"{cls.tmp.name}/{num_labels}"
            build_cross_encoder(directory, num_labels=num_labels)
            predictor = TextRanking(model_id=directory, device="cpu")
            app = FastAPI()
            app.include_router(predict_router(predictor, TextRankingInput, TextRankingOutput))
            cls.clients[num_labels] = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def predict(self, num_labels: int, sentences):
        return self.clients[num_labels].post("/predict", json={"sentences": sentences})

    def test_returns_one_entry_per_pair(self) -> None:
        pairs = [[QUERY, text] for text in TEXTS]
        for num_labels in (1, 3):
            with self.subTest(num_labels=num_labels):
                single = [self.predict(num_labels, pair).json()["scores"] for pair in pairs]
                wrapped = [self.predict(num_labels, [pair]).json()["scores"] for pair in pairs]
                batched = self.predict(num_labels, pairs).json()["scores"]

                self.assertEqual(len(batched), len(pairs))
                for scores in batched:
                    if num_labels == 1:
                        self.assertIsInstance(scores, float)
                    else:
                        self.assertEqual(len(scores), num_labels)
                # NOTE: a pair gets the same scores whether sent alone, wrapped in a list, or batched
                for index in range(len(pairs)):
                    self.assertEqual(single[index], wrapped[index])
                    self.assertEqual(len(single[index]), 1)
                    torch.testing.assert_close(torch.tensor(single[index][0]), torch.tensor(batched[index]))

    def test_rejects_malformed_pairs(self) -> None:
        for sentences in (["a", "b", "c"], ["a"], [], [["a"]], [["a", "b", "c"]], [["a", "b"], ["c"]]):
            with self.subTest(sentences=sentences):
                self.assertEqual(self.predict(1, sentences).status_code, 422)


if __name__ == "__main__":
    unittest.main()
