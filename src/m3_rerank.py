"""Module 3: Reranking -- Cross-encoder top-20 to top-3 + latency benchmark."""

import math
import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._backend = None

    def _load_model(self):
        if self._model is None:
            if not _should_load_neural_model(self.model_name):
                self._model = _LexicalReranker()
                self._backend = "lexical"
                return self._model

            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=True)
                self._backend = "flag"
            except Exception:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                    self._backend = "cross_encoder"
                except Exception:
                    self._model = _LexicalReranker()
                    self._backend = "lexical"
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank retrieved candidates down to top_k."""
        if not documents:
            return []

        model = self._load_model()
        pairs = [(query, doc.get("text", "")) for doc in documents]
        try:
            if self._backend == "flag":
                scores = model.compute_score(pairs)
            else:
                scores = model.predict(pairs)
        except Exception:
            scores = _LexicalReranker().predict(pairs)

        if isinstance(scores, (float, int)):
            scores = [scores]
        scores = [float(score) if _is_finite(score) else 0.0 for score in scores]

        ranked = sorted(
            zip(scores, documents),
            key=lambda item: (item[0], float(item[1].get("score", 0.0))),
            reverse=True,
        )
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(ranked[:top_k], start=1)
        ]


class FlashrankReranker:
    """Lightweight alternative when flashrank is installed."""

    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        try:
            from flashrank import Ranker, RerankRequest
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)

        if self._model is None:
            self._model = Ranker()

        passages = [
            {"id": i, "text": doc.get("text", ""), "meta": doc.get("metadata", {})}
            for i, doc in enumerate(documents)
        ]
        results = self._model.rerank(RerankRequest(query=query, passages=passages))
        by_id = {i: doc for i, doc in enumerate(documents)}
        reranked = []
        for rank, item in enumerate(results[:top_k], start=1):
            doc = by_id.get(item.get("id", rank - 1), {})
            reranked.append(
                RerankResult(
                    text=item.get("text", doc.get("text", "")),
                    original_score=float(doc.get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    rank=rank,
                )
            )
        return reranked


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(max(1, n_runs)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


class _LexicalReranker:
    """Deterministic fallback used when neural reranker weights are unavailable."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [_lexical_score(query, text) for query, text in pairs]


def _lexical_score(query: str, text: str) -> float:
    q_terms = _terms(query)
    d_terms = _terms(text)
    if not q_terms or not d_terms:
        return 0.0
    overlap = q_terms & d_terms
    recall = len(overlap) / len(q_terms)
    precision = len(overlap) / len(d_terms)
    numeric_bonus = 0.2 if set(re.findall(r"\d+", query)) & set(re.findall(r"\d+", text)) else 0.0
    phrase_bonus = 0.2 if any(term in text.lower() for term in q_terms if len(term) > 3) else 0.0
    return recall + 0.5 * precision + numeric_bonus + phrase_bonus


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1
    }


def _is_finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _should_load_neural_model(model_name: str) -> bool:
    if os.getenv("RERANK_FORCE_LEXICAL", "").lower() in {"1", "true", "yes"}:
        return False
    return os.getenv("RERANK_DOWNLOAD_MODEL", "").lower() in {"1", "true", "yes"}


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
