from dataclasses import dataclass
from typing import List

from app.models import DocumentChunk
from app.retrieval.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    chunk: DocumentChunk
    score: float
    authority_rank: int
    status_rank: int


class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        min_score: float = 0.15,
    ):
        self.vector_store = vector_store
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievedChunk]:

        candidates = self.vector_store.search(
            query,
            top_k=top_k,
        )

        filtered = [
            (chunk, score)
            for chunk, score in candidates
            if score >= self.min_score
        ]

        results = []

        for chunk, score in filtered:
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=score,
                    authority_rank=self._authority_rank(
                        chunk
                    ),
                    status_rank=self._status_rank(
                        chunk
                    ),
                )
            )

        results.sort(
            key=lambda result: (
                result.status_rank,
                result.authority_rank,
                result.score,
            ),
            reverse=True,
        )

        return results

    @staticmethod
    def _authority_rank(chunk: DocumentChunk) -> int:
        authority = (
            chunk.policy_authority or ""
        ).lower()

        if authority == "official":
            return 2

        if authority:
            return 1

        return 0

    @staticmethod
    def _status_rank(chunk: DocumentChunk) -> int:
        status = (
            chunk.status or ""
        ).lower()

        if status == "active":
            return 3

        if status == "draft":
            return 2

        if status == "superseded":
            return 1

        return 0