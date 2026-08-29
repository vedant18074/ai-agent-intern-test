from typing import List, Tuple

import faiss
import numpy as np

from app.models import DocumentChunk
from app.retrieval.embeddings import EmbeddingService


class VectorStore:
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.index = None
        self.chunks: List[DocumentChunk] = []

    def build(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build vector store with no chunks.")

        embeddings = self.embedding_service.embed_chunks(chunks)

        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        dimension = vectors.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(vectors)

        self.chunks = list(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Tuple[DocumentChunk, float]]:
        if self.index is None:
            raise RuntimeError(
                "Vector store has not been built yet."
            )

        if not query.strip():
            return []

        query_embedding = self.embedding_service.embed_texts(
            [query]
        )

        query_vector = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_vector,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            results.append(
                (
                    self.chunks[index],
                    float(score),
                )
            )

        return results