from typing import List

from sentence_transformers import SentenceTransformer

from app.models import DocumentChunk


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_texts(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: List[DocumentChunk],
    ) -> List[List[float]]:
        texts = [chunk.content for chunk in chunks]

        return self.embed_texts(texts)