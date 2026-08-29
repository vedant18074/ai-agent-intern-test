from pathlib import Path

from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


class RAGService:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = Path(knowledge_base_path)

        # Load documents
        loader = KnowledgeBaseLoader(str(self.knowledge_base_path))
        documents = loader.load_documents()

        # Split documents into chunks
        chunker = DocumentChunker()
        chunks = []

        for document in documents:
            chunks.extend(chunker.chunk_document(document))

        # Create embeddings and vector index
        embedding_service = EmbeddingService()

        vector_store = VectorStore(
            embedding_service=embedding_service
        )

        vector_store.build(chunks)

        # Retrieval layer
        self.retriever = RetrievalService(
            vector_store=vector_store
        )

        # Evidence safety layer
        self.evidence_analyzer = EvidenceAnalyzer()

    def retrieve(self, query: str, top_k: int = 5):
        retrieved = self.retriever.retrieve(
            query=query,
            top_k=top_k,
        )

        evidence = self.evidence_analyzer.analyze(
            retrieved
        )

        return evidence