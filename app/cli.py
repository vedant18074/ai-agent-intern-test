from app.agent import SupportAgent
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


def build_agent() -> SupportAgent:
    """Build the support agent from the local knowledge base."""

    loader = KnowledgeBaseLoader("knowledge-base")
    documents = loader.load_documents()

    chunker = DocumentChunker()
    chunks = []

    for document in documents:
        chunks.extend(
            chunker.chunk_document(document)
        )

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        embedding_service=embedding_service
    )

    vector_store.build(chunks)

    retrieval_service = RetrievalService(
        vector_store=vector_store
    )

    evidence_analyzer = EvidenceAnalyzer()

    return SupportAgent(
        retrieval_service=retrieval_service,
        evidence_analyzer=evidence_analyzer,
    )


def print_response(response) -> None:
    """Display an agent response in a simple CLI format."""

    print("\nAgent:")
    print(response.message)

    if response.citations:
        print("\nSources:")

        for citation in response.citations:
            print(f"- {citation.format()}")

    if getattr(response, "answer_type", None):
        print(f"\nAnswer type: {response.answer_type}")


def main() -> None:
    """Run the interactive support-agent CLI."""

    print("=" * 60)
    print("Aster & Row Support Agent")
    print("=" * 60)
    print("Type your question below.")
    print("Type 'exit' or 'quit' to stop.\n")

    agent = build_agent()

    while True:
        try:
            query = input("You: ").strip()

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        try:
            response = agent.run_with_llm(query)
            print_response(response)

        except Exception as exc:
            print(
                "\nAgent error: "
                f"{exc}"
            )


if __name__ == "__main__":
    main()