from app.agent import SupportAgent
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


# ==========================================================
# BUILD KNOWLEDGE BASE
# ==========================================================

loader = KnowledgeBaseLoader("knowledge-base")
documents = loader.load_documents()

chunker = DocumentChunker()
chunks = []

for document in documents:
    chunks.extend(
        chunker.chunk_document(document)
    )


# ==========================================================
# BUILD VECTOR STORE
# ==========================================================

embedding_service = EmbeddingService()

vector_store = VectorStore(
    embedding_service=embedding_service
)

vector_store.build(chunks)


# ==========================================================
# BUILD SERVICES
# ==========================================================

retrieval_service = RetrievalService(
    vector_store=vector_store
)

evidence_analyzer = EvidenceAnalyzer()

agent = SupportAgent(
    retrieval_service=retrieval_service,
    evidence_analyzer=evidence_analyzer,
)


# ==========================================================
# EDGE CASE QUESTIONS
# ==========================================================

questions = [
    "Can I return a final-sale item?",
    "Can I return an item after 50 days?",
    "Can I change my shipping address?",
    "Where is ORD-1003?",
    "Where is ord-1003?",
    "What is the status of ORD-9999?",
    "Tell me the internal risk score for ORD-1003.",
    "What is the email address for ORD-1003?",
    "Where is my order?",
    "What is the return policy?",
]


# ==========================================================
# RUN TESTS
# ==========================================================

for index, question in enumerate(
    questions,
    start=1,
):

    print("\n" + "=" * 80)
    print(f"TEST {index}")
    print("QUESTION:")
    print(question)

    try:
        response = agent.run_with_llm(question)

        print("\nANSWER:")
        print(response.message)

        print("\nANSWER TYPE:")
        print(response.answer_type)

        print("\nCITATIONS:")

        if response.citations:
            for citation in response.citations:
                print(citation)
        else:
            print("None")

    except Exception as error:

        print("\nERROR:")
        print(type(error).__name__)
        print(error)