from app.agent import SupportAgent
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


# Load knowledge base
loader = KnowledgeBaseLoader("knowledge-base")
documents = loader.load_documents()

# Chunk documents
chunker = DocumentChunker()
chunks = []

for document in documents:
    chunks.extend(
        chunker.chunk_document(document)
    )

# Create embeddings
embedding_service = EmbeddingService()

# Build vector store
vector_store = VectorStore(
    embedding_service=embedding_service
)

vector_store.build(chunks)

# Create retrieval service
retrieval_service = RetrievalService(
    vector_store=vector_store
)

# Create evidence analyzer
evidence_analyzer = EvidenceAnalyzer()

# Create agent
agent = SupportAgent(
    retrieval_service=retrieval_service,
    evidence_analyzer=evidence_analyzer,
)


questions = [
    
    "How many days do I have to return an eligible item?",
    "What is the return shipping fee?",
    "Where is ORD-1003?",
    "What is the status of ORD-1002?",
    "When will ORD-1003 arrive?",
    "Where is my order?",
    "Where is ORD-9999?",

]


for question in questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    response = agent.run_with_llm(question)

    print("\nANSWER:")
    print(response.message)

    print("\nANSWER TYPE:")
    print(response.answer_type)

    print("\nCITATIONS:")
    for citation in response.citations:
        print(citation)