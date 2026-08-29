from pathlib import Path
from app.models import DocumentChunk
from app.retrieval.chunker import DocumentChunker
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService, RetrievedChunk
from app.retrieval.evidence import EvidenceAnalyzer
from app.retrieval.citations import CitationBuilder
from app.agent import SupportAgent
from app.llm import LLMService

def test_knowledge_base_loader():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))

    documents = loader.load_documents()

    assert len(documents) == 14

def test_current_returns_policy_metadata():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))

    documents = loader.load_documents()

    returns_policy = next(
        document
        for document in documents
        if document.source_file == "01-returns-policy-current.md"
    )

    assert returns_policy.status == "active"
    assert returns_policy.audience == "customer"
    assert returns_policy.policy_authority == "official"

def test_document_chunk_model():
    chunk = DocumentChunk(
        chunk_id="test-001",
        document_id="RET-2026-01",
        content="Standard returns are accepted within 30 days.",
        source_file="01-returns-policy-current.md",
        title="Returns Policy",
        heading="Standard Returns",
        status="active",
        audience="customer",
        policy_authority="official",
        effective_date="2026-04-01",
    )

    assert chunk.chunk_id == "test-001"
    assert chunk.document_id == "RET-2026-01"
    assert chunk.status == "active"
    assert chunk.audience == "customer"

def test_document_chunker():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    chunker = DocumentChunker(max_chunk_size=1200)

    chunks = chunker.chunk_document(documents[0])

    assert len(chunks) > 0

    for chunk in chunks:
        assert chunk.document_id == documents[0].document_id
        assert chunk.source_file == documents[0].source_file
        assert chunk.content
        assert chunk.status == documents[0].status

def test_chunk_preserves_document_metadata():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    document = next(
        document
        for document in documents
        if document.source_file == "01-returns-policy-current.md"
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(document)

    assert chunks

    first_chunk = chunks[0]

    assert first_chunk.document_id == document.document_id
    assert first_chunk.status == document.status
    assert first_chunk.audience == document.audience
    assert first_chunk.policy_authority == document.policy_authority

def test_embedding_service():
    embedding_service = EmbeddingService()

    embeddings = embedding_service.embed_texts(
        ["What is the return policy?"]
    )

    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0

def test_vector_store_build_and_search():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    chunker = DocumentChunker(max_chunk_size=1200)

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)

    vector_store.build(all_chunks)

    results = vector_store.search(
        "What is the return policy?",
        top_k=5,
    )

    assert len(results) > 0
    assert len(results) <= 5

    for chunk, score in results:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)
        assert chunk.content
        assert chunk.source_file

def test_vector_store_empty_query():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(str(knowledge_base_path))
    documents = loader.load_documents()

    chunker = DocumentChunker()

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    embedding_service = EmbeddingService()
    vector_store = VectorStore(embedding_service)

    vector_store.build(all_chunks)

    results = vector_store.search("   ")

    assert results == []

def test_retrieval_service():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    chunker = DocumentChunker()

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    embedding_service = EmbeddingService()

    vector_store = VectorStore(
        embedding_service
    )

    vector_store.build(all_chunks)

    retrieval_service = RetrievalService(
        vector_store
    )

    results = retrieval_service.retrieve(
        "What is the return policy?",
        top_k=5,
    )

    assert results

    for result in results:
        assert isinstance(
            result.chunk,
            DocumentChunk,
        )

        assert result.score >= 0.15
        assert result.status_rank >= 0
        assert result.authority_rank >= 0

def test_status_precedence():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    active_document = next(
        document
        for document in documents
        if document.status == "active"
    )

    superseded_document = next(
        document
        for document in documents
        if document.status == "superseded"
    )

    chunker = DocumentChunker()

    active_chunk = chunker.chunk_document(
        active_document
    )[0]

    superseded_chunk = chunker.chunk_document(
        superseded_document
    )[0]

    vector_store = VectorStore(
        EmbeddingService()
    )

    retrieval_service = RetrievalService(
        vector_store
    )

    active_rank = (
        retrieval_service._status_rank(
            active_chunk
        )
    )

    superseded_rank = (
        retrieval_service._status_rank(
            superseded_chunk
        )
    )

    assert active_rank > superseded_rank

def test_evidence_excludes_internal_content():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    internal_document = next(
        document
        for document in documents
        if document.source_file
        == "14-internal-content-migration-notes.md"
    )

    chunker = DocumentChunker()

    chunks = chunker.chunk_document(
        internal_document
    )

    retrieved = [
        RetrievedChunk(
            chunk=chunk,
            score=0.9,
            authority_rank=2,
            status_rank=3,
        )
        for chunk in chunks
    ]

    analyzer = EvidenceAnalyzer()

    evidence = analyzer.analyze(
        retrieved
    )

    assert evidence.chunks == []
    assert len(evidence.excluded_chunks) > 0

def test_evidence_detects_active_official_conflict():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    product_care = next(
        document
        for document in documents
        if document.source_file
        == "11-product-care.md"
    )

    product_card = next(
        document
        for document in documents
        if document.source_file
        == "12-breeze-tumbler-product-card.md"
    )

    chunker = DocumentChunker()

    care_chunks = chunker.chunk_document(
        product_care
    )

    card_chunks = chunker.chunk_document(
        product_card
    )

    retrieved = [
        RetrievedChunk(
            chunk=chunk,
            score=0.9,
            authority_rank=2,
            status_rank=3,
        )
        for chunk in (
            care_chunks + card_chunks
        )
        if (
            "Breeze Tumbler"
            in (chunk.heading or "")
            or "Cleaning"
            in (chunk.heading or "")
        )
    ]

    analyzer = EvidenceAnalyzer()

    evidence = analyzer.analyze(
        retrieved
    )

    assert evidence.has_conflict
    assert len(evidence.conflicts) >= 1
    assert evidence.requires_handoff

def test_superseded_source_does_not_create_active_conflict():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    active_document = next(
        document
        for document in documents
        if document.source_file
        == "01-returns-policy-current.md"
    )

    legacy_document = next(
        document
        for document in documents
        if document.source_file
        == "02-returns-policy-legacy.md"
    )

    chunker = DocumentChunker()

    active_chunk = chunker.chunk_document(
        active_document
    )[0]

    legacy_chunk = chunker.chunk_document(
        legacy_document
    )[0]

    retrieved = [
        RetrievedChunk(
            chunk=active_chunk,
            score=0.9,
            authority_rank=2,
            status_rank=3,
        ),
        RetrievedChunk(
            chunk=legacy_chunk,
            score=0.8,
            authority_rank=2,
            status_rank=1,
        ),
    ]

    analyzer = EvidenceAnalyzer()

    evidence = analyzer.analyze(
        retrieved
    )

    assert not evidence.has_conflict

def test_citation_builder():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    document = next(
        document
        for document in documents
        if document.source_file
        == "01-returns-policy-current.md"
    )

    chunker = DocumentChunker()

    chunk = chunker.chunk_document(document)[0]

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.85,
        authority_rank=2,
        status_rank=3,
    )

    citation = CitationBuilder.build(
        retrieved
    )

    assert (
        citation.source_file
        == "01-returns-policy-current.md"
    )

    assert citation.heading == chunk.heading
    assert citation.document_id == chunk.document_id
    assert citation.score == 0.85

def test_citation_format():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    document = next(
        document
        for document in documents
        if document.source_file
        == "01-returns-policy-current.md"
    )

    chunker = DocumentChunker()

    chunk = chunker.chunk_document(document)[0]

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.85,
        authority_rank=2,
        status_rank=3,
    )

    citation = CitationBuilder.build(
        retrieved
    )

    formatted = citation.format()

    assert "01-returns-policy-current.md" in formatted
    assert "Section:" in formatted

def test_agent_rejects_empty_query():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    chunker = DocumentChunker()

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    vector_store = VectorStore(
        EmbeddingService()
    )

    vector_store.build(all_chunks)

    retrieval_service = RetrievalService(
        vector_store
    )

    evidence_analyzer = EvidenceAnalyzer()

    agent = SupportAgent(
        retrieval_service,
        evidence_analyzer,
    )

    try:
        agent.run("")
        assert False, "Expected ValueError"
    except ValueError:
        pass

def test_agent_returns_grounded_evidence():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    chunker = DocumentChunker()

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    vector_store = VectorStore(
        EmbeddingService()
    )

    vector_store.build(all_chunks)

    retrieval_service = RetrievalService(
        vector_store
    )

    evidence_analyzer = EvidenceAnalyzer()

    agent = SupportAgent(
        retrieval_service,
        evidence_analyzer,
    )

    response = agent.run(
        "What is the return policy?"
    )

    assert response.answer_type == "grounded"
    assert response.is_grounded
    assert len(response.citations) > 0
    assert response.evidence.chunks

def test_agent_detects_conflicting_information():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    chunker = DocumentChunker()

    all_chunks = []

    for document in documents:
        all_chunks.extend(
            chunker.chunk_document(document)
        )

    vector_store = VectorStore(
        EmbeddingService()
    )

    vector_store.build(all_chunks)

    retrieval_service = RetrievalService(
        vector_store
    )

    evidence_analyzer = EvidenceAnalyzer()

    agent = SupportAgent(
        retrieval_service,
        evidence_analyzer,
    )

    response = agent.run(
        "Can I put the Breeze Tumbler in the dishwasher?"
    )

    assert response.answer_type == "conflict"
    assert response.requires_handoff
    assert response.evidence.has_conflict
    assert len(response.evidence.conflicts) >= 1

def test_llm_service_handles_missing_evidence():
    llm = LLMService()

    response = llm.generate(
        "What is the return policy?",
        [],
    )

    assert "enough information" in response.lower()

def test_llm_service_formats_evidence():
    project_root = Path(__file__).resolve().parents[1]
    knowledge_base_path = project_root / "knowledge-base"

    loader = KnowledgeBaseLoader(
        str(knowledge_base_path)
    )

    documents = loader.load_documents()

    document = next(
        document
        for document in documents
        if document.source_file
        == "01-returns-policy-current.md"
    )

    chunker = DocumentChunker()

    chunk = chunker.chunk_document(document)[0]

    retrieved = RetrievedChunk(
        chunk=chunk,
        score=0.9,
        authority_rank=2,
        status_rank=3,
    )

    llm = LLMService()

    evidence_text = llm._format_evidence(
        [retrieved]
    )

    assert (
        "01-returns-policy-current.md"
        in evidence_text
    )

    assert (
        chunk.heading
        in evidence_text
    )

    assert chunk.content in evidence_text

def test_llm_service_builds_grounded_prompt():
    llm = LLMService()

    prompt = llm._build_prompt(
        "What is the return policy?",
        "Source: 01-returns-policy-current.md",
    )

    assert (
        "What is the return policy?"
        in prompt
    )

    assert (
        "01-returns-policy-current.md"
        in prompt
    )

    assert (
        "ONLY"
        in prompt
    )