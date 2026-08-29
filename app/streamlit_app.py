import sys
from pathlib import Path

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st

from app.agent import SupportAgent
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aster & Row Support",
    page_icon="🛍️",
    layout="centered",
)


# ============================================================
# BUILD AGENT
# ============================================================

@st.cache_resource
def build_agent() -> SupportAgent:
    """Build and cache the support agent."""

    loader = KnowledgeBaseLoader(
        str(PROJECT_ROOT / "knowledge-base")
    )

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


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.title("🛍️ Aster & Row Support")

st.caption(
    "Reliable AI support grounded in the Aster & Row "
    "knowledge base."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This support agent answers customer questions "
        "using the approved knowledge base and performs "
        "order lookups when an order ID is provided."
    )

    st.divider()

    st.subheader("Capabilities")

    st.write("✓ Knowledge-base answers")
    st.write("✓ Source citations")
    st.write("✓ Order lookup")
    st.write("✓ Multi-turn conversations")
    st.write("✓ Privacy protection")
    st.write("✓ Safe abstention")
    st.write("✓ Conflict detection")

    st.divider()

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# LOAD AGENT
# ============================================================

try:

    agent = build_agent()

except Exception as exc:

    st.error(
        "The support agent could not be initialized."
    )

    st.exception(exc)

    st.stop()


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        citations = message.get(
            "citations",
            []
        )

        if citations:

            with st.expander("Sources"):

                for citation in citations:

                    st.write(
                        citation.format()
                    )


# ============================================================
# CHAT INPUT
# ============================================================

query = st.chat_input(
    "Ask about returns, shipping, warranty, or an order..."
)


if query:

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):

        st.markdown(query)

    # --------------------------------------------------------
    # AGENT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Checking the knowledge base..."
        ):

            try:

                response = agent.run_with_llm(
                    query
                )

                answer = (
                    response.message
                    or
                    "I could not generate a reliable answer."
                )

                st.markdown(answer)

                # ------------------------------------------------
                # CITATIONS
                # ------------------------------------------------

                citations = (
                    response.citations
                    or []
                )

                if citations:

                    with st.expander(
                        "Sources"
                    ):

                        for citation in citations:

                            st.write(
                                citation.format()
                            )

                # ------------------------------------------------
                # HUMAN HANDOFF
                # ------------------------------------------------

                if getattr(
                    response,
                    "answer_type",
                    "",
                ) in {
                    "conflict",
                    "insufficient_evidence",
                    "order_not_found",
                }:

                    st.info(
                        "This case may require human support."
                    )

                # ------------------------------------------------
                # SAVE RESPONSE
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "citations": citations,
                    }
                )

            except Exception:

                error_message = (
                    "I couldn't process that request. "
                    "Please try again."
                )

                st.error(
                    error_message
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "citations": [],
                    }
                )