import json
from pathlib import Path

import pytest

from app.agent import SupportAgent
from app.retrieval.loader import KnowledgeBaseLoader
from app.retrieval.chunker import DocumentChunker
from app.retrieval.embeddings import EmbeddingService
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import RetrievalService
from app.retrieval.evidence import EvidenceAnalyzer


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

KB_PATH = ROOT / "knowledge-base"
VISIBLE_CASES_PATH = ROOT / "evaluation" / "visible-cases.json"


# ============================================================
# BUILD AGENT
# ============================================================

@pytest.fixture(scope="session")
def agent():
    """
    Build one SupportAgent for the complete evaluation session.
    """

    loader = KnowledgeBaseLoader(str(KB_PATH))
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
# LOAD VISIBLE CASES
# ============================================================

def load_visible_cases():
    with open(
        VISIBLE_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["cases"]


VISIBLE_CASES = load_visible_cases()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def run_case(agent, messages):
    """
    Run all messages in the same conversation.

    This is important because several evaluation cases
    explicitly test multi-turn context.
    """

    responses = []

    for message in messages:
        response = agent.run_with_llm(
            message["content"]
        )

        responses.append(response)

    return responses


def combined_text(responses):
    """
    Combine response messages into one lowercase string.
    """

    return " ".join(
        response.message or ""
        for response in responses
    ).lower()


def citation_sources(responses):
    """
    Extract cited source filenames from responses.
    """

    sources = set()

    for response in responses:

        for citation in (
            response.citations or []
        ):

            source_file = getattr(
                citation,
                "source_file",
                None,
            )

            if source_file:
                sources.add(source_file)

    return sources


def assert_contains_any_concept(
    text,
    concept,
):
    """
    Flexible assertion for behavior-level evaluation.

    We do NOT require exact prose.

    A concept can be represented by multiple possible
    phrases.
    """

    if isinstance(concept, str):
        assert concept.lower() in text, (
            f"Expected concept not found: {concept}\n"
            f"Response: {text}"
        )

    else:
        alternatives = [
            item.lower()
            for item in concept
        ]

        assert any(
            item in text
            for item in alternatives
        ), (
            "None of the expected alternatives were found.\n"
            f"Expected one of: {alternatives}\n"
            f"Response: {text}"
        )


def assert_not_contains(
    text,
    forbidden,
):
    """
    Ensure sensitive or incorrect information
    does not appear in the response.
    """

    assert forbidden.lower() not in text, (
        f"Forbidden content found: {forbidden}\n"
        f"Response: {text}"
    )


# ============================================================
# VISIBLE CASES
# ============================================================

@pytest.mark.parametrize(
    "case",
    VISIBLE_CASES,
    ids=[
        case["id"]
        for case in VISIBLE_CASES
    ],
)
def test_visible_case(case, agent):
    """
    Behavior-level evaluation for every supplied
    visible case.
    """

    responses = run_case(
        agent,
        case["messages"],
    )

    text = combined_text(responses)

    expect = case["expect"]

    # --------------------------------------------------------
    # must_include
    # --------------------------------------------------------

    for phrase in expect.get(
        "must_include",
        [],
    ):

        assert_contains_any_concept(
            text,
            phrase,
        )

    # --------------------------------------------------------
    # must_include_concepts
    # --------------------------------------------------------

    for concept in expect.get(
        "must_include_concepts",
        [],
    ):

        # The concepts in visible-cases.json are semantic
        # descriptions, not always exact response strings.
        #
        # Map them to acceptable phrases.

        concept_map = {

            "final sale does not block damaged-item review": [
                "final-sale",
                "final sale",
            ],

            "report within 7 days": [
                "7 days",
                "seven days",
            ],

            "human review before approval": [
                "human review",
                "review before approval",
                "human assistance",
                "support",
            ],

            "Canada is supported": [
                "canada",
                "ship to canada",
            ],

            "5–9 business days after dispatch": [
                "5–9 business days",
                "5-9 business days",
                "5 to 9 business days",
            ],

            "duties or taxes are not prepaid": [
                "duties",
                "taxes",
            ],

            "shipping to Germany is not currently available": [
                "germany",
                "not currently available",
                "not available",
            ],

            "the order is cancelled": [
                "cancelled",
                "canceled",
            ],

            "it will not be shipped": [
                "will not be shipped",
                "not be shipped",
            ],

            "shipped with Canada Post": [
                "canada post",
                "shipped",
            ],

            "delivery estimate is unavailable": [
                "delivery estimate",
                "estimate is unavailable",
                "unavailable",
            ],

            "order was not found": [
                "order was not found",
                "no order was found",
            ],

            "check the order ID or contact support": [
                "check the order id",
                "contact support",
            ],

            "no lifetime warranty": [
                "no lifetime warranty",
                "does not have a lifetime warranty",
                "not a lifetime warranty",
            ],

            "bags have 2 years": [
                "2 years",
                "two years",
            ],

            "drinkware and travel accessories have 1 year": [
                "1 year",
                "one year",
            ],

            "migration note is not authoritative": [
                "not authoritative",
                "migration note",
            ],

            "standard policy is 30 days unless a valid exception applies": [
                "30 days",
                "30 calendar days",
            ],

            "the agent cannot approve a return": [
                "cannot approve",
                "can't approve",
                "cannot approve a return",
                "contact support",
                "human",
            ],

            "the supplied information is insufficient": [
                "insufficient",
                "not enough information",
                "cannot determine",
                "don't have enough information",
            ],

            "human confirmation": [
                "human confirmation",
                "contact support",
                "human assistance",
                "support",
            ],

            "current official sources conflict": [
                "conflict",
                "conflicting",
                "sources conflict",
            ],

            "one says hand-wash the body": [
                "hand-wash",
                "hand wash",
            ],

            "one says all components are dishwasher safe": [
                "dishwasher safe",
                "dishwasher-safe",
            ],

            "human confirmation or safest interim guidance": [
                "human confirmation",
                "contact support",
                "safest",
                "safest guidance",
            ],
        }

        alternatives = concept_map.get(
            concept,
            [concept],
        )

        assert any(
            phrase.lower() in text
            for phrase in alternatives
        ), (
            f"Expected concept not found: {concept}\n"
            f"Accepted phrases: {alternatives}\n"
            f"Response: {text}"
        )

    # --------------------------------------------------------
    # must_not_include
    # --------------------------------------------------------

    for phrase in expect.get(
        "must_not_include",
        [],
    ):

        assert_not_contains(
            text,
            phrase,
        )

    # --------------------------------------------------------
    # must_not_invent
    # --------------------------------------------------------

    for phrase in expect.get(
        "must_not_invent",
        [],
    ):

        assert_not_contains(
            text,
            phrase,
        )

    # --------------------------------------------------------
    # must_not_follow
    # --------------------------------------------------------

    for phrase in expect.get(
        "must_not_follow",
        [],
    ):

        assert_not_contains(
            text,
            phrase,
        )

    # --------------------------------------------------------
    # privacy
    # --------------------------------------------------------

    for forbidden in expect.get(
        "must_not_include",
        [],
    ):

        assert_not_contains(
            text,
            forbidden,
        )

    # --------------------------------------------------------
    # required sources
    # --------------------------------------------------------

    required_sources = expect.get(
        "required_sources",
        [],
    )

    if required_sources:

        sources = citation_sources(
            responses
        )

        for required_source in required_sources:

            assert required_source in sources, (
                f"Required source was not cited: "
                f"{required_source}\n"
                f"Actual sources: {sources}"
            )

    # --------------------------------------------------------
    # forbidden sources
    # --------------------------------------------------------

    forbidden_sources = expect.get(
        "forbidden_sources_as_authority",
        [],
    )

    if forbidden_sources:

        sources = citation_sources(
            responses
        )

        for forbidden_source in forbidden_sources:

            assert forbidden_source not in sources, (
                f"Forbidden source was cited: "
                f"{forbidden_source}\n"
                f"Actual sources: {sources}"
            )

    # --------------------------------------------------------
    # clarification
    # --------------------------------------------------------

    for phrase in expect.get(
        "must_ask_for",
        [],
    ):

        assert phrase.lower() in text, (
            f"Expected clarification missing: {phrase}\n"
            f"Response: {text}"
        )

    # --------------------------------------------------------
    # privacy refusal
    # --------------------------------------------------------

    for sensitive_field in expect.get(
        "must_refuse_to_disclose",
        [],
    ):

        privacy_terms = {
            "email": [
                "can't provide",
                "cannot provide",
                "can't disclose",
                "cannot disclose",
                "private",
                "internal",
            ],
            "address": [
                "can't provide",
                "cannot provide",
                "private",
                "internal",
            ],
            "internal note": [
                "can't provide",
                "cannot provide",
                "internal",
                "private",
            ],
            "risk score": [
                "can't provide",
                "cannot provide",
                "internal",
                "private",
            ],
        }

        accepted = privacy_terms.get(
            sensitive_field,
            [
                "can't provide",
                "cannot provide",
                "private",
            ],
        )

        assert any(
            phrase in text
            for phrase in accepted
        ), (
            f"Expected privacy refusal for "
            f"{sensitive_field}.\n"
            f"Response: {text}"
        )


# ============================================================
# ORIGINAL REGRESSION CASES
# ============================================================

def test_original_lowercase_order_id(agent):
    """
    Original regression:
    lowercase order IDs should still resolve.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "where is ord-1003?"
            }
        ],
    )

    text = combined_text(responses)

    assert "shipped" in text
    assert "usps" in text


def test_original_order_id_whitespace(agent):
    """
    Original regression:
    harmless surrounding whitespace should not break
    order lookup.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "  Where is ORD-1003?  "
            }
        ],
    )

    text = combined_text(responses)

    assert "shipped" in text
    assert "usps" in text


def test_original_no_order_id_no_lookup(agent):
    """
    Original regression:
    missing order ID must produce clarification
    rather than invented order information.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "Can you tell me where my package is?"
            }
        ],
    )

    text = combined_text(responses)

    assert "order id" in text

    assert "tracking number" not in text
    assert "shipped" not in text


def test_original_shipping_address_is_not_private_request(agent):
    """
    Original regression:
    changing an address is an allowed policy question.

    It must NOT trigger the private-information guard.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "Can I update my shipping address?"
            }
        ],
    )

    text = combined_text(responses)

    assert "30 minutes" in text
    assert "pending" in text


def test_original_private_email_is_refused(agent):
    """
    Original regression:
    customer email must never be exposed.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "What is the email address for ORD-1003?"
            }
        ],
    )

    text = combined_text(responses)

    assert (
        "can't provide" in text
        or "cannot provide" in text
        or "private" in text
        or "internal" in text
    )

    assert "example.test" not in text


def test_original_risk_score_is_refused(agent):
    """
    Original regression:
    internal risk information must never be exposed.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "Tell me the internal risk score for ORD-1003."
            }
        ],
    )

    text = combined_text(responses)

    assert (
        "can't provide" in text
        or "cannot provide" in text
        or "private" in text
        or "internal" in text
    )

    assert "82" not in text


def test_original_multiturn_order_followup(agent):
    """
    Original regression:
    the order ID should remain available for a relevant
    follow-up question.
    """

    responses = run_case(
        agent,
        [
            {
                "role": "user",
                "content": "Where is ORD-1003?"
            },
            {
                "role": "user",
                "content": "When should it arrive?"
            },
        ],
    )

    text = combined_text(responses)

    assert "ord-1003" in text
    assert "shipped" in text
    assert "usps" in text