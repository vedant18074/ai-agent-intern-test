<h1 align="center">Aster & Row — Reliable RAG Support Agent</h1>

<h2>🎥 Project Demo</h2>

<video controls width="800">
  <source src="demo/aster-row-demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

<p>
  <strong>Demo covers:</strong>
</p>

<ul>
  <li>Knowledge-base question with citations</li>
  <li>Order lookup</li>
  <li>Multi-turn conversation</li>
  <li>Safe refusal / human escalation</li>
  <li>52-test evaluation suite</li>
</ul>

<hr>


# Aster & Row — Reliable RAG Support Agent

Aster & Row is a fictional ecommerce company selling bags, drinkware, and travel accessories. This project implements a small, reliability-focused customer support agent that answers knowledge-base questions, performs safe order lookups, handles multi-turn questions, detects conflicting authoritative information, protects private order data, and abstains when the available evidence is insufficient.

The implementation prioritizes **groundedness and deterministic safety controls over broad agent autonomy**.

---

## Features

- Knowledge-base question answering using Retrieval-Augmented Generation (RAG)
- Active/current policy precedence over superseded or legacy content
- Official-source precedence when multiple sources are available
- Deterministic order lookup using `data/orders.json`
- Protection of private order information such as:
  - customer email
  - shipping address
  - internal notes
  - risk scores
- Multi-turn conversation support
- Prompt-injection resistance for retrieved documents
- Deterministic detection of known current-official-source conflicts
- Safe abstention when the knowledge base does not contain sufficient information
- Customer-facing citations generated from retrieved evidence
- Secret-safe structured observability/logging
- Streamlit interface for demonstration
- Automated evaluation and regression tests

---

# Architecture

The application separates deterministic application logic from the LLM generation layer.

```text
                         ┌─────────────────────┐
                         │      Streamlit      │
                         │    Customer Query   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    SupportAgent     │
                         │ Routing + Safety     │
                         └───────┬─────┬────────┘
                                 │     │
                  Order question │     │ Knowledge question
                                 │     │
                                 ▼     ▼
                       ┌────────────┐  ┌─────────────────┐
                       │   Order    │  │ RetrievalService│
                       │   Lookup   │  └────────┬────────┘
                       └─────┬──────┘           │
                             │                  ▼
                             │        ┌─────────────────┐
                             │        │   Vector Store  │
                             │        │  + Embeddings    │
                             │        └────────┬────────┘
                             │                 │
                             │                 ▼
                             │        ┌─────────────────┐
                             │        │ EvidenceAnalyzer│
                             │        │ Safety + Conflict│
                             │        └────────┬────────┘
                             │                 │
                             │                 ▼
                             │        ┌─────────────────┐
                             │        │ Deterministic   │
                             │        │ Answers / RAG   │
                             │        └────────┬────────┘
                             │                 │
                             │                 ▼
                             │        ┌─────────────────┐
                             │        │ Local LLM       │
                             │        │ Ollama / Qwen   │
                             │        └────────┬────────┘
                             │                 │
                             └────────┬────────┘
                                      ▼
                              ┌─────────────────┐
                              │  AgentResponse  │
                              │  + Citations    │
                              └─────────────────┘
Design principle

The LLM is used primarily to verbalize already-approved evidence.

It is not trusted to independently decide:

whether evidence is authoritative
whether private order information can be disclosed
whether an order should be looked up
whether a known source conflict exists
whether an unsupported claim should be invented

These decisions are handled by application logic.

Technology      Stack
Component	    Choice
Language	    Python 3.10+
UI	            Streamlit
LLM	            Qwen3 1.7B through Ollama
LLM serving	    Ollama local API
Embeddings	    all-MiniLM-L6-v2
Embedding library	Sentence Transformers
Retrieval	        In-memory vector search
Knowledge storage	Markdown files
Order storage	    JSON
Testing	            pytest
Configuration	    python-dotenv
HTTP client	        requests
Embedding approach

Knowledge-base chunks are embedded using:

sentence-transformers/all-MiniLM-L6-v2

The embeddings are normalized and used for similarity-based retrieval.

Storage approach

The project deliberately uses an in-memory vector store rather than introducing a production vector database.

This keeps the implementation small and appropriate for the assignment timebox.

Order information remains in the local mock dataset and is accessed through the dedicated order lookup service.

Project Structure
.
├── README.md
├── knowledge-base/
│   ├── 01-returns-policy-current.md
│   ├── 02-returns-policy-legacy.md
│   ├── 03-final-sale-and-promotions.md
│   ├── 04-damaged-or-wrong-items.md
│   ├── 05-domestic-shipping.md
│   ├── 06-international-shipping.md
│   ├── 07-warranty.md
│   ├── 08-order-changes-and-cancellations.md
│   ├── 09-trailplus-membership.md
│   ├── 10-gift-cards-and-price-adjustments.md
│   ├── 11-product-care.md
│   ├── 12-breeze-tumbler-product-card.md
│   ├── 13-support-escalation.md
│   └── 14-internal-content-migration-notes.md
├── data/
│   ├── orders.json
│   └── orders-data-dictionary.md
├── evaluation/
│   └── visible-cases.json
├── app/
│   ├── agent.py
│   ├── config.py
│   ├── llm.py
│   ├── models.py
│   ├── observability.py
│   ├── tools/
│   │   └── order_lookup.py
│   └── retrieval/
│       ├── citations.py
│       ├── chunker.py
│       ├── embeddings.py
│       ├── evidence.py
│       ├── loader.py
│       ├── retriever.py
│       └── vector_store.py
├── tests/
│   ├── test_evaluation.py
│   ├── test_order_lookup.py
│   └── test_setup.py
├── .env.example
├── .gitignore
├── pytest.ini
├── requirements.txt
└── streamlit_app.py
Setup and Run
Prerequisites
Python 3.10 or later
Git
Ollama
Qwen3 1.7B installed through Ollama

Install the model:

ollama pull qwen3:1.7b
Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd ai-agent-intern-test
Create a virtual environment
Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
Install dependencies
pip install -r requirements.txt
Environment variables

Create a local .env file if required by the application.

Use .env.example as the template.

OPENAI_API_KEY=

Do not commit real credentials to the repository.

The LLM generation path used by the support agent runs locally through Ollama.

Running the Streamlit Application

Make sure Ollama is running and the required model is available:

ollama list

Then start Streamlit:

streamlit run streamlit_app.py

Open the local Streamlit URL shown in the terminal.

Example Interactions
1. Knowledge-base question
Customer

How long does a regular customer have to return an unused backpack?

The agent should provide the current return-window information and cite:

01-returns-policy-current.md

The current policy gives a 30 calendar day return window for the applicable regular-customer case from delivery.

2. Order lookup
Customer

Where is ORD-1007 and when should it arrive?

The order lookup service retrieves the order and returns customer-safe information such as:

status
carrier
tracking information when available
estimated delivery

It does not expose:

customer email
shipping address
risk score
internal notes
3. Multi-turn conversation
Customer

Do you ship internationally?

Follow-up:

What about Canada, and how long does it take?

The second question is handled as part of the same conversation and uses the relevant international-shipping evidence.

4. Safe abstention
Customer

Are all fabrics and adhesives in your bags vegan?

If the knowledge base does not establish this claim, the agent does not invent a certification or guarantee.

Instead, it indicates that the supplied information is insufficient and recommends human confirmation.

5. Conflicting authoritative sources
Customer

Can I put the entire Breeze Tumbler in the dishwasher?

The application detects the contradiction between the current official product-care sources.

Rather than silently selecting one source, it surfaces the conflict and recommends human confirmation or the safest interim guidance.

Reliability and Safety Design
Document precedence

Retrieved chunks contain metadata including:

status
policy authority
audience

Retrieved evidence is ranked using these properties in addition to semantic similarity.

The system prefers:

ACTIVE
  ↓
DRAFT
  ↓
SUPERSEDED

and:

OFFICIAL
  ↓
OTHER

Internal-only content is excluded from customer-facing evidence.

Prompt-injection protection

Retrieved documents are treated as data, not instructions.

For example, an internal migration note attempting to instruct the model to ignore the current return policy does not become an authoritative customer-facing policy.

The application retrieves the current policy and applies document-precedence rules before generating the response.

Order safety

Order-specific questions are routed to the order lookup service.

The LLM does not receive the complete order dataset.

The order lookup service exposes only customer-safe fields.

For cancelled or returned orders, stale carrier, tracking, and estimated-delivery information is removed before the customer-facing response is created.

Privacy-safe observability

Structured logging sanitizes common sensitive values such as:

email addresses
API keys
tokens
secrets
long numeric identifiers
sensitive internal numeric values

Example:

{"event": "test", "query": "Where is my order?", "email": "[REDACTED]", "risk_score": "[REDACTED]"}
Evaluation

Run the complete test suite with:

pytest -q
Final Result
52 passed

Final evaluation: 52/52 tests passing.

Visible Evaluation Coverage

The supplied visible evaluation cases cover:

Category	Cases	Result
Retrieval	2	2/2
Multi-source grounding	1	1/1
Conversation / multi-turn	1	1/1
Groundedness	2	2/2
Tool use	2	2/2
Tool reliability	3	3/3
Privacy	1	1/1
Prompt security	1	1/1
Abstention	1	1/1
Source conflict	1	1/1
Visible cases	15	15/15

The complete automated test suite contains 52 tests, including tests beyond the visible evaluation cases.

Baseline vs Final Evaluation

The initial evaluation run produced:

43 passed
9 failed

After implementing the reliability fixes:

52 passed
0 failed
Evaluation	Passed	Failed	Pass Rate
Baseline	43	        9	82.7%
Final	    52	        0	100%

The baseline failures were concentrated around:

current versus legacy policy grounding
multi-turn/international shipping behavior
unknown-order handling
warranty wording
prompt-injection handling
insufficient-information abstention
current official source conflicts

Bug Diary
Bug 1 — Legacy return policy could influence the answer
Observed failure

The standard return-window case failed because the expected current-policy concept, 30 calendar days, was missing.

Root cause

Semantic retrieval could surface legacy or superseded policy content alongside the current policy.

Retrieval therefore needed metadata-aware precedence rather than relying only on semantic similarity.

Fix

Added status and authority ranking to retrieved evidence.

The system now prefers:

active sources over superseded sources
official sources over non-official sources

Internal-only content is also excluded from customer-facing evidence.

Regression test

The evaluation suite verifies that the standard return answer uses the current returns policy rather than the legacy policy.

Bug 2 — Unknown order response lacked safe recovery guidance
Observed failure

The unknown-order case failed because the response did not clearly provide the expected next step.

Root cause

The lookup correctly determined that the order did not exist, but the customer-facing response was too narrow.

Fix

The order lookup response now clearly communicates that the order was not found and recommends checking the order ID or contacting support.

Regression test

Order lookup tests cover both valid and unknown order IDs.

Bug 3 — Retrieved migration note could distract from the current policy
Observed failure

The prompt-injection case failed because the response did not clearly establish that the migration note was not authoritative and that the standard policy remained 30 days.

Root cause

The migration-note query could retrieve the internal document strongly enough to dominate the evidence set.

Fix

Migration and prompt-injection-related queries retrieve a wider evidence set.

The evidence layer then applies the same safety and precedence rules.

Retrieved content is explicitly treated as data rather than executable instructions.

Regression test

The visible prompt-security case verifies that:

the 60-day instruction is not followed
the current policy remains authoritative
hidden instructions are not revealed
automatic approval is not provided
Bug 4 — Insufficient evidence did not consistently produce safe abstention
Observed failure

The vegan-materials case failed to provide the expected insufficient-information and human-confirmation behavior.

Root cause

The response path did not consistently distinguish between an answerable question and a question for which the knowledge base lacked sufficient evidence.

Fix

Added an explicit insufficient-evidence response path and human-confirmation guidance.

This prevents the model from filling knowledge gaps with unsupported general knowledge.

Regression test

The abstention evaluation case checks for insufficient-information behavior and prevents invented material certification or vegan guarantees.

Bug 5 — Current official product-care conflict needed explicit handling
Observed failure

The Breeze Tumbler case failed because the response did not expose both sides of the current official-source conflict.

Root cause

Retrieval could identify relevant documents, but the application needed deterministic contradiction detection before allowing a normal grounded answer.

Fix

EvidenceAnalyzer detects the known contradiction between:

hand-wash guidance
dishwasher-safe guidance

when both sources are active and official.

The agent then returns a conflict response instead of silently selecting one source.

Regression test

The source-conflict evaluation case verifies that both relevant sources are cited and that human confirmation or safest interim guidance is recommended.

Known Limitations

In-memory vector store
The vector index is rebuilt when the application starts.
Production would use persistent vector storage.
Small local model
Qwen3 1.7B is intentionally lightweight and locally runnable.
A larger production model may provide stronger language generation.
Limited deterministic conflict detection
The current conflict detector focuses on known contradiction patterns required by the assignment.
Production systems should use broader structured policy validation.
No production authentication
Authentication and user management were outside the assignment scope.
Mock order data
Orders are local test data rather than a live ecommerce/order-management integration.
Limited observability
Structured safe logging is implemented, but there is no production monitoring dashboard or distributed tracing.
No production deployment
The application is demonstrated locally through Streamlit and Ollama.
Improvements Before Production

Before deploying this system to real customers, I would prioritize:

Persistent vector storage with versioned indexes.
Stronger document and policy version governance.
Automated policy conflict detection.
Authentication and authorization for order access.
Integration with a real order-management system.
More extensive adversarial prompt-injection testing.
A larger continuously maintained evaluation dataset.
Production monitoring, tracing, and alerting.
Model quality, latency, and cost benchmarking.
Human escalation workflow integration.

AI Coding Tools Used

AI coding assistance was used during development primarily for:

debugging failing tests
identifying likely causes of retrieval and response failures
suggesting implementation approaches
improving code structure and documentation
reviewing edge cases and safety behavior

AI-generated suggestions were treated as proposals rather than trusted implementation decisions. All important changes were validated against the automated evaluation suite.

Example of an incorrect or incomplete AI suggestion

One early approach relied too heavily on the LLM to determine whether retrieved evidence was sufficient and how conflicting sources should be handled.

This was incomplete because the assignment requires reliable, deterministic behavior for:

document precedence
privacy
order lookup
source conflicts
safe abstention

The implementation was therefore changed so that these decisions are handled by application logic, while the LLM is primarily responsible for verbalizing approved evidence.