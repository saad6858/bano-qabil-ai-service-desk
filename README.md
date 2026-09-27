# Bano Qabil AI Service Desk — Python Implementation (Gemini Free-Tier Safe)

Clean four-agent Python implementation for the Bano Qabil AI Service Desk:

```text
User
  ↓
Main Manager Agent
  ├── Website Knowledge Agent → website Pinecone index
  ├── Curriculum Agent       → curriculum Pinecone index
  └── Schedule Agent         → live Google Calendar API
```

Only four agents exist in this implementation:

1. Main Manager
2. Website Knowledge
3. Curriculum
4. Schedule

Student verification/OTP is intentionally outside this Python implementation because this version is focused on the four-agent course architecture.

## Why this version is different

This implementation follows the uploaded **Google Gemini Free Tier — Practical Guide for Bano Qabil Projects** as the Python-side design constraint, not merely as a model-name reference.

The guide's important rules are built into the code:

- Gemini chat model: `gemini-3.5-flash-lite`
- Backup chat model: `gemini-3.1-flash-lite`
- Embeddings: `models/gemini-embedding-001`
- Embedding dimension: 3072
- Batch document embeddings with `embed_documents()`
- Manual Pinecone upsert after batch embedding
- 5-second minimum delay before every RAG query embedding call
- 3-second minimum delay between Gemini chat calls, plus a rolling 15-RPM safety guard
- 60-second cooldown/retry behavior for 429 responses
- Local 15-RPM/500-RPD chat and 100-RPM/1000-RPD embedding guards matching the guide snapshot
- One shared Gemini LLM instance for all four agents
- No autonomous multi-agent conversation loops
- `max_iter=1` for every CrewAI agent
- Delegation disabled
- Retrieval is performed once before the specialist agent call so a RAG tool does not create an extra agent/tool/LLM loop
- Live schedule is queried from Google Calendar, never from RAG
- Public website and curriculum data remain in separate Pinecone indexes

The uploaded guide records the following class-tested quota snapshot:

| Model | RPM | TPM | RPD |
|---|---:|---:|---:|
| Gemini 3.5 Flash Lite | 15 | 250,000 | 500 |
| Gemini 3.1 Flash Lite | 15 | 250,000 | 500 |
| Gemini Embedding 1 | 100 | 30,000 | 1,000 |

Those figures are treated as the project's **observed guide snapshot**, not as a permanent Google guarantee. Google currently states that active limits vary by model/project/tier and should be checked in AI Studio before heavy usage.

## Requirements

- Python 3.13+
- Google AI Studio API key
- Pinecone API key
- Two Pinecone indexes using 3072-dimensional Gemini Embedding 001 vectors:
  - `bq-website-knowledge`
  - `bq-curriculum-knowledge`
- Google Calendar access for the schedule agent

The code never contains real secrets.

## Setup

```bash
cp .env.example .env
```

Fill in the values in `.env`.

Install:

```bash
uv sync
```

or:

```bash
pip install -e .
```

Run the smoke tests:

```bash
pytest -q
```

Run the service desk:

```bash
python -m bq_service_desk
```

Example:

```text
You: How many courses does Bano Qabil currently offer?
```

## Important Pinecone compatibility rule

The Python Gemini implementation expects 3072-dimensional vectors because `gemini-embedding-001` defaults to 3072 dimensions.

Do not point this implementation at the new n8n indexes that were created with OpenAI `text-embedding-3-small` (1536 dimensions).

Use the preserved Python/Gemini indexes, or create separate 3072-dimensional Python indexes.

The application checks index dimensions at startup and fails clearly instead of silently mixing embedding spaces.

## Reindexing local documents

The included ingestion script demonstrates the guide's safe pattern:

```text
documents
  ↓
RecursiveCharacterTextSplitter
  ↓
batch embed_documents()
  ↓
manual Pinecone upsert()
```

Example:

```bash
python scripts/index_documents.py \
  --source-dir data/curriculum \
  --index bq-curriculum-knowledge \
  --namespace curriculum
```

The same script can be used for website exports:

```bash
python scripts/index_documents.py \
  --source-dir data/website \
  --index bq-website-knowledge \
  --namespace website
```

Never replace batch embedding with:

```python
for doc in docs:
    vectorstore.add_documents([doc])
```

That pattern is explicitly avoided because the uploaded guide observed 429 errors from bursty per-document embedding calls.

## Rate-limit behavior

If a Gemini 429 occurs:

1. The rate limiter waits 60 seconds.
2. The same operation is retried.
3. Retries are bounded.
4. The application does not continuously hammer the API.

The guide observed that waiting allowed the bucket/counter to refill.

## Agent design

The manager makes one routing decision.

The selected specialist then answers once.

There is no recursive delegation:

```text
Manager call
   ↓
one route
   ↓
one retrieval/API operation
   ↓
one specialist call
   ↓
final answer
```

This is deliberate. The uploaded guide warns that multi-agent free-tier systems can create many chat and embedding calls in seconds.

## Environment variables

See `.env.example`.

Do not commit `.env`, API keys, service-account files, OAuth files, or private student data.

## What this implementation demonstrates

- CrewAI agents and tasks
- LangChain prompts and document splitting
- Gemini chat model
- Gemini embeddings
- Pinecone vector search
- RAG
- Multi-agent routing
- Live external API tool
- Free-tier rate-limit safety
- Separate public knowledge sources
- Semantic, instructor-readable comments
