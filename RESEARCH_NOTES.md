# Research Notes — September 2026

## Project-source rules

The uploaded project guide is the primary implementation constraint for this
Python version.

It records the class-tested Gemini free-tier snapshot and practical failures:
bursting embeddings caused 429 errors, RAG questions need spacing, multi-agent
systems can multiply calls, and a one-minute cooldown was observed to recover
from 429s.

## Current official Google API status

Google's current model documentation still lists:
- Gemini 3.5 Flash-Lite as a stable model.
- Gemini 3.1 Flash-Lite as a stable model.
- gemini-embedding-001 as a stable text embedding model.

Google's current rate-limit documentation says limits are project/tier/model
dependent, applied per project rather than per API key, and RPD resets at
midnight Pacific Time. It also explicitly recommends checking active AI Studio
limits because limits can change.

Google's embedding documentation states that gemini-embedding-001 defaults to
3072 dimensions and supports smaller output dimensions. This project keeps the
guide's 3072-dimensional default rather than changing dimensions.

## Current Pinecone SDK direction

Pinecone's current Python SDK supports standard vector indexes with:
- `index.upsert(vectors=...)`
- `index.query(vector=..., top_k=..., include_metadata=True)`

The implementation uses those APIs because the Python/Gemini indexes are
standard pre-embedded vector indexes.

## CrewAI design

Current CrewAI exposes `Agent.max_iter`, `Agent.max_rpm`, and
`Agent.allow_delegation`, which are used here to cap agent iterations and
prevent delegation loops.

The implementation additionally performs retrieval outside the agent tool loop
because the project guide specifically warns about embedding/chat amplification
under free-tier multi-agent execution.
