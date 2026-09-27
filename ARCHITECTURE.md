# Python Architecture

## Four agents

### 1. Main Manager
Routes the request and does not answer factual questions.

### 2. Website Knowledge Agent
Answers only from the public Bano Qabil website Pinecone index.

### 3. Curriculum Agent
Answers only from the curriculum Pinecone index.

### 4. Schedule Agent
Answers only from current Google Calendar data.

## Why retrieval happens before the specialist agent

The uploaded Gemini guide warns that a free-tier agent can generate multiple
chat requests when it thinks, calls a tool, observes the result, and thinks
again. The Python implementation avoids that pattern.

Instead:

```text
Manager LLM call
      ↓
route
      ↓
one RAG embedding + Pinecone query OR one Calendar API call
      ↓
Specialist LLM call
```

This keeps the number of Gemini requests predictable.

## Embedding strategy

Document ingestion:

```text
files
 ↓
RecursiveCharacterTextSplitter
 ↓
embed_documents(batch)
 ↓
manual Pinecone upsert
```

Query time:

```text
user question
 ↓
wait 5 seconds
 ↓
embed_query()
 ↓
Pinecone query
```

## Chat strategy

All four agents use one shared `SharedGeminiLLM`.

Every Gemini chat request is separated by at least 3 seconds.

Every agent has:

```text
max_iter=1
allow_delegation=False
```

No autonomous multi-agent loops are used.

## Vector dimensions

Gemini Embedding 001 defaults to 3072 dimensions. The Python implementation
therefore expects 3072-dimensional Pinecone indexes.

This is deliberately separate from the n8n indexes created later with
OpenAI text-embedding-3-small at 1536 dimensions.
