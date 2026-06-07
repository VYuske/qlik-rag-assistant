# Qlik RAG Assistant - Requirements

## Objective

Create a local AI assistant specialized in Qlik documentation.

The assistant must answer questions using only indexed Qlik documentation and operate completely offline.

---

## Functional Requirements

### FR-001

The assistant must answer questions about Qlik documentation.

### FR-002

The assistant must retrieve information from a vector database.

### FR-003

The assistant must cite:

* document source
* page
* section

### FR-004

The assistant must support questions written in Portuguese.

### FR-005

The assistant must search documentation in English.

### FR-006

The assistant must generate Qlik code examples only when documentation supports the answer.

### FR-007

The assistant must perform multiple retrievals if the first retrieval is insufficient.

---

## Technical Requirements

### TR-001

Use Python 3.12

### TR-002

Use LangGraph

### TR-003

Use Ollama

### TR-004

Use qwen3.5:4b as chat model

### TR-005

Use qwen3-embedding:0.6b embeddings

### TR-006

Use Chroma as vector database

### TR-007

Persist vector database locally

---

## Architecture

data/
storage/
src/
tests/

The implementation must follow clean architecture principles.

---

## Deliverables

1. Document ingestion
2. Vector database creation
3. Retriever
4. LangGraph agent
5. CLI interface
6. Streamlit interface
7. Unit tests
