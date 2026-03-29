# RAGBot

RAGBot is a lightweight Retrieval-Augmented Generation API built with FastAPI, LangChain, FAISS, and SQLite. It lets you upload `.txt` or `.pdf` documents, stores document metadata locally, retrieves relevant chunks for each question, and generates grounded answers with either OpenAI or Ollama.

This project is useful when you want a simple local RAG stack that demonstrates the full backend flow: ingestion, chunking, embedding, retrieval, answer generation, persistence, testing, and containerized deployment.

## Features

- Upload plain-text and PDF documents with `POST /documents`
- Persist chunk embeddings in a local FAISS index
- Persist document metadata in SQLite
- Ask grounded questions with `POST /chat`
- Return source chunks with every answer
- Swap between OpenAI and Ollama via environment variables

## Project Structure

```text
ragbot/
  api/
    main.py
    routes/
      chat.py
      documents.py
      health.py
  core/
    indexer.py
    llm.py
    retriever.py
    service.py
  db/
    models.py
    session.py
tests/
  test_indexer.py
  test_llm.py
  test_retriever.py
  test_routes.py
Dockerfile
docker-compose.yml
requirements.txt
README.md
.env.example
```

## Architecture

```text
          +-------------------+
          |   Client / curl   |
          +---------+---------+
                    |
                    v
          +-------------------+
          | FastAPI Endpoints |
          +---------+---------+
                    |
        +-----------+------------+
        |                        |
        v                        v
+---------------+      +------------------+
| SQLite        |      | RAG Service      |
| document meta |      | orchestration    |
+---------------+      +---------+--------+
                                  |
                    +-------------+--------------+
                    |                            |
                    v                            v
           +------------------+        +------------------+
           | Text splitter    |        | LLM client       |
           | + FAISS index    |        | OpenAI / Ollama  |
           +--------+---------+        +------------------+
                    |
                    v
           +------------------+
           | Retrieved chunks |
           +------------------+
```

## Local Setup

1. Create a virtual environment.
2. Install dependencies.
3. Copy `.env.example` to `.env`.
4. Set either `OPENAI_API_KEY` or configure Ollama locally.
5. Run the API.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn ragbot.api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Docker

Build and run the app with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

The service starts on `http://localhost:8000`.
Compose persists SQLite and FAISS data under `./data/`.

## Environment Configuration

Important variables:

- `DATABASE_URL`: SQLite location
- `FAISS_INDEX_PATH`: directory used for persisted FAISS files
- `LLM_PROVIDER`: `auto`, `openai`, or `ollama`
- `OPENAI_API_KEY`: required for OpenAI mode
- `OLLAMA_BASE_URL`: Ollama server URL
- `DEFAULT_TOP_K`: default retrieval depth

## Example API Usage

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Upload a document:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "file=@./sample.txt"
```

List indexed documents:

```bash
curl http://127.0.0.1:8000/documents
```

Ask a question:

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main ideas in the uploaded document?",
    "top_k": 4
  }'
```

## Swapping LLM Providers

Use OpenAI:

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

Use Ollama:

```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3.1
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

With `LLM_PROVIDER=auto`, the app selects OpenAI when `OPENAI_API_KEY` is set and otherwise falls back to Ollama.

## Testing

Run the full test suite with coverage:

```bash
pytest
```

The tests mock embeddings and answer generation so they do not require network access or a live model server.
