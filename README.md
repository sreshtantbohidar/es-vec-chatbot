# ES Vec Chatbot

Vector-based RAG chatbot over Elasticsearch. The migration script vectorizes documents from the
`fatboy_data` index (using the field definitions in `field_mapping.py`) into a new index
`vec_chat_fatboy_data`, and the Flask app (`app.py`) serves a chat UI that retrieves relevant
documents via k-NN search and answers with a local LLM (Ollama).

## Files

| File                 | Purpose                                                                 |
|----------------------|-------------------------------------------------------------------------|
| `vector_migration.py`| One-time migration: embeds `fatboy_data` docs into `vec_chat_fatboy_data` |
| `field_mapping.py`   | `TYPE_MAPPING` — analysis types with the fields/labels to vectorize      |
| `app.py`             | Flask chat app: k-NN retrieval + LLM answer generation                   |
| `.env`               | Connection details (ES host/creds, LLM endpoint, app host/port)          |

## Prerequisites

- Python 3.10–3.12 recommended (Python 3.14 may lack PyTorch/sentence-transformers wheels)
- Elasticsearch 8.x running and reachable
- Ollama (or any OpenAI-compatible endpoint) serving the chat model
- An embedding model (default `all-MiniLM-L6-v2`, 384 dims) — downloaded on first run

## Setup

```bash
pip install elasticsearch sentence-transformers flask requests
```

Copy `.env` values (or keep as-is) — key variables:

| Variable          | Default               | Description                                  |
|-------------------|-----------------------|----------------------------------------------|
| `ES_HOST`         | `http://localhost:9200` | Elasticsearch URL (may include `https://`)  |
| `ES_USER`         | —                     | ES username (basic auth, optional)           |
| `ES_PASS`         | —                     | ES password (basic auth, optional)           |
| `ES_VERIFY_CERTS` | `false`               | Set `true` to verify TLS certs               |
| `ES_INDEX`        | `vec_chat_fatboy_data`| Index the chat app searches                   |
| `LLM_BASE_URL`    | `http://localhost:11434` | LLM endpoint (OpenAI-compatible `/v1` or native Ollama) |
| `LLM_API_KEY`     | —                     | Bearer token for the LLM endpoint            |
| `LLM_MODEL`       | `llama3`              | Chat model name                              |
| `EMBED_MODEL`     | `all-MiniLM-L6-v2`    | Embedding model (must match migration dims)  |
| `EMBED_DIMS`      | `384`                 | Vector dimensions (must match the model)     |
| `APP_HOST`        | `127.0.0.1`           | Flask bind host                              |
| `APP_PORT`        | `5000`                | Flask bind port                              |

## Run the migration

```bash
python vector_migration.py
```

What it does:

1. Connects to ES using `ES_HOST` / `ES_USER` / `ES_PASS` from `.env`
2. **Deletes and recreates** `vec_chat_fatboy_data` (⚠️ wipes the target index if it exists)
3. Scans every document in `fatboy_data`
4. Builds one text block per doc from all fields defined in `field_mapping.py`, formatted as
   `Label: value` pairs (e.g. `Location Name: X, Description: Y`)
5. Embeds that text with the embedding model and writes each doc (original fields + the vector +
   a readable `combined_text`) to `vec_chat_fatboy_data`

## Run the chat app

```bash
python app.py
```

Then open `http://<APP_HOST>:<APP_PORT>` (e.g. `http://0.0.0.0:8000`). The chat flow:

1. The user message is embedded with the same model used for ingestion
2. A k-NN search (`combined_text_vector`, cosine) returns the top 3 documents
3. Retrieved context + session memory are assembled into a prompt and sent to the LLM
4. The answer is returned; conversation memory is auto-compressed into a rolling summary

## How the field mapping works

`field_mapping.py` groups analysis types (infra, training, force, sitrep, etc.) with the fields
relevant to each. `vector_migration.py` takes the **union of all fields across every type**
(deduplicated, with their human-readable labels) and uses that to build the vectorization text —
so every analysis type's fields contribute to retrieval context.
