import os
import time
import random
import threading
import numpy as np
import requests
from flask import Flask, request, jsonify, render_template_string, session
from elasticsearch import Elasticsearch
from field_mapping import TYPE_MAPPING

try:
    from flask_session import Session
except ImportError:
    Session = None  # fall back to cookie sessions if flask-session is missing


def load_env_file(path=".env"):
    """Load KEY=VALUE pairs from .env (used when python-dotenv is not installed)."""
    try:
        from dotenv import load_dotenv
        load_dotenv(path)
        return
    except ImportError:
        pass
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()

# Initialize Flask App
app = Flask(__name__)
_secret = os.getenv("FLASK_SECRET_KEY")
if not _secret:
    # Cookie/server sessions are forgeable without a real secret — refuse to guess.
    raise RuntimeError("FLASK_SECRET_KEY is not set. Add it to .env (e.g. `python -c \"import secrets; print(secrets.token_hex(32))\"`).")
app.secret_key = _secret

# Server-side sessions: Flask's default cookie session caps at ~4KB, which RAG
# responses blow through silently. Filesystem sessions have no such limit.
if Session is not None:
    app.config["SESSION_TYPE"] = "filesystem"
    app.config["SESSION_FILE_DIR"] = os.getenv("SESSION_FILE_DIR", os.path.join(os.path.dirname(__file__), ".flask_sessions"))
    os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
    Session(app)

# Configuration Constants (connection details from .env)
ES_URL = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"
ES_INDEX = os.getenv("ES_INDEX", "vec_chat_fatboy_data")
# LLM backend: primary OpenAI-compatible endpoint (LLM_BASE_URL), legacy Ollama vars as fallback
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
# Ollama native endpoint (strip /v1 suffix if present — embeddings use /api/embeddings)
OLLAMA_BASE_URL = LLM_BASE_URL.replace("/v1", "").replace("/chat/completions", "")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or "llama3"
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"
APP_HOST = os.getenv("APP_HOST", "127.0.0.1")
APP_PORT = int(os.getenv("APP_PORT", "5000"))

# Initialize Elasticsearch Client
try:
    es = Elasticsearch(
        ES_URL,
        basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
        verify_certs=ES_VERIFY_CERTS,
        request_timeout=120,
    )
    if not es.ping():
        print("Warning: Could not connect to Elasticsearch.")
except Exception as e:
    print(f"Elasticsearch Initialization Error: {e}")

def _embed_one(text, max_retries=3):
    """Single Ollama embedding call with retry on transient errors only."""
    url = OLLAMA_BASE_URL + "/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=120)
            if response.status_code == 200:
                return response.json()["embedding"]
            # 400/404 are deterministic (bad model/payload) — don't waste time retrying
            if response.status_code < 500:
                break
        except requests.RequestException as e:
            # Network errors are transient — retry
            if attempt >= max_retries - 1:
                raise RuntimeError(f"Ollama embedding network error: {e}")
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama embedding error {response.status_code}: {response.text}")


def get_embedding(text, chunk_size=7500):
    """Generate embedding via Ollama's /api/embeddings endpoint.
    
    User queries can be long, so we chunk and average to preserve quality.
    """
    try:
        if len(text) <= chunk_size:
            return _embed_one(text)

        overlap = 500
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + chunk_size])
            start += chunk_size - overlap

        vectors = [_embed_one(chunk) for chunk in chunks]
        return np.mean(vectors, axis=0).tolist()
    except Exception as e:
        print(f"Ollama embedding exception: {e}")
    # Small random fallback to avoid zero-magnitude cosine similarity error
    import random
    return [random.uniform(-0.01, 0.01) for _ in range(768)]


# Build list of all per-field vector column names from field_mapping
VECTOR_FIELD_NAMES = []
_seen = set()
for _mapping in TYPE_MAPPING.values():
    for _field in _mapping["fields"]:
        vec_name = "vec_" + _field
        if vec_name not in _seen:
            _seen.add(vec_name)
            VECTOR_FIELD_NAMES.append(vec_name)

# Collect available field labels for the system prompt
AVAILABLE_FIELDS = []
_seen_labels = set()
for _mapping in TYPE_MAPPING.values():
    for _field, _label in zip(_mapping["fields"], _mapping["field_labels"]):
        if _label not in _seen_labels:
            _seen_labels.add(_label)
            AVAILABLE_FIELDS.append(_label)

# Total doc count (refreshed on first query)
_total_docs = None

def _get_doc_count():
    global _total_docs
    if _total_docs is None:
        try:
            _total_docs = es.count(index=ES_INDEX)["count"]
        except Exception:
            _total_docs = "unknown"
    return _total_docs


def query_ollama(prompt):
    """Query the local LLM. Supports OpenAI-compatible endpoints (LLM_BASE_URL ending
    in /v1 or /chat/completions) and the native Ollama generate API as fallback."""
    try:
        headers = {"Content-Type": "application/json"}
        if LLM_API_KEY:
            headers["Authorization"] = f"Bearer {LLM_API_KEY}"

        if LLM_BASE_URL.endswith("/v1") or LLM_BASE_URL.endswith("/chat/completions"):
            url = LLM_BASE_URL + "/chat/completions" if not LLM_BASE_URL.endswith("/chat/completions") else LLM_BASE_URL
            payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": "You are an intelligent, helpful RAG chatbot."},
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            # Native Ollama generate API
            url = LLM_BASE_URL + "/api/generate"
            payload = {"model": LLM_MODEL, "prompt": prompt, "stream": False}
            response = requests.post(url, json=payload, headers=headers, timeout=LLM_TIMEOUT)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
    except Exception as e:
        print(f"LLM API Error: {e}")
    return "Error communicating with local LLM layer."


def compress_history(summary_so_far, old_turns):
    """Compress older conversation turns into a unified running summary."""
    turns_text = "\n".join([f"{t['role']}: {t['content']}" for t in old_turns])
    
    prompt = f"""
    You are a memory compaction system. Combine the current summary and the new chat turns into a single, highly condensed summary of the conversation's core facts and context. Avoid conversational filler.

    Current Summary: {summary_so_far}
    New turns to absorb:
    {turns_text}

    Updated Summary:"""
    
    return query_ollama(prompt)


# Pending chat summaries computed by background compression threads,
# keyed by a per-session id. Applied at the start of the next request.
_pending_summaries = {}
_summaries_lock = threading.Lock()


def _compress_in_background(session_id, summary_so_far, old_turns):
    try:
        updated = compress_history(summary_so_far, old_turns)
        if updated and not updated.startswith("Error"):
            with _summaries_lock:
                _pending_summaries[session_id] = updated
    except Exception as e:
        print(f"Background compression failed: {e}")


def manage_memory():
    """Maintains a rolling summary and purges raw logs exceeding threshold."""
    if 'history' not in session:
        session['history'] = []
    if 'chat_summary' not in session:
        session['chat_summary'] = ""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()

    # Apply any summary produced by an earlier background compression
    with _summaries_lock:
        pending = _pending_summaries.pop(session['session_id'], None)
    if pending:
        session['chat_summary'] = pending

    # Compress the oldest turns once raw history grows past 8 messages.
    # Runs in a background thread so it doesn't add latency to this response.
    if len(session['history']) > 8:
        turns_to_compress = session['history'][:4]
        session['history'] = session['history'][4:]
        threading.Thread(
            target=_compress_in_background,
            args=(session['session_id'], session['chat_summary'], turns_to_compress),
            daemon=True,
        ).start()


@app.route("/", methods=["GET"])
def index():
    """Render a lightweight UI layout for local testing."""
    return render_template_string(HTML_TEMPLATE)


# Keywords that signal an aggregation/listing question ("show me all locations",
# "how many records...") rather than a similarity question. These are better
# served by a broad fetch than kNN, which returns only wording-similar docs.
_AGG_KEYWORDS = (
    "how many", "count", "list all", "show all", "show me all", "all records",
    "all locations", "every record", "total number", "how much",
)

# Enumeration-intent phrases that, combined with a recognized field keyword
# (e.g. "infrastructure"), justify a FULL DATABASE terms aggregation.
_LIST_INTENT = (
    "types of", "type of", "kinds of", "kind of", "what ", "which ",
    "list", "enumerate", "categories of", "category of", "variety of",
)


def _is_aggregation_query(message):
    lowered = message.lower()
    return any(kw in lowered for kw in _AGG_KEYWORDS)


def _wants_enumeration(message):
    """True when the message asks to enumerate something (types, list, which...)."""
    lowered = message.lower()
    return any(kw in lowered for kw in _AGG_KEYWORDS) or any(kw in lowered for kw in _LIST_INTENT)


# Maps question keywords -> source fields whose distinct values can be
# aggregated across the FULL database via ES terms aggregations.
# Field names verified against live ES mappings (see also field_mapping.py).
_LISTABLE_FIELDS = [
    (("location", "place"), [("location_name", "Location Name"),
                             ("base_location_name", "Base Location Name"),
                             ("start_location_name", "Start Location Name"),
                             ("end_location_name", "End Location Name"),
                             ("general_area", "General Area"),
                             ("mil_dist_loc_name", "Military District Location")]),
    (("equipment", "weapon", "artillery", "howitzer", "mrl", "tank"), [
        ("equipment_name", "Equipment Name"),
        ("equipment_type", "Equipment Type")]),
    (("infra", "airfield", "storage", "camp"), [("infra_type", "Infra Type"),
                                                ("infra_name", "Infra Name"),
                                                ("infra_stage", "Infra Stage")]),
    (("radar", "elint", "sigint"), [("radar_type", "Radar Type"),
                                    ("radar_name", "Radar Name"),
                                    ("category", "Category")]),
    (("formation", "orbat", "corps", "brigade", "division"), [
        ("enemy_formation_name", "Enemy Formation Name"),
        ("orbat_title", "ORBAT Title"),
        ("formation_type", "Formation Type"),
        ("army_name", "Army Name"),
        ("div_name", "Division Name"),
        ("theatre_comd_name", "Theatre Command")]),
    (("pass", "transgression", "sighting"), [("pass_name", "Pass Name"),
                                             ("transgression_sighting_type", "Transgression/Sighting Type")]),
    (("training", "exercise"), [("training_name", "Training Name"),
                                ("training_type", "Training Type"),
                                ("training_force_type", "Training Force Type")]),
    (("person", "officer", "commander"), [("person_name", "Person Name"),
                                          ("designation", "Designation")]),
]

_AGG_MAX_VALUES = 300  # cap on distinct values fed to the LLM per field


def _detect_list_fields(message):
    """Return [(field_name, label)] the message asks to enumerate, or []."""
    lowered = message.lower()
    for keywords, fields in _LISTABLE_FIELDS:
        if any(kw in lowered for kw in keywords):
            return fields
    return []


def _terms_agg(field_name, size=_AGG_MAX_VALUES):
    """Distinct-value counts for one field across the WHOLE index."""
    # Dynamic text mappings expose a .keyword sub-field; try it first.
    for candidate in (f"{field_name}.keyword", field_name):
        body = {"size": 0, "aggs": {"vals": {"terms": {"field": candidate, "size": size}}}}
        res = es.search(index=ES_INDEX, body=body)
        buckets = res.get("aggregations", {}).get("vals", {}).get("buckets", [])
        if buckets:
            return buckets
        # No error means the field exists but was empty — try next candidate
    return []


def _build_full_listing(user_message):
    """Aggregate requested fields across the entire database.

    Returns (context_text, ok). Falls back to None when no field matched.
    """
    fields = _detect_list_fields(user_message)
    if not fields:
        return None
    blocks = []
    for field, label in fields:
        buckets = _terms_agg(field)
        if not buckets:
            continue
        lines = [f'{label} — {b["key"]} ({b["doc_count"]} records)' for b in buckets]
        total_distinct = len(buckets)
        suffix = "" if total_distinct < _AGG_MAX_VALUES else " (first 300 shown)"
        blocks.append(
            f"FULL DATABASE AGGREGATION — {label}: {total_distinct} distinct values{suffix}\n"
            + "\n".join(lines)
        )
    return "\n\n".join(blocks) if blocks else None


def _fetch_all_docs(max_docs=50):
    """Broad fetch for listing/counting questions: recent docs with all fields."""
    res = es.search(
        index=ES_INDEX,
        body={"query": {"match_all": {}}, "size": max_docs, "sort": ["_doc"]},
    )
    return res.get("hits", {}).get("hits", [])


def _retrieve_context(user_message, query_vector):
    """Return (retrieved_context_text, num_results).

    Aggregation-style questions get a broad match_all fetch; everything else
    goes through multi-field kNN.
    """
    try:
        # Full-database terms aggregation when the question asks to enumerate
        # a recognized field type (locations, equipment, radars, ...).
        full_listing = _build_full_listing(user_message)
        if full_listing and _wants_enumeration(user_message):
            return full_listing, 0  # num_results=0 signals aggregated listing
        if _is_aggregation_query(user_message):
            hits = _fetch_all_docs()
        else:
            knn_queries = [
                {"field": vf, "query_vector": query_vector, "k": 10, "num_candidates": 100}
                for vf in VECTOR_FIELD_NAMES
            ]
            res = es.search(index=ES_INDEX, body={"knn": knn_queries, "_source": True})
            hits = res.get("hits", {}).get("hits", [])

        # Deduplicate by stable doc _id
        seen_ids = set()
        unique_docs = []
        for hit in hits:
            doc_id = hit["_id"]
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_docs.append(hit)

        if not unique_docs:
            return "No relevant context found in index.", 0

        doc_blocks = []
        for i, hit in enumerate(unique_docs[:10], 1):
            src = hit["_source"]
            # Stable ID header so follow-ups can reference docs unambiguously
            header = f"Document {i} (id: {hit['_id']}):"
            combined = src.get("combined_text", "")
            if combined:
                doc_blocks.append(f"{header}\n{combined}")
            else:
                parts = []
                for k, v in src.items():
                    if not k.startswith("vec_") and k != "combined_text" and v:
                        parts.append(f"{k}: {v}")
                if parts:
                    doc_blocks.append(f"{header}\n" + "\n".join(parts))
        return "\n\n".join(doc_blocks), len(doc_blocks)
    except Exception as e:
        print(f"Elasticsearch Query Exception: {e}")
        return "No relevant context found in index.", 0


@app.route("/chat", methods=["POST"])
def chat():
    """Main RAG Chat Pipeline with Infinite Memory Handling."""
    user_message = (request.get_json(silent=True) or {}).get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid message."}), 400

    # 1. Initialize and Manage Session Memory Context
    if 'history' not in session:
        session['history'] = []
    manage_memory()

    # 2-3. Vectorize and retrieve context (kNN, or broad fetch for listing questions)
    query_vector = get_embedding(user_message)
    retrieved_context, num_results = _retrieve_context(user_message, query_vector)

    # 4. Construct Context-Aware Prompt using Local Memory Buffers
    recent_history_text = "\n".join([f"{t['role']}: {t['content']}" for t in session['history']])
    total_docs = _get_doc_count()

    system_prompt = f"""You are an intelligent, helpful RAG chatbot for a military intelligence database.

DATABASE INFO:
- Total records in database: {total_docs}
- Documents provided with this prompt: {num_results}
- Available fields: {', '.join(AVAILABLE_FIELDS[:15])}, and others
- Data includes: location names, coordinates, equipment details, activity dates, enemy formations, infrastructure info
- Note: Not all fields are populated in every record. Some records may only have a subset of fields.

INSTRUCTIONS:
- Answer ONLY based on the provided documents below. Do NOT make up information.
- If the documents don't contain enough information, say so honestly.
- When asked how many records the DATABASE has, use "Total records in database" ({total_docs}), NOT the number of documents provided. The provided documents are only a sample relevant to the question.
- When asked to list or show all of something (e.g. all locations), enumerate every distinct value across ALL provided documents, not just the first few.
- If a section labeled "FULL DATABASE AGGREGATION" is present, it contains exact distinct values and record counts computed over the ENTIRE database — present them completely (grouped/summarized if very long) and do not caveat that it may be incomplete.
- When referencing a document to the user, cite it as its stable id shown in parentheses, e.g. "document (id: abc123)".
- When listing data, use the actual values from the documents.
- Format your response clearly with bullet points or tables when presenting multiple records.
- If asked about a field that doesn't exist in the retrieved records, explain what fields ARE available.

[Chat History]:
{recent_history_text}

[Retrieved Documents]:
{retrieved_context}

[User Question]: {user_message}

Answer based on the documents above:"""

    # 5. Query Ollama and Update Live Session Logs
    bot_response = query_ollama(system_prompt)
    if bot_response == "Error communicating with local LLM layer.":
        # Don't persist failed responses into conversation memory
        return jsonify({"response": bot_response}), 503

    session['history'].append({"role": "user", "content": user_message})
    session['history'].append({"role": "assistant", "content": bot_response})
    session.modified = True  # Explicitly save changes to the session storage

    return jsonify({
        "response": bot_response,
        "debug_summary": session['chat_summary'],
        "debug_history_len": len(session['history'])
    })


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear memory registers to start a fresh chat session."""
    session.clear()
    return jsonify({"status": "Session wiped successfully."})


# Embedded Minimal HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Infinite Memory ES Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f4f6f9; margin: 0; padding: 20px; }
        .chat-container { max-width: 700px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); overflow: hidden; display: flex; flex-direction: column; height: 80vh; }
        .chat-header { background: #007bff; color: white; padding: 15px; font-weight: bold; display: flex; justify-content: space-between; align-items: center;}
        .chat-box { flex: 1; padding: 20px; overflow-y: auto; border-bottom: 1px solid #eee; }
        .message { margin-bottom: 15px; padding: 10px 15px; border-radius: 6px; max-width: 80%; }
        .user { background: #e1ffc7; align-self: flex-end; margin-left: auto; }
        .bot { background: #f1f0f0; align-self: flex-start; }
        .input-area { display: flex; padding: 15px; background: #fafafa; }
        input { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; margin-left: 10px; border-radius: 4px; cursor: pointer; }
        .clear-btn { background: #dc3545; font-size: 12px; padding: 5px 10px; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <span>Tectum RAG Chatbot (ES + Ollama)</span>
            <button class="clear-btn" onclick="clearChat()">Clear Memory</button>
        </div>
        <div class="chat-box" id="chatBox">
            <div class="message bot">Hello! Ask me anything regarding your indexed documents.</div>
        </div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Type your message here..." onkeypress="handleKey(event)">
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            if(!message) return;
            
            appendMessage(message, 'user');
            input.value = '';

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message })
                });
                const data = await res.json();
                appendMessage(data.response, 'bot');
            } catch(e) {
                appendMessage('Failed to get a response.', 'bot');
            }
        }
        function handleKey(e) { if(e.key === 'Enter') sendMessage(); }
        function appendMessage(text, sender) {
            const box = document.getElementById('chatBox');
            const div = document.createElement('div');
            div.className = `message ${sender}`;
            div.innerText = text;
            box.appendChild(div);
            box.scrollTop = box.scrollHeight;
        }
        async function clearChat() {
            await fetch('/clear', { method: 'POST' });
            document.getElementById('chatBox').innerHTML = '<div class="message bot">Memory cleared! How can I help you now?</div>';
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
