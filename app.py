import os
import requests
from flask import Flask, request, jsonify, render_template_string, session
from elasticsearch import Elasticsearch
from field_mapping import TYPE_MAPPING


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
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super_secret_infinite_memory_key")

# Configuration Constants (connection details from .env)
ES_URL = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"
ES_INDEX = os.getenv("ES_INDEX", "vec_chat_fatboy_data")
# LLM backend: primary OpenAI-compatible endpoint (LLM_BASE_URL), legacy Ollama vars as fallback
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("OLLAMA_URL") or "http://localhost:11434").rstrip("/")
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

def _embed_one(text):
    """Single Ollama embedding call."""
    url = LLM_BASE_URL + "/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(url, json=payload, timeout=120)
    if response.status_code == 200:
        return response.json()["embedding"]
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

        import numpy as np
        vectors = [_embed_one(chunk) for chunk in chunks]
        return np.mean(vectors, axis=0).tolist()
    except Exception as e:
        print(f"Ollama embedding exception: {e}")
    return [0.0] * 768


# Build list of all per-field vector column names from field_mapping
VECTOR_FIELD_NAMES = []
_seen = set()
for _mapping in TYPE_MAPPING.values():
    for _field in _mapping["fields"]:
        vec_name = "vec_" + _field
        if vec_name not in _seen:
            _seen.add(vec_name)
            VECTOR_FIELD_NAMES.append(vec_name)


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


def manage_memory():
    """Maintains a rolling summary and purges raw logs exceeding threshold."""
    if 'history' not in session:
        session['history'] = []
    if 'chat_summary' not in session:
        session['chat_summary'] = ""

    # Threshold: If raw turns exceed 4 messages (2 exchanges), compress the oldest 2
    if len(session['history']) > 4:
        turns_to_compress = session['history'][:2]
        # Keep the rest in active memory
        session['history'] = session['history'][2:]
        
        # Run background summarization block
        updated_summary = compress_history(session['chat_summary'], turns_to_compress)
        session['chat_summary'] = updated_summary


@app.route("/", methods=["GET"])
def index():
    """Render a lightweight UI layout for local testing."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/chat", methods=["POST"])
def chat():
    """Main RAG Chat Pipeline with Infinite Memory Handling."""
    user_message = request.json.get("message", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a valid message."}), 400

    # 1. Initialize and Manage Session Memory Context
    if 'history' not in session:
        session['history'] = []
    manage_memory()

    # 2. Vectorize user message for ES retrieval
    query_vector = get_embedding(user_message)

    # 3. Retrieve context via multi-field kNN search across all vec_* fields
    retrieved_context = "No relevant context found in index."
    try:
        # Build one kNN clause per vector field
        knn_queries = [
            {"field": vf, "query_vector": query_vector, "k": 3, "num_candidates": 50}
            for vf in VECTOR_FIELD_NAMES
        ]
        search_query = {
            "knn": knn_queries,
            "_source": True
        }
        res = es.search(index=ES_INDEX, body=search_query)
        hits = res['hits']['hits']

        if hits:
            retrieved_context = "\n---\n".join([
                str(hit['_source']) for hit in hits
            ])
    except Exception as e:
        print(f"Elasticsearch Query Exception: {e}")

    # 4. Construct Context-Aware Prompt using Local Memory Buffers
    recent_history_text = "\n".join([f"{t['role']}: {t['content']}" for t in session['history']])
    
    system_prompt = f"""
    You are an intelligent, helpful RAG chatbot. Answer the user's current query using the provided knowledge documents, historical summary, and recent chat history.

    [Long-term Summary of Past Conversation]:
    {session['chat_summary']}

    [Recent Interactive History]:
    {recent_history_text}

    [Verified Documents from Elasticsearch Index]:
    {retrieved_context}

    [Current Query]: {user_message}

    Refined Answer:"""

    # 5. Query Ollama and Update Live Session Logs
    bot_response = query_ollama(system_prompt)
    
    session['history'].append({"role": "user", "content": user_message})
    session['history'].append({"role": "assistant", "content": bot_response})
    session.modified = True  # Explicitly save changes to the session storage cookie

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
            <span>Local RAG Chatbot (ES + Ollama)</span>
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
    app.run(host=APP_HOST, port=APP_PORT, debug=True)
