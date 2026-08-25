import os
import re
import time
import random
import threading
import numpy as np
from datetime import datetime, date, timedelta
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
def _env_int(name, default):
    """Read an int env var, stripping inline comments like '120  # note'."""
    raw = os.getenv(name, "")
    raw = raw.split("#", 1)[0].strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default

LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 30)
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


# Fields that can carry a location entity for exact-match filtering.
_LOCATION_FILTER_FIELDS = (
    "location_name", "base_location_name", "start_location_name",
    "end_location_name", "general_area", "mil_dist_loc_name",
)

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b")
_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july",
                "august", "september", "october", "november", "december")
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_JUNK_LOCATIONS = {"unknown", "location not known", "na", "none"}

# Filler words ignored when extracting keyword-search terms from a question.
_STOPWORD_TERMS = {
    "the", "and", "with", "mention", "give", "you", "have", "from",
    "about", "what", "which", "when", "where", "how", "latest",
    "information", "records", "data", "show", "list", "please",
}

# Relative date ranges: "current/this month", "last month", "last 6 months",
# "past 30 days", "this year", "last year". Resolved against today's date.
# NOTE: the generic "last/past N <unit>" patterns below must come AFTER the
# fixed phrases (e.g. "last month") so they don't shadow them.
_RELATIVE_RANGE_RES = (
    (re.compile(r"\b(?:current|this)\s+month\b"), "cur_month"),
    (re.compile(r"\blast\s+month\b|\bprevious\s+month\b"), "prev_month"),
    (re.compile(r"\b(?:current|this)\s+year\b"), "cur_year"),
    (re.compile(r"\blast\s+year\b|\bprevious\s+year\b"), "prev_year"),
    # Generic: last/past N months | years | days | weeks | hours
    (re.compile(r"\b(?:last|past|previous|prior)\s+(\d+)\s*months?\b"), "n_months"),
    (re.compile(r"\b(?:last|past|previous|prior)\s+(\d+)\s*years?\b"), "n_years"),
    (re.compile(r"\b(?:last|past|previous|prior)\s+(\d+)\s*days?\b"), "n_days"),
    (re.compile(r"\b(?:last|past|previous|prior)\s+(\d+)\s*weeks?\b"), "n_weeks"),
    (re.compile(r"\b(?:last|past|previous|prior)\s+(\d+)\s*hours?\b"), "n_hours"),
)


def _resolve_relative_range(message):
    """Return {gte, lte} ISO dates for a relative range mentioned in the
    message ("last 6 months", "current month", ...), or None."""
    lowered = message.lower()
    today = datetime.now().date()
    import calendar as _cal

    def month_bounds(y, m):
        return (date(y, m, 1).isoformat(),
                date(y, m, _cal.monthrange(y, m)[1]).isoformat())

    for rx, kind in _RELATIVE_RANGE_RES:
        m = rx.search(lowered)
        if not m:
            continue
        if kind == "cur_month":
            gte, lte = month_bounds(today.year, today.month)
            return {"gte": gte, "lte": lte}
        if kind == "prev_month":
            first = (today.replace(day=1) - timedelta(days=1))
            gte, lte = month_bounds(first.year, first.month)
            return {"gte": gte, "lte": lte}
        if kind == "n_months":
            n = int(m.group(1))
            # Rolling window ending today: [today - n months + 1 day, today]
            anchor = today.replace(day=1)
            y, mo = anchor.year, anchor.month
            for _ in range(n):
                mo -= 1
                if mo == 0:
                    mo, y = 12, y - 1
            return {"gte": date(y, mo, anchor.day).isoformat(), "lte": today.isoformat()}
        if kind == "n_years":
            n = int(m.group(1))
            return {"gte": f"{today.year - n}-01-01", "lte": today.isoformat()}
        if kind == "n_days":
            n = int(m.group(1))
            return {"gte": (today - timedelta(days=n)).isoformat(), "lte": today.isoformat()}
        if kind == "n_weeks":
            n = int(m.group(1))
            return {"gte": (today - timedelta(weeks=n)).isoformat(), "lte": today.isoformat()}
        if kind == "n_hours":
            n = int(m.group(1))
            start_dt = datetime.now() - timedelta(hours=n)
            return {"gte": start_dt.strftime("%Y-%m-%d %H:%M:%S"), "lte": today.isoformat()}
        if kind == "cur_year":
            return {"gte": f"{today.year}-01-01", "lte": f"{today.year}-12-31"}
        if kind == "prev_year":
            return {"gte": f"{today.year-1}-01-01", "lte": f"{today.year-1}-12-31"}
    return None


def _extract_entities(message):
    """Pull locations, exact dates, and month/year ranges from the message."""
    lowered = message.lower()
    dates = _DATE_RE.findall(lowered)

    # Month (+ optional year) range: "summary for april 2026", "records in may"
    month_year = None
    year_match = _YEAR_RE.search(lowered)
    for i, name in enumerate(_MONTH_NAMES, 1):
        if name in lowered:
            month_year = {"month": i, "year": int(year_match.group(1)) if year_match else None}
            break

    # Bare year: "summarise records from year 2026" (only when no exact date given)
    year_only = int(year_match.group(1)) if (year_match and not month_year and not dates) else None

    locations = []
    for m in re.finditer(r"(?:\bin|\bat|\bnear|\bfor)\s+location\s+(?:as\s+)?([a-z][a-z\s]{2,40})", lowered):
        loc = m.group(1).strip().rstrip(" ?.,!")
        if loc and loc not in _JUNK_LOCATIONS:
            locations.append(loc)
    for m in re.finditer(r"location\s+(?:as\s+)?([a-z][a-z\s]{2,40}?)(?:\s+and\b|$|\?|with)", lowered):
        loc = m.group(1).strip()
        if loc and loc not in _JUNK_LOCATIONS and loc not in locations:
            locations.append(loc)
    return locations, dates, month_year, year_only


def _resolve_locations(locations):
    """Fuzzy-match user-typed place names against distinct values in the index.

    Fixes typos like 'afganistan' -> 'afghanistan'. Returns resolved names
    (original casing preserved) or [].
    """
    if not locations:
        return []
    try:
        res = es.search(index=ES_INDEX, body={
            "size": 0,
            "aggs": {"vals": {"terms": {"field": "location_name.keyword", "size": 5000}}},
        })
        known = [b["key"] for b in res["aggregations"]["vals"]["buckets"]
                 if b["key"].lower() not in _JUNK_LOCATIONS]
    except Exception as e:
        print(f"Location resolve exception: {e}")
        return locations
    import difflib
    lowered_known = {k.lower(): k for k in known}
    resolved = []
    for loc in locations:
        if loc in lowered_known:                      # exact
            resolved.append(lowered_known[loc])
            continue
        close = difflib.get_close_matches(loc, list(lowered_known.keys()), n=1, cutoff=0.75)
        if close:                                     # typo / spelling variant
            resolved.append(lowered_known[close[0]])
        else:                                         # substring match
            partial = [k for k in known if loc in k.lower()]
            if partial:
                resolved.append(partial[0])
    return resolved


def _search_by_date_range(month_year=None, year_only=None, max_docs=30):
    """Fetch docs whose activity_date falls in a month/year or whole year."""
    must = [{"exists": {"field": "activity_date"}}]
    if month_year:
        y = month_year["year"] or "*"
        start = f"{y}-{month_year['month']:02d}-01"
        import calendar
        last_day = calendar.monthrange(int(month_year["year"]) if month_year["year"] else 2024, month_year["month"])[1]
        end = f"{y}-{month_year['month']:02d}-{last_day:02d}"
        must.append({"range": {"activity_date": {"gte": start, "lte": end}}})
    elif year_only:
        must.append({"range": {"activity_date": {"gte": f"{year_only}-01-01",
                                                 "lte": f"{year_only}-12-31"}}})
    else:
        return None
    try:
        res = es.search(index=ES_INDEX, body={
            "query": {"bool": {"must": must}}, "size": max_docs, "sort": ["_doc"],
        })
        hits = res.get("hits", {}).get("hits", [])
        return hits or []
    except Exception as e:
        print(f"Date range search exception: {e}")
        return None


def _search_by_entity(locations, dates, max_docs=30, debug=None):
    """Exact-match retrieval: docs whose location field equals a named place
    (optionally filtered by activity date). Returns hits or None."""
    if not locations:
        return None
    should = []
    for loc in locations:
        variants = {loc, loc.title()}
        for variant in variants:
            for field in _LOCATION_FILTER_FIELDS:
                should.append({"term": {f"{field}.keyword": variant}})
    body = {
        "query": {"bool": {"should": should, "minimum_should_match": 1}},
        "size": max_docs,
        "sort": ["_doc"],
    }
    if dates:
        date_clauses = [{"term": {"activity_date.keyword": d}} for d in dates]
        body["query"]["bool"]["must"] = {"bool": {"should": date_clauses, "minimum_should_match": 0}}
    if debug is not None:
        debug.append({"name": "Entity exact-match (location/date)", "query": body})
    try:
        res = es.search(index=ES_INDEX, body=body)
        return res.get("hits", {}).get("hits", [])
    except Exception as e:
        print(f"Entity search exception: {e}")
        return None


def _retrieve_context(user_message, query_vector, debug=None):
    """Return (retrieved_context_text, num_results).

    Aggregation-style questions get a broad match_all fetch; everything else
    goes through multi-field kNN.

    When `debug` is a list, every Elasticsearch query executed (name + body)
    and the final fetched records are appended to it for UI inspection.
    """
    try:
        hits = None

        def _log(name, body):
            if debug is not None:
                debug.append({"name": name, "query": body})

        # 1) Entity retrieval: locations (fuzzy-resolved), exact dates,
        # month/year or year ranges. Beats kNN for named entities & dates.
        locations, dates, month_year, year_only = _extract_entities(user_message)
        resolved = []
        if locations:
            resolved = _resolve_locations(locations)
            if resolved:
                entity_hits = _search_by_entity(resolved, dates, debug=debug)
                if entity_hits:
                    hits = entity_hits
                elif not dates:
                    # Location known but date filtered nothing out; drop date filter
                    pass
        if hits is None and (month_year or year_only):
            range_hits = _search_by_date_range(month_year, year_only)
            if debug is not None:
                debug.append({"name": "Month/year date range", "query": {
                    "note": f"month={month_year} year_only={year_only}"}})
            if range_hits:
                hits = range_hits
        if hits is None and dates:
            # Exact date mentioned but no location: search all docs on that date
            date_clauses = [{"term": {"activity_date.keyword": d}} for d in dates]
            exact_body = {
                "query": {"bool": {"should": date_clauses, "minimum_should_match": 1}},
                "size": 30, "sort": ["_doc"],
            }
            _log("Exact date search", exact_body)
            res = es.search(index=ES_INDEX, body=exact_body)
            dhits = res.get("hits", {}).get("hits", [])
            if dhits:
                hits = dhits

        if hits is None:
            rel_range = _resolve_relative_range(user_message)
            if rel_range:
                rel_body = {
                    "query": {"bool": {
                        "must": [{"exists": {"field": "activity_date"}}],
                        "filter": [{"range": {"activity_date": rel_range}}],
                    }},
                    "size": 30, "sort": [{"activity_date": {"order": "desc"}}],
                }
                _log(f"Relative date range {rel_range['gte']} → {rel_range['lte']}", rel_body)
                try:
                    res = es.search(index=ES_INDEX, body=rel_body)
                    rhits = res.get("hits", {}).get("hits", [])
                    if rhits:
                        hits = rhits
                except Exception as e:
                    print(f"Relative range search exception: {e}")

        # 1b) Near-date fallback: when a location matched but the exact day
        # didn't, surface the closest dated records instead of nothing.
        if hits is None and resolved:
            target_ms = 0
            if dates and re.match(r"\d{4}-\d{2}-\d{2}", dates[0]):
                target_ms = int(datetime.strptime(dates[0], "%Y-%m-%d").timestamp() * 1000)
            body = {
                "query": {"bool": {"should": [
                    {"term": {f"{f}.keyword": v}} for v in resolved for f in _LOCATION_FILTER_FIELDS
                ], "minimum_should_match": 1,
                    "filter": [{"exists": {"field": "activity_date"}}]}},
                "size": 10,
                "sort": [{"_script": {
                    "type": "number",
                    "script": {
                        "lang": "painless",
                        "source": "long t = (long) params['target']; if (doc[params.fld].empty) return 9000000000000L; return Math.abs(doc[params.fld].value.toInstant().toEpochMilli() - t)",
                        "params": {"target": target_ms, "fld": "activity_date"},
                    },
                }}],
            }
            try:
                res = es.search(index=ES_INDEX, body=body)
                nhits = res.get("hits", {}).get("hits", [])
                if nhits:
                    hits = nhits
            except Exception as e:
                print(f"Near-date search exception: {e}")

        # 2) Full-database terms aggregation when the question asks to enumerate
        # a recognized field type (locations, equipment, radars, ...).
        if not hits:
            full_listing = _build_full_listing(user_message)
            if full_listing and _wants_enumeration(user_message):
                if debug is not None:
                    debug.append({"name": "Full-database terms aggregation", "query": {
                        "note": "terms aggregations over listable fields (see _LISTABLE_FIELDS)"}})
                return full_listing, 0  # num_results=0 signals aggregated listing
            if _is_aggregation_query(user_message):
                agg_body = {"query": {"match_all": {}}, "size": 50, "sort": ["_doc"]}
                _log("Broad fetch (match_all)", agg_body)
                hits = es.search(index=ES_INDEX, body=agg_body).get("hits", {}).get("hits", [])
                hits = hits or _fetch_all_docs()
            else:
                knn_queries = [
                    {"field": vf, "query_vector": query_vector, "k": 10, "num_candidates": 100}
                    for vf in VECTOR_FIELD_NAMES
                ]
                knn_body = {"knn": knn_queries, "_source": True}
                _log("Multi-field kNN semantic search", {
                    "knn_fields": VECTOR_FIELD_NAMES,
                    "k": 10, "num_candidates": 100})
                res = es.search(index=ES_INDEX, body=knn_body)
                hits = res.get("hits", {}).get("hits", [])

                # Keyword fallback: kNN is semantic and can miss exact token
                # mentions ("with mention of grenade"). If no top-kNN doc
                # actually contains the message's salient keywords, run a
                # plain match on combined_text and prefer its hits.
                _kw_terms = [w for w in re.findall(r"[a-z][a-z\-]{2,}", user_message.lower())
                             if w not in _STOPWORD_TERMS]
                if _kw_terms:
                    kw_body = {
                        "query": {"bool": {"should": [
                            {"match": {"combined_text": t}} for t in _kw_terms[:5]
                        ], "minimum_should_match": 1}},
                        "size": 10,
                    }
                    _log(f"Keyword fallback (combined_text: {_kw_terms[:5]})", kw_body)
                    try:
                        kw_res = es.search(index=ES_INDEX, body=kw_body)
                        khits = kw_res.get("hits", {}).get("hits", [])
                        # Only override when a keyword hit scores well AND the
                        # top kNN docs don't already contain those keywords.
                        if khits:
                            kw_ids = {h["_id"] for h in khits}
                            knn_has_kw = any(h["_id"] in kw_ids for h in hits)
                            if not knn_has_kw and (khits[0].get("_score") or 0) >= 5.0:
                                hits = khits
                    except Exception as e:
                        print(f"Keyword fallback exception: {e}")

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

        if debug is not None:
            debug.append({"name": "Fetched records (top 10)", "query": {
                "note": "documents fed to the LLM as context"},
                "records": [_hit_record(h) for h in unique_docs[:10]]})

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


def _hit_record(hit):
    """Flatten one ES hit into {id, score, fields{}} for UI display."""
    src = hit.get("_source", {})
    fields = {}
    for k, v in src.items():
        if k.startswith("vec_") or k == "combined_text" or v in (None, "", []):
            continue
        fields[k] = v
    return {"id": hit["_id"], "score": round(hit.get("_score") or 0, 3), "fields": fields}


# Per-turn retrieval debug info, keyed by turn id (bounded ring buffer).
_RETRIEVAL_DEBUG = {}
_DEBUG_MAX_TURNS = 50
_debug_lock = threading.Lock()


@app.route("/retrieval-debug/<turn_id>", methods=["GET"])
def retrieval_debug(turn_id):
    """Return the ES queries and fetched records used for a chat turn."""
    with _debug_lock:
        entry = _RETRIEVAL_DEBUG.get(turn_id)
    if not entry:
        return jsonify({"error": "unknown turn id"}), 404
    return jsonify(entry)


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
    t0 = time.time()
    query_vector = get_embedding(user_message)
    retrieval_debug = []
    retrieved_context, num_results = _retrieve_context(user_message, query_vector,
                                                       debug=retrieval_debug)
    turn_id = os.urandom(8).hex()
    with _debug_lock:
        _RETRIEVAL_DEBUG[turn_id] = {
            "question": user_message, "turn_id": turn_id,
            "queries": retrieval_debug}
        # Bounded ring buffer
        while len(_RETRIEVAL_DEBUG) > _DEBUG_MAX_TURNS:
            _RETRIEVAL_DEBUG.pop(next(iter(_RETRIEVAL_DEBUG)))

    # 4. Construct Context-Aware Prompt using Local Memory Buffers
    recent_history_text = "\n".join([f"{t['role']}: {t['content']}" for t in session['history']])
    total_docs = _get_doc_count()

    system_prompt = f"""You are an intelligent, helpful RAG chatbot for a military intelligence database.

DATABASE INFO:
- Today's date: {datetime.now().strftime('%A, %Y-%m-%d')}. Use this to interpret relative time
  references ("current month", "last month", "recent"). Documents dated within the current
  calendar month ({datetime.now().strftime('%B %Y')}) ARE current month records — do not claim
  none of the documents contain current month data when their activity dates fall in this month.
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
- If a section labeled "FULL DATABASE AGGREGATION" is present, it contains exact distinct values and record counts computed over the ENTIRE database. This data IS complete and exhaustive — enumerate the values in your answer (grouped/summarized ONLY if there are more than ~30) and NEVER say it may be incomplete, non-exhaustive, or a sample.
- When you present values from a FULL DATABASE AGGREGATION, do not add disclaimers like "this is not an exhaustive list". The aggregation scanned every record in the database.
- When referencing a document to the user, cite it as its stable id shown in parentheses, e.g. "document (id: abc123)".
- When listing data, use the actual values from the documents.
- Format your response clearly with bullet points or tables when presenting multiple records.
- ANALYZE, don't just dump. When presenting retrieved records, synthesize them into a meaningful
  summary: group related records (by location, date, unit, equipment type), state counts and
  date ranges, highlight patterns (recurring locations, repeated activity types, concentration
  in time), and call out the most significant or recent items. End with a short "Key
  Observations" section of 2-4 analytical takeaways drawn from the data. Never answer with a
  bare list of one field's values when richer context is present in the documents.
- PREDICTIONS: When asked to predict future outcomes, trends, or possibilities, you may reason
  beyond the documents, but ONLY under these rules:
  1. First present what the data shows (patterns, frequencies, date clustering, recurring
     units/equipment/locations) — every factual claim must trace to a cited document.
  2. Label every forward-looking statement explicitly as "Prediction:" or "Possible outcome:"
     and justify it from an observed pattern (e.g. "activity at X recurred monthly in the data,
     so continued activity is plausible").
  3. State confidence qualitatively (low/moderate/high) based on how strong and recent the
     pattern is; a pattern seen once justifies low confidence only.
  4. If the retrieved records are too few or show no meaningful pattern to base a prediction on,
     say so honestly instead of speculating.
  5. Never present predictions as established facts.
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
        "turn_id": turn_id,
        "debug_summary": session['chat_summary'],
        "debug_history_len": len(session['history']),
        "elapsed_seconds": round(time.time() - t0, 1)
    })


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear memory registers to start a fresh chat session."""
    session.clear()
    return jsonify({"status": "Session wiped successfully."})


# Embedded Chat UI Template
# Markdown rendering: tries marked + DOMPurify from CDN; falls back to a tiny
# built-in renderer (bold / italics / code / bullets / tables) when offline.
HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Tectum RAG Chatbot</title>
    <script src="https://cdn.jsdelivr.net/npm/marked@12/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3/dist/purify.min.js"></script>
    <style>
        :root {
            --bg: #eef1f6;
            --panel: #ffffff;
            --header: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            --accent: #2563eb;
            --user-bubble: linear-gradient(135deg, #2563eb, #3b82f6);
            --bot-bubble: #f4f6fa;
            --text: #1f2937;
            --muted: #6b7280;
            --border: #e5e7eb;
        }
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: var(--bg);
            margin: 0; padding: 20px;
            display: flex; justify-content: center; align-items: center;
            min-height: calc(100vh - 40px);
        }
        .chat-container {
            max-width: 820px; width: 100%; height: 85vh;
            background: var(--panel); border-radius: 16px;
            box-shadow: 0 10px 30px rgba(15, 23, 42, .12);
            display: flex; flex-direction: column; overflow: hidden;
        }
        .chat-header {
            background: var(--header); color: white; padding: 16px 20px;
            display: flex; justify-content: space-between; align-items: center;
        }
        .chat-header .title { font-size: 17px; font-weight: 700; letter-spacing: .2px; }
        .chat-header .subtitle { font-size: 12px; opacity: .85; margin-top: 2px; }
        .clear-btn {
            background: rgba(255,255,255,.18); color: white; border: 1px solid rgba(255,255,255,.35);
            padding: 6px 14px; border-radius: 999px; cursor: pointer; font-size: 12px;
            transition: background .2s;
        }
        .clear-btn:hover { background: rgba(255,255,255,.32); }

        .chat-box { flex: 1; overflow-y: auto; padding: 24px; scroll-behavior: smooth; }

        .row { display: flex; gap: 10px; margin-bottom: 18px; align-items: flex-start; }
        .row.user { flex-direction: row-reverse; }
        .avatar {
            width: 34px; height: 34px; border-radius: 50%; flex-shrink: 0;
            display: flex; align-items: center; justify-content: center;
            font-size: 15px; color: white; user-select: none;
        }
        .row.bot .avatar { background: var(--accent); }
        .row.user .avatar { background: #059669; }
        .bubble-wrap { max-width: 82%; display: flex; flex-direction: column; }
        .row.user .bubble-wrap { align-items: flex-end; }
        .message {
            padding: 12px 16px; border-radius: 14px; line-height: 1.55;
            font-size: 14.5px; color: var(--text); word-break: break-word;
        }
        .message.bot { background: var(--bot-bubble); border: 1px solid var(--border); border-top-left-radius: 4px; }
        .message.user { background: var(--user-bubble); color: white; border-top-right-radius: 4px; }
        .msg-meta { font-size: 11px; color: var(--muted); margin-top: 5px; padding: 0 4px; }

        /* Markdown content styling */
        .message.bot h1, .message.bot h2, .message.bot h3,
        .message.bot h4, .message.bot h5 { margin: 14px 0 8px; color: #111827; line-height: 1.3; }
        .message.bot h1 { font-size: 19px; } .message.bot h2 { font-size: 17px; } .message.bot h3 { font-size: 15px; }
        .message.bot p { margin: 8px 0; }
        .message.bot p:first-child { margin-top: 0; }
        .message.bot p:last-child { margin-bottom: 0; }
        .message.bot ul, .message.bot ol { margin: 8px 0; padding-left: 22px; }
        .message.bot li { margin: 4px 0; }
        .message.bot li::marker { color: var(--accent); font-weight: bold; }
        .message.bot strong { color: #111827; }
        .message.bot code {
            background: #e8edf5; padding: 1px 5px; border-radius: 4px;
            font-family: Consolas, monospace; font-size: 13px; color: #b91c1c;
        }
        .message.bot pre {
            background: #0f172a; color: #e2e8f0; padding: 12px 14px;
            border-radius: 8px; overflow-x: auto; margin: 10px 0;
        }
        .message.bot pre code { background: none; color: inherit; padding: 0; }
        .message.bot blockquote {
            margin: 8px 0; padding: 6px 14px; border-left: 3px solid var(--accent);
            background: #eff6ff; border-radius: 0 6px 6px 0; color: #374151;
        }
        .message.bot hr { border: none; border-top: 1px solid var(--border); margin: 12px 0; }
        .message.bot table {
            border-collapse: collapse; margin: 10px 0; width: 100%;
            font-size: 13.5px; display: block; overflow-x: auto;
        }
        .message.bot th, .message.bot td { border: 1px solid var(--border); padding: 6px 10px; text-align: left; }
        .message.bot th { background: #eff6ff; color: #1e3a8a; font-weight: 600; }
        .message.bot tr:nth-child(even) td { background: #f9fafb; }

        /* Typing indicator */
        .typing { display: inline-flex; gap: 5px; padding: 4px 2px; }
        .typing span {
            width: 8px; height: 8px; border-radius: 50%; background: #94a3b8;
            animation: bounce 1.2s infinite ease-in-out;
        }
        .typing span:nth-child(2) { animation-delay: .15s; }
        .typing span:nth-child(3) { animation-delay: .3s; }
        @keyframes bounce { 0%,60%,100% { transform: translateY(0); opacity:.5;} 30% { transform: translateY(-5px); opacity:1;} }

        .input-area {
            display: flex; gap: 10px; padding: 14px 18px; background: #fafbfd;
            border-top: 1px solid var(--border);
        }
        input {
            flex: 1; padding: 12px 16px; border: 1.5px solid var(--border);
            border-radius: 999px; font-size: 14px; outline: none; transition: border-color .2s;
        }
        input:focus { border-color: var(--accent); }
        .send-btn {
            background: var(--user-bubble); color: white; border: none;
            padding: 0 24px; border-radius: 999px; cursor: pointer; font-size: 14px; font-weight: 600;
            transition: transform .15s, box-shadow .15s;
        }
        .send-btn:hover { box-shadow: 0 4px 12px rgba(37, 99, 235, .35); transform: translateY(-1px); }
        .send-btn:disabled { opacity: .6; cursor: wait; transform: none; box-shadow: none; }

        /* Per-message debug button */
        .debug-btn {
            align-self: flex-start; margin-top: 4px;
            background: #eef2ff; color: #4338ca; border: 1px solid #c7d2fe;
            border-radius: 999px; font-size: 11px; padding: 3px 10px; cursor: pointer;
            transition: background .15s;
        }
        .debug-btn:hover { background: #e0e7ff; }

        /* Modal */
        .modal-overlay {
            display: none; position: fixed; inset: 0; background: rgba(15,23,42,.55);
            z-index: 1000; justify-content: center; align-items: center; padding: 24px;
        }
        .modal-overlay.open { display: flex; }
        .modal {
            background: white; border-radius: 14px; width: min(960px, 100%);
            max-height: 88vh; display: flex; flex-direction: column;
            box-shadow: 0 24px 60px rgba(0,0,0,.25); overflow: hidden;
        }
        .modal-header {
            padding: 14px 20px; background: var(--header); color: white;
            display: flex; justify-content: space-between; align-items: center;
        }
        .modal-header h3 { margin: 0; font-size: 15px; font-weight: 600; }
        .modal-close {
            background: rgba(255,255,255,.18); color: white; border: none;
            border-radius: 50%; width: 28px; height: 28px; cursor: pointer; font-size: 14px;
        }
        .modal-close:hover { background: rgba(255,255,255,.32); }
        .tabs { display: flex; gap: 4px; padding: 10px 16px 0; border-bottom: 1px solid var(--border); }
        .tab-btn {
            background: none; border: none; padding: 8px 16px; cursor: pointer;
            font-size: 13.5px; color: var(--muted); border-bottom: 2.5px solid transparent;
            font-weight: 600;
        }
        .tab-btn.active { color: var(--accent); border-bottom-color: var(--accent); }
        .modal-body { overflow-y: auto; padding: 16px 20px; }

        /* Queries panel */
        .q-block { margin-bottom: 14px; border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }
        .q-title {
            background: #f8fafc; padding: 8px 12px; font-size: 12.5px; font-weight: 600;
            color: #1e3a8a; border-bottom: 1px solid var(--border);
        }
        .q-json {
            margin: 0; padding: 10px 12px; background: #0f172a; color: #a5d6ff;
            font-family: Consolas, monospace; font-size: 12px; white-space: pre-wrap;
            word-break: break-word; max-height: 260px; overflow-y: auto;
        }

        /* Records table */
        .tbl-info { font-size: 12px; color: var(--muted); margin-bottom: 8px; }
        .records-table { width: 100%; border-collapse: collapse; font-size: 12.5px; table-layout: auto; }
        .records-table th {
            background: #eff6ff; color: #1e3a8a; text-align: left; position: sticky; top: 0;
            padding: 8px 10px; border: 1px solid var(--border); white-space: nowrap;
        }
        .records-table td {
            border: 1px solid var(--border); padding: 7px 10px; vertical-align: top;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            max-width: 260px; cursor: pointer;
        }
        .records-table td.expanded {
            white-space: normal; overflow: visible; text-overflow: clip; cursor: default;
        }
        .records-table tr:nth-child(even) td { background: #f9fafb; }
        .rec-id { font-family: Consolas, monospace; font-size: 11px; color: #6d28d9; white-space: nowrap; }
        .pagination { display: flex; gap: 6px; align-items: center; justify-content: center; margin-top: 12px; flex-wrap: wrap; }
        .page-btn {
            border: 1px solid var(--border); background: white; color: var(--text);
            border-radius: 6px; padding: 5px 11px; cursor: pointer; font-size: 12.5px;
        }
        .page-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
        .page-btn.active { background: var(--accent); color: white; border-color: var(--accent); }
        .page-btn:disabled { opacity: .45; cursor: default; }
        .empty-note { color: var(--muted); font-size: 13px; text-align: center; padding: 24px 0; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div>
                <div class="title">🛰 Tectum RAG Chatbot</div>
                <div class="subtitle">Elasticsearch vector search + LLM</div>
            </div>
            <button class="clear-btn" onclick="clearChat()">Clear Memory</button>
        </div>
        <div class="chat-box" id="chatBox"></div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Ask about your indexed documents…" onkeypress="handleKey(event)" autocomplete="off">
            <button class="send-btn" id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <div class="modal-overlay" id="debugModal">
        <div class="modal">
            <div class="modal-header">
                <h3 id="modalTitle">Retrieval details</h3>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <div class="tabs">
                <button class="tab-btn active" id="tabQueriesBtn" onclick="showTab('queries')">Queries</button>
                <button class="tab-btn" id="tabRecordsBtn" onclick="showTab('records')">Records</button>
            </div>
            <div class="modal-body" id="modalBody"></div>
        </div>
    </div>

    <script>
        const chatBox = document.getElementById('chatBox');
        const sendBtn = document.getElementById('sendBtn');
        const userInput = document.getElementById('userInput');

        function escapeHtml(text) {
            return text.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
        }

        // Render LLM markdown to safe HTML.
        // Primary path: marked + DOMPurify from CDN. Fallback: minimal renderer
        // (headings, bold, italics, code, lists, tables, paragraphs) for offline use.
        function renderMarkdown(text) {
            if (window.marked && window.DOMPurify) {
                return DOMPurify.sanitize(marked.parse(text));
            }
            let html = escapeHtml(text);
            html = html.replace(/^###### (.*)$/gm, '<h5>$1</h5>')
                       .replace(/^##### (.*)$/gm, '<h5>$1</h5>')
                       .replace(/^#### (.*)$/gm, '<h4>$1</h4>')
                       .replace(/^### (.*)$/gm, '<h3>$1</h3>')
                       .replace(/^## (.*)$/gm, '<h2>$1</h2>')
                       .replace(/^# (.*)$/gm, '<h1>$1</h1>');
            html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
                       .replace(/(^|\\W)\\*([^*\\n]+)\\*(?=\\W|$)/g, '$1<em>$2</em>')
                       .replace(/`([^`]+)`/g, '<code>$1</code>');
            // bullet lists
            html = html.replace(/(?:^|\\n)((?:[ \\t]*[-*•][^\\n]*(?:\\n|$))+)/g, function(m, block){
                const items = block.trim().split('\\n').map(l =>
                    '<li>' + l.replace(/^[ \\t]*[-*•][ \\t]*/, '') + '</li>').join('');
                return '\\n<ul>' + items + '</ul>\\n';
            });
            // simple pipe tables
            html = html.replace(/(?:^|\\n)(\\|.+\\|)(?:\\n\\|[ :-]+\\|)((?:\\n\\|.+\\|)+)/g, function(m, head, rows){
                const cells = r => r.split('|').slice(1, -1).map(c => c.trim());
                const th = cells(head).map(c => '<th>'+c+'</th>').join('');
                const trs = rows.trim().split('\\n').map(r =>
                    '<tr>' + cells(r).map(c => '<td>'+c+'</td>').join('') + '</tr>').join('');
                return '\\n<table><thead><tr>'+th+'</tr></thead><tbody>'+trs+'</tbody></table>\\n';
            });
            html = html.replace(/\\n{2,}/g, '</p><p>').replace(/\\n/g, '<br>');
            return '<p>' + html + '</p>';
        }

        function nowTime() {
            return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        }

        function addRow(sender) {
            const row = document.createElement('div');
            row.className = 'row ' + sender;
            const avatar = document.createElement('div');
            avatar.className = 'avatar';
            avatar.textContent = sender === 'user' ? '🧑' : '🤖';
            const wrap = document.createElement('div');
            wrap.className = 'bubble-wrap';
            const bubble = document.createElement('div');
            bubble.className = 'message ' + sender;
            wrap.appendChild(bubble);
            row.appendChild(avatar);
            row.appendChild(wrap);
            chatBox.appendChild(row);
            return { row, wrap, bubble };
        }

        function appendMessage(text, sender, elapsedSeconds, asMarkdown, turnId) {
            const { wrap, bubble } = addRow(sender);
            if (sender === 'user') {
                bubble.textContent = text;
            } else if (asMarkdown === false) {
                bubble.textContent = text;
            } else {
                bubble.innerHTML = renderMarkdown(text);
            }
            const meta = document.createElement('span');
            meta.className = 'msg-meta';
            let label = nowTime();
            if (sender === 'bot' && elapsedSeconds) label += ' · ' + elapsedSeconds + 's';
            meta.innerText = label;
            wrap.appendChild(meta);
            // Debug button: show ES queries + records used for this answer
            if (turnId) {
                const btn = document.createElement('button');
                btn.className = 'debug-btn';
                btn.textContent = '🔍 View queries & data';
                btn.onclick = () => openDebugModal(turnId, text);
                wrap.appendChild(btn);
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }

        // ---- Retrieval debug modal ----
        let debugData = null;      // payload from /retrieval-debug/<id>
        let debugAnswer = '';      // the chat answer for header context
        let currentTab = 'queries';
        let recordsPage = 1;
        const PAGE_SIZE = 10;

        async function openDebugModal(turnId, answerText) {
            debugAnswer = answerText || '';
            document.getElementById('modalTitle').textContent = 'Retrieval details';
            document.getElementById('modalBody').innerHTML =
                '<div class="empty-note">Loading…</div>';
            document.getElementById('debugModal').classList.add('open');
            try {
                const res = await fetch('/retrieval-debug/' + turnId);
                debugData = await res.json();
            } catch(e) {
                debugData = null;
                document.getElementById('modalBody').innerHTML =
                    '<div class="empty-note">⚠️ Failed to load retrieval details.</div>';
                return;
            }
            recordsPage = 1;
            renderTab();
        }

        function closeModal() {
            document.getElementById('debugModal').classList.remove('open');
        }
        document.getElementById('debugModal').addEventListener('click', function(e){
            if (e.target === this) closeModal();
        });
        document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

        function showTab(tab) {
            currentTab = tab;
            document.getElementById('tabQueriesBtn').classList.toggle('active', tab === 'queries');
            document.getElementById('tabRecordsBtn').classList.toggle('active', tab === 'records');
            renderTab();
        }

        function renderTab() {
            const body = document.getElementById('modalBody');
            if (!debugData || !debugData.queries) return;
            if (currentTab === 'queries') {
                body.innerHTML = debugData.queries.length
                    ? debugData.queries.map(q =>
                        '<div class="q-block"><div class="q-title">' + escapeHtml(q.name) +
                        '</div><pre class="q-json">' +
                        escapeHtml(JSON.stringify(q.query, null, 2)) + '</pre></div>'
                      ).join('')
                    : '<div class="empty-note">No queries recorded.</div>';
            } else {
                body.innerHTML = renderRecordsTable();
                bindPager();
            }
        }

        // Collect all records from every query entry that carries them.
        function collectRecords() {
            const recs = [];
            (debugData.queries || []).forEach(q => (q.records || []).forEach(r => recs.push(r)));
            return recs;
        }

        function fieldLabel(key) {
            return key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
        }

        function fmtVal(v) {
            if (Array.isArray(v)) return JSON.stringify(v);
            if (v !== null && typeof v === 'object') return JSON.stringify(v);
            return String(v);
        }

        function renderRecordsTable() {
            const recs = collectRecords();
            if (!recs.length) {
                return '<div class="empty-note">No records were fetched for this question' +
                       ' (e.g. it was answered from a full-database aggregation).</div>';
            }
            // Union of fields across records, ordered sensibly
            const allFields = [];
            recs.forEach(r => Object.keys(r.fields).forEach(k => {
                if (!allFields.includes(k)) allFields.push(k);
            }));
            const priority = ['activity_date','location_name','description','equipment_name',
                              'enemy_formation_name'];
            allFields.sort((a,b) => {
                const ia = priority.indexOf(a), ib = priority.indexOf(b);
                return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
            });

            const totalPages = Math.max(1, Math.ceil(recs.length / PAGE_SIZE));
            if (recordsPage > totalPages) recordsPage = totalPages;
            const start = (recordsPage - 1) * PAGE_SIZE;
            const pageRecs = recs.slice(start, start + PAGE_SIZE);

            let html = '<div class="tbl-info">' + recs.length + ' record(s) · showing ' +
                (start + 1) + '–' + (start + pageRecs.length) + ' of ' + totalPages + ' page(s)</div>';
            html += '<table class="records-table"><thead><tr><th>#</th><th>Doc ID</th>' +
                    '<th>Score</th>' +
                    allFields.map(f => '<th>' + escapeHtml(fieldLabel(f)) + '</th>').join('') +
                    '</tr></thead><tbody>';
            pageRecs.forEach((r, i) => {
                html += '<tr><td>' + (start + i + 1) + '</td>' +
                        '<td class="rec-id">' + escapeHtml(r.id) + '</td>' +
                        '<td>' + r.score + '</td>' +
                        allFields.map(f => {
                            const v = r.fields[f];
                            if (v === undefined) return '<td></td>';
                            const full = escapeHtml(fmtVal(v));
                            return '<td class="cell-clamp" title="Click to expand">' + full + '</td>';
                        }).join('') + '</tr>';
            });
            html += '</tbody></table>';

            html += '<div class="pagination">';
            html += '<button class="page-btn" id="pgPrev" ' + (recordsPage<=1?'disabled':'') + '>‹ Prev</button>';
            const winStart = Math.max(1, Math.min(recordsPage - 2, totalPages - 4));
            const winEnd = Math.min(totalPages, winStart + 4);
            for (let p = Math.max(1,winStart); p <= winEnd; p++) {
                html += '<button class="page-btn pg-num ' + (p===recordsPage?'active':'') +
                        '" data-page="' + p + '">' + p + '</button>';
            }
            html += '<button class="page-btn" id="pgNext" ' + (recordsPage>=totalPages?'disabled':'') + '>Next ›</button>';
            html += '</div>';
            return html;
        }

        function bindPager() {
            const prev = document.getElementById('pgPrev');
            const next = document.getElementById('pgNext');
            if (prev) prev.onclick = () => { recordsPage--; renderTab(); };
            if (next) next.onclick = () => { recordsPage++; renderTab(); };
            document.querySelectorAll('.pg-num').forEach(b => {
                b.onclick = () => { recordsPage = parseInt(b.dataset.page); renderTab(); };
            });
            // Expand / collapse long cell text on click
            document.querySelectorAll('.cell-clamp').forEach(td => {
                td.onclick = () => td.classList.toggle('expanded');
            });
        }

        function showTyping() {
            const { row, wrap, bubble } = addRow('bot');
            bubble.innerHTML = '<span class="typing"><span></span><span></span><span></span></span>';
            chatBox.scrollTop = chatBox.scrollHeight;
            return row;
        }

        async function sendMessage() {
            const message = userInput.value.trim();
            if (!message || sendBtn.disabled) return;

            appendMessage(message, 'user');
            userInput.value = '';
            sendBtn.disabled = true;
            const startedAt = Date.now();
            const typingRow = showTyping();

            try {
                const res = await fetch('/chat', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ message })
                });
                typingRow.remove();
                if (res.status === 503) {
                    appendMessage('⚠️ The LLM backend is not responding. Please try again.', 'bot', 0, false);
                    return;
                }
                const data = await res.json();
                const elapsed = data.elapsed_seconds || ((Date.now() - startedAt) / 1000).toFixed(1);
                appendMessage(data.response, 'bot', elapsed, true, data.turn_id);
            } catch(e) {
                typingRow.remove();
                appendMessage('⚠️ Failed to get a response.', 'bot', 0, false);
            } finally {
                sendBtn.disabled = false;
                userInput.focus();
            }
        }

        function handleKey(e) { if (e.key === 'Enter') sendMessage(); }

        async function clearChat() {
            await fetch('/clear', { method: 'POST' });
            chatBox.innerHTML = '';
            appendMessage('Memory cleared! How can I help you now?', 'bot', 0, false);
        }

        // Greeting
        appendMessage('Hello! 👋 Ask me anything regarding your indexed documents.', 'bot', 0, false);
        userInput.focus();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT, debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
