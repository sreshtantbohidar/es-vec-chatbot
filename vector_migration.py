import os
import sys
import time
import argparse
import requests
from datetime import timedelta
from elasticsearch import Elasticsearch, helpers
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

# 1. Configuration (connection details from .env)
ELASTIC_URL = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")
ES_VERIFY_CERTS = os.getenv("ES_VERIFY_CERTS", "false").lower() == "true"
EMBED_MODEL = os.getenv("EMBED_MODEL") or "nomic-embed-text"
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "768"))
OLLAMA_BASE_URL = (os.getenv("OLLAMA_URL") or os.getenv("LLM_BASE_URL") or "http://localhost:11434").rstrip("/")
SOURCE_INDEX = "fatboy_data"
NEW_INDEX = "vec_chat_fatboy_data"

# Collect ALL unique fields across every analysis type, paired with labels.
# Each field will get its own dense_vector column in Elasticsearch.
FIELD_LABEL_PAIRS = []
_seen_fields = set()
for _mapping in TYPE_MAPPING.values():
    for _field, _label in zip(_mapping["fields"], _mapping["field_labels"]):
        if _field not in _seen_fields:
            _seen_fields.add(_field)
            FIELD_LABEL_PAIRS.append((_field, _label))


def _to_embed_text(field_name, val):
    """Convert a field value to a clean string suitable for embedding.
    
    - Lists (e.g. coordinates [lat, lng]) → "lat, lng"
    - Numbers → str(val)
    - Strings → stripped str(val)
    """
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val).strip()

VECTOR_FIELD_PREFIX = "vec_"

# Initialize clients
es = Elasticsearch(
    ELASTIC_URL,
    basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
    verify_certs=ES_VERIFY_CERTS,
    request_timeout=120,
)


def _embed_one(text, max_retries=3):
    """Single Ollama embedding call with retry on transient 5xx errors.
    
    Logs full response body on 400 errors for diagnosis.
    """
    url = OLLAMA_BASE_URL + "/api/embeddings"
    payload = {"model": EMBED_MODEL, "prompt": text}
    last_response = None
    for attempt in range(max_retries):
        response = requests.post(url, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json()["embedding"]
        last_response = response
        # Log the full error body on first 400 so we can diagnose
        if response.status_code == 400 and attempt == 0:
            print(f"    [400 BODY] {response.text[:500]}")
            print(f"    [400 PAYLOAD] model={payload['model']} prompt_len={len(payload['prompt'])} prompt_preview={repr(payload['prompt'][:80])}")
        if response.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
            wait = 2 ** attempt
            print(f"    Retry {attempt+1}/{max_retries} (status {response.status_code}), waiting {wait}s...")
            time.sleep(wait)
            continue
        # On 400, also retry once after a longer pause (Ollama may need model reload)
        if response.status_code == 400 and attempt < max_retries - 1:
            wait = 3
            print(f"    Retry {attempt+1}/{max_retries} (status 400), waiting {wait}s...")
            time.sleep(wait)
            continue
    last_response.raise_for_status()
    return last_response.json()["embedding"]


def get_embedding(text):
    """Generate embedding via Ollama's /api/embeddings endpoint.

    Individual field values are short enough to fit nomic-embed-text's
    context window, so no chunking is needed here.
    """
    return _embed_one(text)


# 2. Build index mapping: one dense_vector per unique field
index_mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {}
    }
}

for _field, _label in FIELD_LABEL_PAIRS:
    vec_field_name = VECTOR_FIELD_PREFIX + _field
    index_mapping["mappings"]["properties"][vec_field_name] = {
        "type": "dense_vector",
        "dims": EMBED_DIMS,
        "index": True,
        "similarity": "cosine"
    }


def create_new_index():
    if es.indices.exists(index=NEW_INDEX):
        es.indices.delete(index=NEW_INDEX)
    es.indices.create(index=NEW_INDEX, body=index_mapping)
    print(f"Created new index: {NEW_INDEX} with {len(FIELD_LABEL_PAIRS)} vector fields")


def _format_eta(seconds):
    """Format seconds into human-readable string."""
    return str(timedelta(seconds=int(seconds)))


def _print_separator():
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Migrate documents from ES source index to vectorized target index."
    )
    parser.add_argument(
        "--doc", type=int, default=None, metavar="N",
        help="Process only the first N documents (default: all documents)"
    )
    parser.add_argument(
        "--skip-index", action="store_true",
        help="Skip index creation (use existing index)"
    )
    return parser.parse_args()


def migrate_and_vectorize(max_docs=None):
    # --- Pre-flight: count total documents ---
    _print_separator()
    print("PRE-FLIGHT CHECK")
    _print_separator()

    try:
        count_resp = es.count(index=SOURCE_INDEX)
        total_docs = count_resp["count"]
    except Exception as e:
        print(f"Warning: Could not count source docs ({e}), proceeding anyway...")
        total_docs = None

    # Count non-empty fields across all types
    total_fields = len(FIELD_LABEL_PAIRS)
    total_possible_embeddings = (total_docs * total_fields) if total_docs else "unknown"

    # Determine actual doc limit
    if max_docs is not None and total_docs:
        effective_total = min(max_docs, total_docs)
    elif max_docs is not None:
        effective_total = max_docs
    else:
        effective_total = total_docs

    print(f"  Source index:        {SOURCE_INDEX}")
    print(f"  Target index:        {NEW_INDEX}")
    print(f"  Total in source:     {total_docs or 'scanning...'}")
    print(f"  Processing limit:    {max_docs or 'ALL'} docs")
    print(f"  Fields per document: {total_fields}")
    print(f"  Embedding model:     {EMBED_MODEL} ({EMBED_DIMS} dims)")
    print(f"  Ollama endpoint:     {OLLAMA_BASE_URL}")

    # Validate critical config
    if not EMBED_MODEL:
        print("\n  FATAL: EMBED_MODEL is empty! Set it in .env or leave unset for default.")
        sys.exit(1)
    if not OLLAMA_BASE_URL:
        print("\n  FATAL: Ollama URL is empty! Set OLLAMA_URL or LLM_BASE_URL in .env.")
        sys.exit(1)
    print(f"  Batch size:          50 docs")
    print(f"  Retry attempts:      3 (exponential backoff)")
    est = (effective_total * total_fields) if effective_total else "unknown"
    print(f"  Est. embed calls:    {est}")
    _print_separator()

    # --- Begin migration ---
    query = {"query": {"match_all": {}}}
    scan_iterator = helpers.scan(es, query=query, index=SOURCE_INDEX, scroll="5m", size=100)

    batch = []
    batch_size = 20
    total_indexed = 0
    total_embed_calls = 0
    total_empty_fields = 0
    total_errors = 0
    start_time = time.time()
    last_print_time = start_time

    print("\nMIGRATION IN PROGRESS")
    _print_separator()

    for doc_num, doc in enumerate(scan_iterator, start=1):
        # Stop if we've hit the --doc limit
        if max_docs and doc_num > max_docs:
            break
        source_data = doc["_source"]
        doc_id = doc["_id"]
        fields_embedded = 0
        fields_empty = 0
        fields_failed = 0

        # Embed each field separately
        for field, label in FIELD_LABEL_PAIRS:
            vec_field_name = VECTOR_FIELD_PREFIX + field
            val = source_data.get(field)
            if val and str(val).strip():
                try:
                    embed_text = _to_embed_text(field, val)
                    embedding = get_embedding(embed_text)
                    source_data[vec_field_name] = embedding
                    total_embed_calls += 1
                    fields_embedded += 1
                except Exception as e:
                    print(f"  [ERROR] doc {doc_id[:12]}.. field='{field}': {e}")
                    source_data[vec_field_name] = [0.0] * EMBED_DIMS
                    fields_failed += 1
                    total_errors += 1
                time.sleep(0.1)
            else:
                source_data[vec_field_name] = [0.0] * EMBED_DIMS
                fields_empty += 1
                total_empty_fields += 1

        # Build combined_text for readability
        text_parts = []
        for field, label in FIELD_LABEL_PAIRS:
            val = source_data.get(field)
            if val and str(val).strip():
                text_parts.append(f"{label}: {_to_embed_text(field, val)}")
        source_data["combined_text"] = " ".join(text_parts)

        action = {
            "_index": NEW_INDEX,
            "_id": doc_id,
            "_source": source_data
        }
        batch.append(action)

        # --- Verbose per-doc log (every 10 docs or on error) ---
        now = time.time()
        if doc_num % 10 == 0 or fields_failed > 0 or (now - last_print_time) > 5:
            elapsed = now - start_time
            speed = total_embed_calls / elapsed if elapsed > 0 else 0
            display_total = effective_total or total_docs or '?'
            remaining = (effective_total or total_docs or 0) - total_indexed - len(batch)
            eta = remaining / speed if speed > 0 and remaining and remaining > 0 else None
            pct = (doc_num / display_total * 100) if display_total != '?' else 0

            print(
                f"  [{doc_num:>6}/{display_total}] "
                f"({pct:5.1f}%) "
                f"embed={fields_embedded}/{total_fields} "
                f"empty={fields_empty} "
                f"fail={fields_failed} "
                f"| total_embeds={total_embed_calls} errors={total_errors} "
                f"| elapsed={_format_eta(elapsed)} "
                f"speed={speed:.1f} embed/s "
                f"| ETA={_format_eta(eta) if eta else '?'}"
            )
            last_print_time = now

        # --- Flush batch to ES ---
        if len(batch) >= batch_size:
            bulk_start = time.time()
            success, errors = helpers.bulk(es, batch, raise_on_error=False)
            bulk_time = time.time() - bulk_start
            total_indexed += len(batch)
            if errors:
                # Show first 3 errors for diagnosis
                for err in errors[:3]:
                    op_type = list(err.keys())[0] if err else '?'
                    doc_info = err.get(op_type, {})
                    reason = doc_info.get('error', {}).get('reason', str(doc_info.get('error', '')))[:200]
                    print(f"  >> BATCH ERROR [{op_type}] id={doc_info.get('_id', '?')}: {reason}")
                if len(errors) > 3:
                    print(f"  >> ... and {len(errors) - 3} more errors")
            print(
                f"  >> BATCH FLUSHED: {success}/{len(batch)} docs to ES ({bulk_time:.1f}s) "
                f"| total_indexed={total_indexed}"
            )
            batch = []

    # --- Flush remaining batch ---
    if batch:
        success, errors = helpers.bulk(es, batch, raise_on_error=False)
        total_indexed += len(batch)
        if errors:
            for err in errors[:3]:
                op_type = list(err.keys())[0] if err else '?'
                doc_info = err.get(op_type, {})
                reason = doc_info.get('error', {}).get('reason', str(doc_info.get('error', '')))[:200]
                print(f"  >> FINAL BATCH ERROR [{op_type}] id={doc_info.get('_id', '?')}: {reason}")

    elapsed_total = time.time() - start_time
    _print_separator()
    print("MIGRATION COMPLETE")
    _print_separator()
    print(f"  Documents indexed:     {total_indexed}")
    print(f"  Embeddings generated:  {total_embed_calls}")
    print(f"  Empty fields (zero vec): {total_empty_fields}")
    print(f"  Embedding errors:      {total_errors}")
    print(f"  Total time:            {_format_eta(elapsed_total)}")
    print(f"  Avg speed:             {total_embed_calls / elapsed_total:.1f} embed/s" if elapsed_total > 0 else "")
    _print_separator()


if __name__ == "__main__":
    args = parse_args()
    _print_separator()
    print("VECTOR MIGRATION — Per-Field Embedding")
    _print_separator()
    if not args.skip_index:
        create_new_index()
    else:
        print("Skipping index creation (--skip-index)")
    migrate_and_vectorize(max_docs=args.doc)
    print("\nMigration successful!")
