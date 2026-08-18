import os
from elasticsearch import Elasticsearch, helpers
from sentence_transformers import SentenceTransformer
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
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIMS = int(os.getenv("EMBED_DIMS", "384"))
SOURCE_INDEX = "fatboy_data"
NEW_INDEX = "vec_chat_fatboy_data"

# Fields to combine and vectorize for the chatbot context.
# Built from ALL fields defined across TYPE_MAPPING (field_mapping.py),
# paired with their human-readable labels for better retrieval context.
FIELD_LABEL_PAIRS = []
_seen_fields = set()
for _mapping in TYPE_MAPPING.values():
    for _field, _label in zip(_mapping["fields"], _mapping["field_labels"]):
        if _field not in _seen_fields:
            _seen_fields.add(_field)
            FIELD_LABEL_PAIRS.append((_field, _label))

VECTOR_SOURCE_FIELDS = [field for field, _ in FIELD_LABEL_PAIRS]
VECTOR_FIELD_NAME = "combined_text_vector"
COMBINED_TEXT_FIELD = "combined_text"

# Initialize clients
# Basic auth is used when ES_USER/ES_PASS are set in .env; cert verification is
# disabled by default for self-signed HTTPS (enable via ES_VERIFY_CERTS=true).
es = Elasticsearch(
    ELASTIC_URL,
    basic_auth=(ES_USER, ES_PASS) if ES_USER else None,
    verify_certs=ES_VERIFY_CERTS,
)
model = SentenceTransformer(EMBED_MODEL)

# 2. Define New Index Mapping (Dynamic template keeps original fields, adds vector)
index_mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 1
    },
    "mappings": {
        "properties": {
            VECTOR_FIELD_NAME: {
                "type": "dense_vector",
                "dims": EMBED_DIMS,
                "index": True,
                "similarity": "cosine"
            },
            COMBINED_TEXT_FIELD: {
                "type": "text"
            }
        }
    }
}

def create_new_index():
    if es.indices.exists(index=NEW_INDEX):
        es.indices.delete(index=NEW_INDEX)
    es.indices.create(index=NEW_INDEX, body=index_mapping)
    print(f"Created new index: {NEW_INDEX}")

def extract_text_for_vector(source_doc):
    """Extracts and concatenates labelled values from all fields defined in field_mapping.py."""
    text_parts = []
    for field, label in FIELD_LABEL_PAIRS:
        val = source_doc.get(field)
        if val:
            text_parts.append(f"{label}: {val}")
    return " ".join(text_parts)

def migrate_and_vectorize():
    # Scan all documents from the source index (handles pagination automatically)
    query = {"query": {"match_all": {}}}
    scan_iterator = helpers.scan(es, query=query, index=SOURCE_INDEX, scroll="5m", size=100)

    batch = []
    batch_size = 100
    total_indexed = 0

    print(f"Starting document scanning and vectorization from {SOURCE_INDEX}...")
    for doc in scan_iterator:
        source_data = doc["_source"]
        doc_id = doc["_id"]

        # Combine labelled fields into a single text block for the vector embedding
        combined_text = extract_text_for_vector(source_data)

        if combined_text.strip():
            # Generate vector embedding
            embedding = model.encode(combined_text).tolist()
        else:
            # If no target text exists, fill with zeros to keep mapping valid
            embedding = [0.0] * EMBED_DIMS

        # Keep all original fields and inject the vector + readable text
        source_data[VECTOR_FIELD_NAME] = embedding
        source_data[COMBINED_TEXT_FIELD] = combined_text.strip()

        # Prepare bulk index action
        action = {
            "_index": NEW_INDEX,
            "_id": doc_id,
            "_source": source_data
        }
        batch.append(action)

        if len(batch) >= batch_size:
            helpers.bulk(es, batch)
            total_indexed += len(batch)
            print(f"Indexed batch of {len(batch)} documents.")
            batch = []

    # Index remaining documents
    if batch:
        helpers.bulk(es, batch)
        total_indexed += len(batch)
        print(f"Indexed final batch of {len(batch)} documents.")

    print(f"Total documents indexed into {NEW_INDEX}: {total_indexed}")

if __name__ == "__main__":
    create_new_index()
    migrate_and_vectorize()
    print("Migration successful!")
