"""Batch-test the RAG chatbot across field-type x question-style combinations.

Usage (with the app running on APP_HOST:APP_PORT):
    python test_chatbot_matrix.py

Requires the Flask app to be up; reads .env for APP_PORT if present.
"""
import json
import os
import re
import sys
import time
import urllib.request


def load_env(path=".env"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    # strip inline comments and quotes
                    value = value.split("#", 1)[0].strip().strip('"').strip("'")
                    os.environ.setdefault(key.strip(), value)


load_env(os.path.join(os.path.dirname(__file__), ".env"))

BASE = f"http://{os.getenv('APP_HOST', '127.0.0.1')}:{os.getenv('APP_PORT', '8999')}"
JAR_PATH = os.path.join(os.environ.get("TEMP", "/tmp"), "ck_matrix.txt")


def get_session_cookie():
    """Hit / once and capture the session cookie."""
    req = urllib.request.Request(BASE + "/", method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        set_cookie = resp.headers.get("Set-Cookie", "")
    m = re.match(r"([^=]+)=([^;]*)", set_cookie)
    return f"{m.group(1)}={m.group(2)}" if m else ""


def ask(cookie, message, timeout=300):
    body = json.dumps({"message": message}).encode()
    headers = {"Content-Type": "application/json", "Cookie": cookie}
    req = urllib.request.Request(BASE + "/chat", data=body, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("response", ""), round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001 - report any failure as a case result
        return f"[ERROR] {e}", round(time.time() - t0, 1)


# Question templates x entity values drawn from real index data.
CASES = [
    # --- location field ---
    ("location list",        "show me all locations"),
    ("location detail typo", "give details of all records with location as afganistan"),
    ("location detail ok",   "show details from location hyderabad"),
    ("location count",       "how many records for location afghanistan"),
    ("location date combo",  "what happened on 2026-04-18 in location afghanistan"),
    # --- equipment ---
    ("equipment types",      "what types of equipment are in the database"),
    ("equipment detail",     "show me records with equipment AH-64 Apache"),
    ("equipment list",       "list all equipment names"),
    # --- infra ---
    ("infra types",          "what types of infrastructure are in the database"),
    ("infra detail",         "show records with infra type airfields"),
    # --- formation/orbat ---
    ("formation types",      "which enemy formations are tracked"),
    ("formation list",       "list all orbat titles"),
    # --- training ---
    ("training types",       "what training types are recorded"),
    ("training detail",      "show records with training type combat"),
    # --- radar/elint ---
    ("radar fields",         "list all radar types"),
    # --- persons ---
    ("person fields",        "which person names appear in the database"),
    # --- temporal ---
    ("date exact",           "what happened on 2026-04-18 in afghanistan"),
    ("month summary",        "summary for month of april 2026"),
    ("year summary",         "summarise records from year 2026"),
    ("year 2023",            "summarise records from year 2023"),
    # --- counting ---
    ("db count",             "how many records you have"),
    # --- generic kNN fallback ---
    ("semantic query",       "tell me about enemy armoured vehicle movements"),
    ("nonsense",             "xyzzy plugh quantum banana"),
]


def main():
    try:
        cookie = get_session_cookie()
    except Exception as e:
        print(f"Cannot reach {BASE} — is the app running? ({e})")
        sys.exit(2)

    results = []
    for label, question in CASES:
        response, secs = ask(cookie, question)
        first_line = response.replace("\n", " ")[:160]
        bad = (
            "[ERROR]" in response
            or response.startswith("Failed to get a response")
            or "Error communicating" in response
        )
        results.append((label, question, bad, secs, first_line))
        print(f"{'FAIL' if bad else 'ok  '} | {label:22} | {secs:>5}s | {first_line}")
        sys.stdout.flush()

    fails = [r for r in results if r[2]]
    print(f"\n=== {len(results) - len(fails)}/{len(results)} passed, {len(fails)} flagged ===")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
