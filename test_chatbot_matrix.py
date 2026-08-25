"""Batch-test the RAG chatbot across field-type x question-style combinations.

Usage (with the app running on APP_HOST:APP_PORT):
    python test_chatbot_matrix.py

Requires the Flask app to be up; reads .env for APP_PORT if present.

Each case carries a `checks` list of (description, callable(response) -> bool).
A case FAILS when any check returns False or the response is an error string.
Checks are heuristic (substring / regex) — they catch regressions in routing,
retrieval strategy and answer quality, not exact wording.
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
            return (data.get("response", ""), data.get("turn_id", ""),
                    round(time.time() - t0, 1))
    except Exception as e:  # noqa: BLE001 - report any failure as a case result
        return f"[ERROR] {e}", "", round(time.time() - t0, 1)


def fetch_debug(cookie, turn_id):
    if not turn_id:
        return None
    req = urllib.request.Request(BASE + "/retrieval-debug/" + turn_id,
                                 headers={"Cookie": cookie}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


# ---- reusable check helpers -------------------------------------------------

def contains(*terms):
    """All terms (case-insensitive) must appear in the response."""
    terms_l = [t.lower() for t in terms]
    def check(resp):
        low = resp.lower()
        return all(t in low for t in terms_l)
    return check


def contains_any(*terms):
    terms_l = [t.lower() for t in terms]
    def check(resp):
        low = resp.lower()
        return any(t in low for t in terms_l)
    return check


def not_contains(*terms):
    terms_l = [t.lower() for t in terms]
    def check(resp):
        low = resp.lower()
        return not any(t in low for t in terms_l)
    return check


def mentions_id(doc_id):
    """Response must cite the given stable document id."""
    return contains(doc_id)


def used_query_named(name_part):
    """Retrieval debug must include a query whose name contains name_part."""
    def check(resp, debug):
        if not debug:
            return False
        return any(name_part.lower() in q["name"].lower() for q in debug.get("queries", []))
    return check


def debug_records_at_least(n):
    def check(resp, debug):
        if not debug:
            return False
        recs = [r for q in debug.get("queries", []) for r in (q.get("records") or [])]
        return len(recs) >= n
    return check


def no_records_check(resp, debug):
    """Debug endpoint must exist and be well-formed (even when 0 records)."""
    return isinstance(debug, dict) and "queries" in debug


# Question templates x entity values drawn from real index data.
# Tuple: (label, question, [check, ...]) — checks take (response) or
# (response, debug) depending on their arity (detected at run time).
CASES = [
    # --- UI / API plumbing ---
    ("turn_id returned", "hello, what can you do?", [no_records_check]),
    # --- location field ---
    ("location list",        "show me all locations",
     [used_query_named("aggregation"), contains_any("location")]),
    ("location detail typo", "give details of all records with location as afganistan",
     [contains_any("afghanistan"), not_contains("no information", "cannot find")]),
    ("location detail ok",   "show details from location hyderabad",
     [contains_any("hyderabad")]),
    ("location count",       "how many records for location afghanistan",
     [contains_any("afghanistan")]),
    ("location date combo",  "what happened on 2026-04-18 in location afghanistan",
     [contains_any("afghanistan", "2026-04-18")]),
    # --- bare location mention (no 'location' keyword) ---
    ("location bare mention", "What latest from location Ranchi",
     [used_query_named("entity"), contains_any("ranchi")]),
    ("location bare mention 2", "what latest information you have from Guwahati",
     [contains_any("guwahati")]),
    # --- equipment ---
    ("equipment types",      "what types of equipment are in the database",
     [contains_any("equipment")]),
    ("equipment detail",     "show me records with equipment AH-64 Apache",
     [contains_any("apache")]),
    ("equipment list",       "list all equipment names",
     [used_query_named("aggregation")]),
    ("equipment keyword",    "latest information with mention of grenade",
     [contains_any("grenade")]),
    # --- infra ---
    ("infra types",          "what types of infrastructure are in the database",
     [contains_any("infra")]),
    ("infra detail",         "show records with infra type airfields",
     [contains_any("airfield")]),
    # --- formation/orbat ---
    ("formation types",      "which enemy formations are tracked",
     [contains_any("formation")]),
    ("formation list",       "list all orbat titles",
     [used_query_named("aggregation")]),
    # --- training ---
    ("training types",       "what training types are recorded",
     [contains_any("training")]),
    ("training detail",      "show records with training type combat",
     [contains_any("combat", "training")]),
    # --- radar/elint ---
    ("radar fields",         "list all radar types",
     [contains_any("radar")]),
    # --- persons ---
    ("person fields",        "which person names appear in the database",
     [contains_any("person")]),
    # --- temporal: explicit ---
    ("date exact",           "what happened on 2026-04-18 in afghanistan",
     [contains_any("2026-04-18", "afghanistan")]),
    ("month summary",        "summary for month of april 2026",
     [contains_any("april", "2026")]),
    ("year summary",         "summarise records from year 2026",
     [contains_any("2026")]),
    ("year 2023",            "summarise records from year 2023",
     [contains_any("2023")]),
    # --- temporal: relative ranges (resolved against today) ---
    ("current month",        "give me a summary of current month data",
     [used_query_named("relative date range"),
      contains_any("current month", "august")]),
    ("last month",           "give me a summary of last month data",
     [used_query_named("relative date range"), contains_any("july")]),
    ("last 6 months",        "give me a summary of last 6 month data",
     [used_query_named("relative date range")]),
    ("last n days",          "show records from last 14 days",
     [used_query_named("relative date range")]),
    ("last n weeks",         "show records from last 6 weeks",
     [used_query_named("relative date range")]),
    ("last n months",        "summary of last 3 months data",
     [used_query_named("relative date range")]),
    ("last n years",         "summary of last 2 years data",
     [used_query_named("relative date range")]),
    ("last n hours",         "show records from last 48 hours",
     [used_query_named("relative date range")]),
    ("this year",            "summarize records from this year",
     [used_query_named("relative date range")]),
    ("last year",            "summarize records from last year",
     [used_query_named("relative date range")]),
    # --- counting ---
    ("db count",             "how many records you have",
     [contains_any("49", "total records", "database")]),
    # --- analysis quality ---
    ("analysis synthesis",   "give details on current month records",
     [contains_any("key observation", "observation", "analysis")]),
    # --- prediction grounding ---
    ("prediction grounded",  "analyze all records from location lyari and predict future outcome based on it",
     [contains_any("lyari"),
      contains_any("prediction", "possible outcome", "cannot", "no meaningful pattern")]),
    # --- generic kNN fallback ---
    ("semantic query",       "tell me about enemy armoured vehicle movements",
     [debug_records_at_least(1)]),
    ("nonsense",             "xyzzy plugh quantum banana", []),
]


def _arity(check):
    return 2 if getattr(check, "__code__", None) and check.__code__.co_argcount == 2 else 1


def main():
    try:
        cookie = get_session_cookie()
    except Exception as e:
        print(f"Cannot reach {BASE} — is the app running? ({e})")
        sys.exit(2)

    results = []
    for case in CASES:
        label, question = case[0], case[1]
        checks = case[2] if len(case) > 2 else []
        response, turn_id, secs = ask(cookie, question)
        first_line = response.replace("\n", " ")[:160]
        bad = (
            "[ERROR]" in response
            or response.startswith("Failed to get a response")
            or "Error communicating" in response
        )
        failed_checks = []
        if not bad:
            debug = fetch_debug(cookie, turn_id)
            for chk in checks:
                try:
                    if _arity(chk) == 2:
                        ok = chk(response, debug)
                    else:
                        ok = chk(response)
                except Exception as e:  # a broken check counts as a failure
                    ok = False
                    failed_checks.append(f"{getattr(chk, '__name__', 'check')}({e})")
                    continue
                if not ok:
                    failed_checks.append(getattr(chk, "__name__", str(chk))[:60])
        if failed_checks:
            bad = True
        results.append((label, question, bad, secs, first_line, failed_checks))
        flag = "FAIL" if bad else "ok  "
        detail = (" | failed: " + "; ".join(failed_checks)) if failed_checks else ""
        print(f"{flag} | {label:26} | {secs:>5}s | {first_line}{detail}")
        sys.stdout.flush()

    fails = [r for r in results if r[2]]
    print(f"\n=== {len(results) - len(fails)}/{len(results)} passed, {len(fails)} flagged ===")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
