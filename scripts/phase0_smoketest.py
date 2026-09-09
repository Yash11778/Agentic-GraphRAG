"""Phase 0 smoke test — confirms Gemini and TigerGraph credentials in .env actually work."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests


def test_gemini():
    print("── Gemini ──────────────────────────────")
    try:
        from pipelines.utils import setup_gemini, gemini_generate
        client = setup_gemini()
        answer = gemini_generate(client, "Reply with exactly: OK", max_tokens=10)
        print(f"  OK — response: {answer!r}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def test_tigergraph():
    print("── TigerGraph ──────────────────────────")
    host = os.environ.get("TG_HOST", "")
    secret = os.environ.get("TG_PASSWORD", "")
    graph = os.environ.get("TG_GRAPH", "MyDatabase")
    if not host or not secret:
        print("  FAILED: TG_HOST or TG_PASSWORD not set")
        return False
    try:
        resp = requests.get(f"{host}/api/ping", timeout=10)
        print(f"  ping status: {resp.status_code}")
    except Exception as e:
        print(f"  FAILED (ping): {e}")
        return False
    try:
        resp = requests.post(
            f"{host}/gsql/v1/tokens",
            json={"secret": secret, "graph": graph},
            timeout=10,
        )
        resp.raise_for_status()
        token = resp.json().get("token")
        print(f"  OK — token acquired for graph {graph!r}: {token[:8]}...")
        return True
    except Exception as e:
        print(f"  FAILED (token): {e}")
        return False


if __name__ == "__main__":
    g = test_gemini()
    t = test_tigergraph()
    print("──────────────────────────────────────")
    print(f"Gemini:     {'PASS' if g else 'FAIL'}")
    print(f"TigerGraph: {'PASS' if t else 'FAIL'}")
