import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from pipelines.pipeline1_llm import pipeline1
from pipelines.pipeline2_rag import pipeline2
from pipelines.pipeline3_graphrag import pipeline3
from eval.judge import llm_judge_with_source, compute_bertscore


# ── Startup: preload models in background so first request is fast ─────────────
_preload_done  = False
_preload_error = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _preload_done, _preload_error
    def _preload():
        global _preload_done, _preload_error
        try:
            from pipelines.pipeline2_rag import _load as load2
            from pipelines.pipeline3_graphrag import _load as load3, _ensure_entity_cache
            load2()
            load3()
            _ensure_entity_cache()   # pre-warm TigerGraph entity ID cache
            # BERTScore (distilbert + torch) is NOT preloaded — it costs ~300MB RAM
            # which exceeds the free-tier 512MB limit. It loads lazily on first eval request.
            _preload_done = True
            print("INFO:     Models preloaded successfully", flush=True)
        except Exception as e:
            _preload_error = str(e)
            print(f"WARNING:  Preload failed: {e}", flush=True)
    threading.Thread(target=_preload, daemon=True).start()

    # Keep the TigerGraph Savanna workspace warm while this Space is running, so a
    # visitor's first GraphRAG query doesn't hit a cold/idle graph. (Does not wake a
    # fully-suspended workspace — that needs auto-start enabled in the TG console.)
    def _keepalive():
        import requests as _rq
        host = os.environ.get("TG_HOST", "")
        while host:
            try:
                _rq.get(f"{host}/api/ping", timeout=(3, 6))
            except Exception:
                pass
            time.sleep(240)
    threading.Thread(target=_keepalive, daemon=True).start()
    yield


# ── Thread pools ──────────────────────────────────────────────────────────────
# Pipeline pool: one worker per pipeline (3 concurrent).
# Eval pool: judge × 3 + BERTScore × 1 all run in parallel after pipelines finish.
_pipeline_executor = ThreadPoolExecutor(max_workers=3)
_eval_executor     = ThreadPoolExecutor(max_workers=4)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question:     str
    ground_truth: str = ""


@app.post("/compare")
def compare(req: QueryRequest):
    """Run all three pipelines in parallel and return a combined comparison result."""
    futures = {
        'llm_only':  _pipeline_executor.submit(pipeline1, req.question),
        'basic_rag': _pipeline_executor.submit(pipeline2, req.question),
        'graphrag':  _pipeline_executor.submit(pipeline3, req.question),
    }

    results: dict = {}
    for name, future in futures.items():
        try:
            results[name] = future.result(timeout=90)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"detail": f"{name} pipeline failed: {e}"},
            )

    p1, p2, p3 = results['llm_only'], results['basic_rag'], results['graphrag']

    basic_rag_ok = p2.get('status') != 'faiss_unavailable'

    # GraphRAG only counts as a genuine result when it actually retrieved graph context
    # and produced an answer from it. A retrieval miss returns a ~9-token sentinel — counting
    # that as a "reduction" would fabricate a 99%+ win, so we report it honestly instead.
    graphrag_ok = bool(p3.get('graph_context_found')) and p3.get('context_tokens', 0) > 0
    graphrag_status = p3.get('status', 'ok' if graphrag_ok else 'no_context')

    if graphrag_ok and basic_rag_ok:
        # Token reduction: GraphRAG vs Basic RAG
        token_reduction = round(
            (1 - p3['total_tokens'] / max(p2['total_tokens'], 1)) * 100, 1
        )
        # Cost reduction: derived from actual cost_usd values (not a token proxy)
        cost_reduction = round(
            (1 - p3['cost_usd'] / max(p2['cost_usd'], 1e-9)) * 100, 1
        )
    else:
        # No graph context, or FAISS unavailable → no genuine comparison to report.
        token_reduction = None
        cost_reduction = None

    result = {
        'llm_only':            p1,
        'basic_rag':           p2,
        'graphrag':            p3,
        'graphrag_status':     graphrag_status,
        'token_reduction_pct': token_reduction,
        'cost_reduction_pct':  cost_reduction,
    }

    if req.ground_truth:
        # Run all 4 eval tasks in parallel — previously sequential, adding ~8s total.
        eval_futures = {
            'judge_llm_only':  _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p1['answer']),
            'judge_basic_rag': _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p2['answer']),
            'judge_graphrag':  _eval_executor.submit(llm_judge_with_source, req.question, req.ground_truth, p3['answer']),
            'bertscore':       _eval_executor.submit(compute_bertscore, [p3['answer']], [req.ground_truth]),
        }
        for key, fut in eval_futures.items():
            try:
                value = fut.result(timeout=60)
                # judge_* futures return (verdict, source); bertscore returns a dict as-is.
                if key.startswith('judge_'):
                    result[key], result[f'{key}_source'] = value
                else:
                    result[key] = value
            except Exception:
                # Honest failure -- no key means "judge did not run", not "it passed".
                if key.startswith('judge_'):
                    result[key], result[f'{key}_source'] = 'ERROR', 'error'
                else:
                    result[key] = 'ERROR'

        # If GraphRAG had no graph context, it produced no grounded answer and must be
        # judged like anything else -- a no-context miss should show up as a FAIL in
        # the reported pass rate, not be silently excused.

    return result


@app.get("/")
def root():
    return {"status": "ok", "message": "GraphRAG API — POST /compare"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Returns whether the model preload has finished. Poll this before the first query."""
    return {
        "ready": _preload_done,
        "error": _preload_error,
    }


@app.get("/debug")
def debug():
    """Returns which components are loaded — useful for diagnosing startup issues."""
    from pipelines.pipeline2_rag import _embedder, _index
    return {
        "embedder_loaded": _embedder is not None,
        "faiss_loaded":    _index is not None,
        "preload_done":    _preload_done,
        "preload_error":   _preload_error,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False)
