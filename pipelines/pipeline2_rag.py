import os
import pickle
import time
from pathlib import Path

from pipelines.utils import count_tokens, gemini_generate, make_result, setup_gemini

_ROOT     = Path(__file__).parent.parent.resolve()
_embedder = None
_index    = None
_chunks   = None
_client   = None
_load_error: str = ""   # set once if the index files are missing; cleared on success


def _index_paths():
    """Return (faiss_path, chunks_path) for the Basic RAG index.

    Default: the small demo index (rag_index_demo.faiss / chunks_demo.pkl, ~2 MB total) —
    committed to the repo, loads instantly, and returns the same top-k chunks for the demo
    questions as the full index, so answers + token counts are identical. No OOM, no
    download. Set BASIC_RAG_FULL=1 to use the full 464k-chunk index for local benchmarking.
    """
    use_full = os.getenv('BASIC_RAG_FULL', '').lower() in ('1', 'true', 'yes')
    if use_full:
        return (_ROOT / 'data/chunks/rag_index.faiss',
                _ROOT / 'data/chunks/chunks.pkl')
    demo_faiss = _ROOT / 'data/chunks/rag_index_demo.faiss'
    if demo_faiss.exists():
        return demo_faiss, _ROOT / 'data/chunks/chunks_demo.pkl'
    # Fall back to the full index if the demo artifact isn't present.
    return (_ROOT / 'data/chunks/rag_index.faiss',
            _ROOT / 'data/chunks/chunks.pkl')


def _load():
    global _embedder, _index, _chunks, _client, _load_error
    if _embedder is not None and _index is not None and _chunks is not None:
        return
    faiss_path, chunks_path = _index_paths()
    if not faiss_path.exists() or not chunks_path.exists():
        _load_error = (
            "Basic RAG index not found. Run `python scripts/build_faiss_demo.py` to "
            "generate data/chunks/rag_index_demo.faiss + chunks_demo.pkl, then commit them."
        )
        return
    try:
        import faiss
        from fastembed import TextEmbedding
        _embedder   = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        _index      = faiss.read_index(str(faiss_path))
        with open(str(chunks_path), 'rb') as f:
            _chunks = pickle.load(f)
        _client     = setup_gemini()
        _load_error = ""
    except Exception as e:
        _embedder = _index = _chunks = _client = None
        _load_error = str(e)


def _embed(text: str):
    import numpy as np
    emb  = list(_embedder.embed([text]))[0]
    emb  = np.array(emb, dtype=np.float32).reshape(1, -1)
    norm = (emb ** 2).sum(axis=1, keepdims=True) ** 0.5
    return emb / (norm + 1e-10)


def pipeline2(question: str, top_k: int = 8) -> dict:
    _load()
    if _load_error:
        answer  = f"Basic RAG unavailable: {_load_error}"
        result  = make_result('basic_rag', answer, 0, count_tokens(answer), 0.0)
        result['sources'] = []
        result['status']  = 'faiss_unavailable'
        return result
    emb = _embed(question)
    _, idxs = _index.search(emb, top_k)

    retrieved = [_chunks[i] for i in idxs[0] if i < len(_chunks)]
    context   = '\n\n---\n\n'.join(c['text'] for c in retrieved)
    sources   = [c.get('source', c.get('title', '')) for c in retrieved]

    prompt = (
        "You are an expert assistant. Use ONLY the context below to answer the question. "
        "Be thorough — extract every relevant fact, name, date, and detail present in the context. "
        "If the context covers multiple aspects of the question, address all of them. "
        "Do not say 'the context does not state' — if the answer is in the context, state it clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    start   = time.time()
    answer  = gemini_generate(_client, prompt, max_tokens=512)
    latency = round(time.time() - start, 3)

    p_tok  = count_tokens(prompt)
    c_tok  = count_tokens(answer)
    result = make_result('basic_rag', answer, p_tok, c_tok, latency)
    result['sources'] = sources
    return result
