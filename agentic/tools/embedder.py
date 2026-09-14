"""Query embedding, shared by every pipeline that retrieves by similarity.

One process-wide instance, because loading the ONNX model costs a second or two
and a 150-question benchmark would otherwise pay that on every question.

The model here must be the one `build_vectors.py` used; if they ever diverge the
vectors are not comparable and similarity search returns confident nonsense, so
the model name lives in `graph_schema.py` and is read, never retyped.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agentic.graph_schema import EMBEDDING_MODEL

_model = None


def embed_query(text: str) -> list[float]:
    """Embed one query string into the corpus's vector space."""
    return embed_queries([text])[0]


def embed_queries(texts: list[str]) -> list[list[float]]:
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return [vector.tolist() for vector in _model.embed(texts)]
