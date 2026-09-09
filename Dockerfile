FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the fastembed ONNX model so it's baked into the image
# and never downloaded at runtime (avoids 60-90s delay on first request)
RUN python -c "from fastembed import TextEmbedding; list(TextEmbedding('sentence-transformers/all-MiniLM-L6-v2').embed(['warmup'])); print('fastembed model cached')"

COPY . .

# The small demo FAISS index (data/chunks/rag_index_demo.faiss + chunks_demo.pkl, ~2 MB)
# is committed to the repo and copied in above — Basic RAG loads it instantly, no download.

ENV PYTHONUNBUFFERED=1

EXPOSE 8080
CMD uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8080}
