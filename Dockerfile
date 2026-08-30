# Multi-stage would save little here: the bulk is torch + model weights, which
# are needed at runtime. Instead we bake the models into the image so container
# start is fast and does not depend on the Hugging Face Hub being reachable.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only torch: the default wheel bundles CUDA and is ~5x larger for no benefit
# on a CPU host.
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple -r requirements-local.txt

COPY app/ ./app/
COPY ui/ ./ui/
COPY data/corpus/ ./data/corpus/
COPY scripts/ ./scripts/

# Bake the embedding + reranker weights into the image, then build the index at
# build time. The container therefore starts ready to serve instead of spending
# ~30s ingesting on every cold start.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && python -m app.cli ingest

# Hugging Face Spaces runs as a non-root user; make the cache writable.
RUN chmod -R 777 /app/.cache /app/data || true

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.api:app --host 0.0.0.0 --port ${PORT}"]
