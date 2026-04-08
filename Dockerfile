# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Metadata ─────────────────────────────────────────────────────────────────
LABEL maintainer="SpectraQual Team"
LABEL description="SpectraQual — PCB Quality Control OpenEnv Environment"
LABEL version="1.0.0"

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer cache) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy source code ──────────────────────────────────────────────────────────
COPY . .

# ── Environment variables (overridden at runtime) ─────────────────────────────
ENV API_BASE_URL="https://openrouter.ai/api/v1"
ENV MODEL_NAME="meta-llama/llama-3.3-70b-instruct"
ENV HF_TOKEN=""

# ── Expose Streamlit port (HF Spaces default) ─────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# ── Default command: launch FastAPI server ───────────────────────────────
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
