# ── Build React Frontend ──────────────────────────────────────────────────────
FROM node:18-alpine AS frontend-builder
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm install
COPY client/ ./
RUN npm run build

# ── Base image for Backend ────────────────────────────────────────────────────
FROM python:3.10-slim
WORKDIR /app

# ── Install system deps ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir fastapi uvicorn

# ── Copy project files ────────────────────────────────────────────────────────
COPY . .

# ── Copy built React app to a public directory in API ──────────────────────────
COPY --from=frontend-builder /app/client/dist /app/client/dist

# ── Expose FastAPI port (Hugging Face Spaces uses 7860) ────────────────────────
EXPOSE 7860

# ── Setup User for HF Spaces ──────────────────────────────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app
COPY --from=frontend-builder --chown=user /app/client/dist $HOME/app/client/dist

# ── Run FastAPI backend ───────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
