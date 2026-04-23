# ── Build React Frontend ──────────────────────────────────────────────────────
FROM node:18-alpine AS frontend-builder
WORKDIR /app/client
COPY client/package.json client/package-lock.json ./
RUN npm install
COPY client/ ./
RUN npm run build

# ── Base image for Backend ────────────────────────────────────────────────────
FROM python:3.10-slim

# Install system deps as root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Setup non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user fastapi uvicorn

# Copy project files
COPY --chown=user . .
COPY --from=frontend-builder --chown=user /app/client/dist ./client/dist

# Expose port
EXPOSE 7860

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
