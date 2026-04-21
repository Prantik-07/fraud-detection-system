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

# ── Expose FastAPI port ───────────────────────────────────────────────────────
EXPOSE 8000

# ── Run FastAPI backend ───────────────────────────────────────────────────────
# In production, the FastAPI app should be updated to mount the React build directory at /.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
