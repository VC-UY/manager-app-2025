# ─── Lightweight Python runtime ─────────────────────────────────────────────
# Using python:3.12-slim (~50 MB) instead of ubuntu:noble (~80 MB + manual
# Python install) avoids the 3–5 GB image trap. No CUDA/cuDNN/NVIDIA libs —
# MNIST runs fine on CPU-only PyTorch.
FROM python:3.12-slim

LABEL description="Distributed Learning Volunteer Node — Frugal distributed ML on volunteer machines"
LABEL maintainer="your-username"

# Avoid Python output buffering so Docker logs appear immediately
ENV PYTHONUNBUFFERED=1

# ─── System dependencies ────────────────────────────────────────────────────
# build-essential provides gcc/g++ for Python C extensions (psutil, etc.).
# --no-install-recommends keeps the layer small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies (cached layer) ─────────────────────────────────────
# Placed BEFORE the source copy so that a code change in src/ or volunteer.py
# does NOT reinstall PyTorch. Volunteers pulling an updated image download only
# the changed layer (a few KB), not the full dependency stack (hundreds of MB).
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=600 --retries=10 -r requirements.txt

# ─── Data (cached layer) ─────────────────────────────────────────────────────
# Placed before application code so that a code change in src/ or volunteer.py
# does NOT invalidate the data layer. MNIST / CIFAR10 change rarely — volunteers
# pulling an updated image download only the code layer (a few KB).
COPY data/ ./data/

# ─── Application code ───────────────────────────────────────────────────────
# Selective copy — only what the volunteer node needs at runtime.
# coordinator.py, manager.py, launch_experiment.py, and deploy.sh are excluded.
COPY src/ ./src/
COPY volunteer.py .

# ─── Runtime directories ────────────────────────────────────────────────────
# Created at build time; docker-compose mounts ./results over /app/results.
RUN mkdir -p /app/results /app/logs

# ─── Entrypoint ─────────────────────────────────────────────────────────────
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ─── Drop privileges ────────────────────────────────────────────────────────
# Run as non-root for security. The bind-mounted results directory on the host
# must be writable by the container's volunteer UID (typically 1000).
RUN useradd --no-create-home volunteer && \
    chown -R volunteer:volunteer /app

USER volunteer

ENTRYPOINT ["/entrypoint.sh"]
