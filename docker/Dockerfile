FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ───────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Project code ─────────────────────────────────────────────
COPY . /app/buzzspot
WORKDIR /app/buzzspot

# ── Model weights (copy from build context or download) ──────
# Uncomment if weights are bundled with the Docker image:
# COPY weights/ /app/buzzspot/weights/

# ── Default command: run inference ──────────────────────────
ENTRYPOINT ["python", "docker/entrypoint.py"]
