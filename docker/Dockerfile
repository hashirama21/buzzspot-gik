FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    git wget unzip && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/buzzspot
WORKDIR /app/buzzspot

#  Model weights (copy from build context or download)
# Uncomment if weights are bundled with the Docker image:
# COPY weights/ /app/buzzspot/weights/

ENTRYPOINT ["python", "docker/entrypoint.py"]
