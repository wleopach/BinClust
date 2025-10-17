# Dockerfile for the clustering pipeline
FROM python:3.11-slim

# Silence GitPython init warning if git is absent
ENV GIT_PYTHON_REFRESH=quiet

# System deps (some packages may be needed for scientific libs)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (leverage layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


# Create data and outputs root dirs (data typically mounted at runtime)
RUN mkdir -p /app/data /app/outputs

# Copy source code (including src/.env if present)
COPY src /app/src

# Default command: auto-detect a Parquet (preferred) or CSV in /app/data, use PCA + KMeans
ENTRYPOINT ["python", "/app/src/main.py"]