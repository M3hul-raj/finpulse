FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# Install Python dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code, frontend, and precomputed artifacts
COPY src/ src/
COPY frontend/ frontend/
COPY model_results/ model_results/
COPY start.sh .
RUN chmod +x start.sh

# Pre-generate deterministic synthetic dataset at build time (no real customer data)
RUN python src/customer_generator.py --seed 42

# Run as non-root user
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", 10000)}/health')"

CMD ["bash", "start.sh"]
