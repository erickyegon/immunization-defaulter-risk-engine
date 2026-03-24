FROM python:3.11-slim

LABEL maintainer="Dr. Erick Kiprotich Yegon <erickyegon@gmail.com>"
LABEL description="IZ Defaulter Prediction Model — XGBoost + SHAP"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed reports/shap mlruns

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run as non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default: run full training pipeline
CMD ["python", "main.py", "--stage", "all"]

# To serve API only (after training):
# docker run -p 8000:8000 iz-defaulter uvicorn src.api.main:app --host 0.0.0.0 --port 8000
