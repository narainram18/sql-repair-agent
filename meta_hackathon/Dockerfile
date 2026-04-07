# Dockerfile
# ----------
# Meta OpenEnv Hackathon — SQL Repair Environment
# Base: Python 3.10 slim for minimal footprint
# Exposes port 7860 (Hugging Face Spaces default)

FROM python:3.10-slim

# Metadata labels
LABEL maintainer="hackathon-submission"
LABEL description="DBOps Agent SQL Repair Environment — Meta OpenEnv Hackathon"
LABEL version="1.0.0"

# Prevent Python from writing .pyc files and enable unbuffered stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Install Python dependencies
# We keep this minimal: only FastAPI, uvicorn, pydantic, requests, openai
# sqlite3 is part of Python standard library — no extra install needed
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY sql_repair_env.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Switch to non-root user
USER appuser

# Expose Hugging Face Spaces default port
EXPOSE 7860

# Health check — hits the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
    || exit 1

# Launch the FastAPI server
CMD ["python", "-m", "uvicorn", "sql_repair_env:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]