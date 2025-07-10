# MSC Framework v4.0 - Multi-stage Dockerfile

# Stage 1: Base image with Python
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as app

# Create non-root user
RUN useradd -m -u 1000 mscuser && \
    mkdir -p /app/data /app/checkpoints /app/logs && \
    chown -R mscuser:mscuser /app

# Copy application code
COPY --chown=mscuser:mscuser . .

# Switch to non-root user
USER mscuser

# Create necessary directories
RUN mkdir -p data/checkpoints data/logs static

# Expose ports
EXPOSE 5000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/system/health || exit 1

# Set default environment variables
ENV MSC_DATA_DIR=/app/data \
    API_HOST=0.0.0.0 \
    API_PORT=5000

# Default command
CMD ["python", "msc_framework_enhanced.py", "--mode", "both"]

# === Alternative: Production image with Gunicorn ===
# FROM app as production
# 
# # Install production server
# RUN pip install gunicorn[eventlet]
# 
# # Copy gunicorn config
# COPY gunicorn.conf.py .
# 
# # Run with gunicorn
# CMD ["gunicorn", "--config", "gunicorn.conf.py", "msc_framework_enhanced:app"]