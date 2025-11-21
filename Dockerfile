# EDON CAV Engine - Production Dockerfile
# Supports both REST API and gRPC service

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gRPC dependencies
RUN pip install --no-cache-dir grpcio grpcio-tools

# Copy application code
COPY . /app

# Install EDON SDK (for examples)
RUN pip install --no-cache-dir -e ./sdk/python

# Generate protobuf files
RUN cd integrations/grpc/edon_grpc_service && \
    python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto

# Set environment variables (no secrets in image)
ENV EDON_AUTH_ENABLED=false \
    EDON_BASE_URL=http://0.0.0.0:8000 \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)" || exit 1

# Start both REST API and gRPC server
CMD ["bash", "-c", "\
    echo 'Starting EDON CAV Engine...' && \
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 & \
    sleep 2 && \
    python -m integrations.grpc.edon_grpc_service.server --port 50051 & \
    wait \
"]
