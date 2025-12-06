# How to Start EDON Core Server for Testing

## Option 1: Build Docker Image (Recommended for Production)

### Build the Image

```powershell
# Build from Dockerfile
docker build -t edon-server:v1.0.1 .

# Then run it
docker run -d --name edon-server -p 8002:8000 -p 50052:50051 edon-server:v1.0.1
```

### Or Load from Tarball (If Available)

```powershell
# Load pre-built image
docker load -i release/edon-server-v1.0.1.tar

# Then run it
docker run -d --name edon-server -p 8002:8000 -p 50052:50051 edon-server:v1.0.1
```

---

## Option 2: Run Directly with Python (Easier for Testing)

### Start REST API Server

```powershell
# In one terminal
python -m uvicorn app.main:app --host 127.0.0.1 --port 8002
```

### Start gRPC Server (Optional, for gRPC testing)

```powershell
# In another terminal
python -m integrations.grpc.edon_grpc_service.server --port 50052
```

### Verify It's Running

```powershell
# Health check
curl http://127.0.0.1:8002/health

# Or in Python
python -c "import requests; print(requests.get('http://127.0.0.1:8002/health').json())"
```

---

## Quick Test Script

I'll create a simple script to start the server:

