# EDON Docker Deployment

One-command deployment for OEM robotics teams.

## Quick Start

```bash
git clone <repo>
cd edon-cav-engine
docker compose up --build
```

**That's it!** EDON is now running:
- REST API: `http://localhost:8001`
- gRPC: `localhost:50051`

## Verify It's Running

```bash
# Health check
curl http://localhost:8001/health

# Test with Python SDK
python -c "from edon import EdonClient; print(EdonClient().health())"
```

## Configuration

Edit `docker-compose.yml` to customize:

```yaml
services:
  edon:
    environment:
      EDON_AUTH_ENABLED: "true"        # Enable authentication
      EDON_API_TOKEN: "your-secret"    # Set your token
      EDON_BASE_URL: "http://0.0.0.0:8001"
    ports:
      - "8001:8001"   # REST API
      - "50051:50051" # gRPC
```

## Production Deployment

For production, set:
- `EDON_AUTH_ENABLED=true`
- Strong `EDON_API_TOKEN`
- Use reverse proxy (nginx/traefik) for HTTPS
- Mount model files as volumes if external

## Troubleshooting

**Port already in use?**
```bash
# Change ports in docker-compose.yml
ports:
  - "8002:8001"   # Use different host port
  - "50052:50051"
```

**Models not found?**
- Ensure model files are in `data/raw/wesad/` or set `EDON_MODEL_DIR` env var

**gRPC not working?**
- Check that protobuf files are generated (done automatically in Dockerfile)
- Verify port 50051 is accessible

## Manual Build

```bash
docker build -t edon-cav-engine .
docker run -p 8001:8001 -p 50051:50051 edon-cav-engine
```

