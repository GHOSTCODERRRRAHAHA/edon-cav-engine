# EDON SDK Structure

## Folder Tree

```
sdk/
  python/
    edon/                    # Main SDK package
      __init__.py           # Package initialization, exports
      client.py             # EdonClient class
      transport.py          # Transport abstraction
      rest_transport.py     # REST HTTP transport
      grpc_transport.py     # gRPC transport
      exceptions.py         # Exception classes
    setup.py                # Setuptools setup
    pyproject.toml          # Modern Python packaging
    README.md               # SDK documentation

examples/
  python/
    simple_call.py          # Basic REST example
    simple_grpc_example.py  # Basic gRPC example

docs/
  API_OVERVIEW.md           # API documentation
  OEM_INTEGRATION.md        # Integration guide

integrations/
  grpc/
    edon_grpc_service/
      server.py             # gRPC server implementation
      edon.proto            # Protocol buffer definitions
      generate_proto.sh      # Protobuf generation script
```

## Key Features

### ✅ Clean Separation
- SDK only calls APIs (REST or gRPC)
- No internal engine imports
- Transport layer abstraction

### ✅ Environment Variables
- `EDON_BASE_URL` - REST API base URL
- `EDON_API_TOKEN` - Authentication token

### ✅ Dual Transport Support
- REST (default) - HTTP-based
- gRPC (optional) - High-performance streaming

### ✅ Installable Package
```bash
pip install -e sdk/python
pip install -e "sdk/python[grpc]"
```

## Usage

```python
from edon import EdonClient, TransportType

# REST (default, uses EDON_BASE_URL env var)
client = EdonClient()

# gRPC
client = EdonClient(transport=TransportType.GRPC)

# Compute CAV
result = client.cav(window)
```

