# EDON gRPC Service

gRPC microservice for real-time EDON CAV computation.

## Setup

1. Install dependencies:
```bash
pip install grpcio grpcio-tools
```

2. Generate protobuf code:
```bash
cd integrations/grpc/edon_grpc_service
chmod +x generate_proto.sh
./generate_proto.sh
```

Or manually:
```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto
```

3. Run the server:
```bash
python edon_grpc_server.py --port 50051
```

## API

### GetState (Single Request/Response)

Single CAV computation request.

### StreamState (Server-Side Streaming)

Continuous state updates pushed from server.

## Protocol Buffer Definition

See `edon.proto` for message definitions.

