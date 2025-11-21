#!/bin/bash
# Generate Python code from .proto file

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Generate protobuf and gRPC code
python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    edon.proto

echo "Generated edon_pb2.py and edon_pb2_grpc.py"
