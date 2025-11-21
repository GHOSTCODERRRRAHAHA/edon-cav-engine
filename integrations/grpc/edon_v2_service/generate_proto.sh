#!/bin/bash
# Generate Python protobuf files for v2 gRPC service

cd "$(dirname "$0")"

echo "Generating Python protobuf files for edon_v2.proto..."

python -m grpc_tools.protoc \
    -I. \
    --python_out=. \
    --grpc_python_out=. \
    edon_v2.proto

if [ $? -eq 0 ]; then
    echo "✓ Generated edon_v2_pb2.py and edon_v2_pb2_grpc.py"
else
    echo "✗ Failed to generate protobuf files"
    exit 1
fi

