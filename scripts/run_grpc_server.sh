#!/bin/bash
# Run EDON gRPC Server

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Add project to Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Check if protobuf files are generated
GRPC_DIR="$PROJECT_ROOT/integrations/grpc/edon_grpc_service"
if [ ! -f "$GRPC_DIR/edon_pb2.py" ]; then
    echo "Generating protobuf files..."
    cd "$GRPC_DIR"
    if [ -f "generate_proto.sh" ]; then
        chmod +x generate_proto.sh
        ./generate_proto.sh
    else
        python3 -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. edon.proto
    fi
fi

# Run the server
cd "$PROJECT_ROOT"
python3 integrations/grpc/edon_grpc_service/edon_grpc_server.py "$@"

