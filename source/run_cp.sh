#!/bin/bash
export MSYS_NO_PATHCONV=1
IMAGE_NAME="sts_solver_group"

echo "Building Docker image..."
docker build -q -t $IMAGE_NAME . > /dev/null

HOST_DIR=$(pwd -W)
echo "Mounting: $HOST_DIR"

echo "--------------------------------"
echo "Running CP experiments with UNLIMITED resources..."

docker run --rm \
    --ulimit stack=-1 \
    --memory-swap=-1 \
    -v "$HOST_DIR://sts_solver" \
    $IMAGE_NAME \
    python3 source/CP/run_cp.py

echo "Done."