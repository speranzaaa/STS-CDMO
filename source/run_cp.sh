#!/bin/bash
export MSYS_NO_PATHCONV=1
IMAGE_NAME="sts_solver_group"

echo "Building Docker image..."
docker build -q -t $IMAGE_NAME . > /dev/null

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

HOST_DIR=$(pwd -W)
echo "Mounting: $HOST_DIR"

echo "--------------------------------"
echo "Running CP experiments ..."

docker run --rm \
    --ulimit stack=-1 \
    --memory-swap=-1 \
    -v "$HOST_DIR://sts_solver" \
    $IMAGE_NAME \
    python3 source/CP/run_cp.py "$@"

echo "Done."
