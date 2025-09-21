#!/usr/bin/env bash
set -euo pipefail

# Build the Docker image for AMD64 and export it to a tar file.
# Usage: ./build-tar.sh [image_name] [tag] [tar_output]
# Defaults: image_name=tflitecompiler, tag=latest, tar_output=<image_name>_<tag>.tar

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTEXT_DIR="$SCRIPT_DIR"

IMAGE_NAME="${1:-tflitecompiler}"
TAG="${2:-latest}"
TAR_OUT="${3:-${IMAGE_NAME}_${TAG}.tar}"

# Allow overriding the platform via env var
PLATFORM="${PLATFORM:-linux/amd64}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH" >&2
  exit 1
fi

echo "Building ${IMAGE_NAME}:${TAG} for ${PLATFORM} from ${CONTEXT_DIR}..."

if docker buildx version >/dev/null 2>&1; then
  # Use buildx and load the image into the local docker engine
  docker buildx build --platform "${PLATFORM}" -t "${IMAGE_NAME}:${TAG}" --load "${CONTEXT_DIR}"
else
  # Fallback to classic docker build
  docker build --platform "${PLATFORM}" -t "${IMAGE_NAME}:${TAG}" "${CONTEXT_DIR}"
fi

echo "Saving image to ${TAR_OUT}..."
docker save -o "${TAR_OUT}" "${IMAGE_NAME}:${TAG}"

echo "Done. Image saved at ${TAR_OUT}"

