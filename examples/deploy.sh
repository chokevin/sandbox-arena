#!/bin/bash
# Deploy and run the training job on AKS.
#
# Prerequisites:
#   - AKS cluster with agent-sandbox controller + extensions installed
#   - sandbox-router deployed
#   - ACR attached to the cluster
#
# Usage:
#   ./deploy.sh <acr-name>
#   Example: ./deploy.sh myregistry

set -e

ACR_NAME="${1:?Usage: ./deploy.sh <acr-name>}"
ACR_LOGIN_SERVER="${ACR_NAME}.azurecr.io"
IMAGE="${ACR_LOGIN_SERVER}/sandbox-arena/trainer:latest"

echo "=== Building and pushing trainer image ==="
az acr build --registry "${ACR_NAME}" --image sandbox-arena/trainer:latest .

echo "=== Applying sandbox configs ==="
kubectl apply -f sandbox/sandbox-template.yaml
kubectl apply -f sandbox/warm-pool.yaml

echo "=== Deploying training job ==="
# Delete previous job if exists
kubectl delete job sandbox-arena-training --ignore-not-found

# Apply job with image substituted
sed "s|\${TRAINING_IMAGE}|${IMAGE}|g" sandbox/job.yaml | kubectl apply -f -

echo "=== Job submitted. Follow logs: ==="
echo "  kubectl logs -f job/sandbox-arena-training"
