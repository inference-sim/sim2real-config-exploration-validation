#!/bin/bash
# Run BLIS pipelineruns sequentially (they share the same model Helm release name).
set -e

CLUSTER_DIR="$(cd "$(dirname "$0")/cluster" && pwd)"
NS="jchen"

run_and_wait() {
    local name="$1"
    local yaml="$2"

    echo "========================================"
    echo "$(date +%H:%M:%S) Launching $name"
    echo "========================================"

    kubectl delete pipelinerun "$name" -n "$NS" 2>/dev/null || true
    kubectl apply -f "$yaml"

    echo "$(date +%H:%M:%S) Waiting for $name to complete..."
    while true; do
        status=$(kubectl get pipelinerun "$name" -n "$NS" -o jsonpath='{.status.conditions[0].status}' 2>/dev/null)
        reason=$(kubectl get pipelinerun "$name" -n "$NS" -o jsonpath='{.status.conditions[0].reason}' 2>/dev/null)

        if [ "$status" = "True" ]; then
            echo "$(date +%H:%M:%S) $name SUCCEEDED"
            break
        elif [ "$status" = "False" ]; then
            echo "$(date +%H:%M:%S) $name FAILED (reason: $reason)"
            kubectl get taskruns -n "$NS" -l "tekton.dev/pipelineRun=$name" --no-headers
            break
        fi
        sleep 30
    done
    echo ""
}

run_and_wait "blis1-spc-ce70b-cg" "$CLUSTER_DIR/pipelinerun-shared-prefix-codegen-blis1.yaml"
run_and_wait "blis2-spc-ce70b-cg" "$CLUSTER_DIR/pipelinerun-shared-prefix-codegen-blis2.yaml"
run_and_wait "blis3-spc-ce70b-cg" "$CLUSTER_DIR/pipelinerun-shared-prefix-codegen-blis3.yaml"

echo "========================================"
echo "$(date +%H:%M:%S) ALL BLIS RUNS COMPLETE"
echo "========================================"
kubectl get pipelineruns -n "$NS" --no-headers | grep "blis.*spc-ce70b"
