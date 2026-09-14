#!/usr/bin/env bash
# Run from the repository root and retain build/test output for A5 debugging.
set -euo pipefail
task_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$task_repo"
task_log="mla_mxfp4_a5_$(date +%Y%m%d-%H%M%S).log"
bash operators/mla_prolog_v3_mxfp4/scripts/run_a5.sh "$@" 2>&1 | tee "$task_log"
