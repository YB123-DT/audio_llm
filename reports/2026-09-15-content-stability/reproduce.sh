#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
content_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
args=(--representations reports/2026-09-15-edge-representation/edge_vectors.npz
 --metadata reports/2026-09-15-edge-representation/edge_vectors.csv
 --output-dir reports/2026-09-15-content-stability --bootstrap 10000)
"$content_python" scripts/analyze_edge_content_geometry.py "${args[@]}"
"$content_python" scripts/analyze_edge_content_counterfactual.py "${args[@]}" \
 --reference-oof reports/2026-09-15-edge-representation/probe_analysis/oof_predictions.csv
