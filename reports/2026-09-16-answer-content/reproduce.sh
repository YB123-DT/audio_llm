#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
content_probe_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$content_probe_python" scripts/analyze_answer_content.py \
 --representations reports/2026-09-16-module-b/decoder_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --cross-predictions reports/2026-09-16-module-b/analysis/oof_predictions.csv \
 --cross-summary reports/2026-09-16-module-b/analysis/summary.csv \
 --output-dir reports/2026-09-16-answer-content \
 --bootstrap 10000
