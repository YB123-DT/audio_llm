#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
natural_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
CUDA_VISIBLE_DEVICES=6 "$natural_python" scripts/run_natural_edge_centering.py \
 --manifest artifacts/ravdess_happy_sad_intensity01.csv \
 --ravdess-root /data2/yb/paper/RAVDESS --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt --device cuda:0 --seed 1234 \
 --output-csv reports/2026-09-15-natural-edge-centering/scores.csv
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$natural_python" scripts/analyze_natural_edge_centering.py \
 --input-csv reports/2026-09-15-natural-edge-centering/scores.csv \
 --reference-csv reports/2026-09-15-audio-edge/effects.csv \
 --output-dir reports/2026-09-15-natural-edge-centering/analysis
