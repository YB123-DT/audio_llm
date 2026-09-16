#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
crema_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
crema_report=reports/2026-09-16-crema-confirmation
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Frozen upstream revision and manifest are enforced by preparation script.
"$crema_python" scripts/prepare_crema_confirmation.py \
 --output-dir "$crema_report" --audio-root /data2/yb/paper/CREMA-D --download
CUDA_VISIBLE_DEVICES=6 "$crema_python" scripts/run_crema_confirmation.py \
 --manifest "$crema_report/manifest.csv" \
 --audio-checksums "$crema_report/audio_checksums.csv" \
 --audio-root /data2/yb/paper/CREMA-D \
 --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
 --output "$crema_report/intervention_states.npz" --device cuda:0 --resume
"$crema_python" scripts/analyze_crema_confirmation.py \
 --representations "$crema_report/intervention_states.npz" \
 --metadata "$crema_report/manifest.csv" --output-dir "$crema_report/analysis" --bootstrap 10000
