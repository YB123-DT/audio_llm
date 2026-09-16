#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
module_b_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=6 "$module_b_python" scripts/extract_module_b_states.py \
 --manifest artifacts/ravdess_happy_sad_intensity01.csv \
 --ravdess-root /data2/yb/paper/RAVDESS \
 --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
 --output reports/2026-09-16-module-b/decoder_states.npz \
 --module-a-vectors reports/2026-09-16-module-a/baseline_vectors.npz \
 --metadata /data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden/metadata.csv \
 --device cuda:0
"$module_b_python" scripts/analyze_module_b.py \
 --representations reports/2026-09-16-module-b/decoder_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --module-a-summary reports/2026-09-16-module-a/baseline_analysis/summary.csv \
 --output-dir reports/2026-09-16-module-b/analysis \
 --bootstrap 10000
