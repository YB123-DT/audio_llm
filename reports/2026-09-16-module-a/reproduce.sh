#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
module_a_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
"$module_a_python" scripts/analyze_module_a_baseline.py \
 --legacy-hidden /data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden/representations.pt \
 --edge-vectors reports/2026-09-15-edge-representation/edge_vectors.npz \
 --model-config /data2/yb/paper_runtime_models/Qwen2-0.5B/config.json \
 --representations reports/2026-09-16-module-a/baseline_vectors.npz \
 --metadata /data2/yb/audio_llm_runs/ravdess_happy_sad_intensity01_systemprompt/hidden/metadata.csv \
 --output-dir reports/2026-09-16-module-a/baseline_analysis
CUDA_VISIBLE_DEVICES=6 "$module_a_python" scripts/run_module_a_interface_control.py \
 --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
 --output reports/2026-09-16-module-a/interface-control.json --device cuda:0
