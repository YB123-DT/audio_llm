#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
module_c_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=6 "$module_c_python" scripts/extract_module_c_states.py \
 --manifest artifacts/ravdess_happy_sad_intensity01.csv \
 --ravdess-root /data2/yb/paper/RAVDESS \
 --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
 --module-b-vectors reports/2026-09-16-module-b/decoder_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --output reports/2026-09-16-module-c/sublayer_states.npz \
 --device cuda:0
"$module_c_python" scripts/analyze_module_c.py \
 --representations reports/2026-09-16-module-c/sublayer_states.npz \
 --reference-states reports/2026-09-16-module-b/decoder_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --cached-predictions reports/2026-09-16-answer-content/oof_predictions.csv \
 --output-dir reports/2026-09-16-module-c/analysis \
 --bootstrap 10000
