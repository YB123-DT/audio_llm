#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
early_answer_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=6 "$early_answer_python" scripts/run_early_answer_ablation.py \
 --manifest artifacts/ravdess_happy_sad_intensity01.csv \
 --ravdess-root /data2/yb/paper/RAVDESS \
 --slam-llm-root /data2/yb/paper/SLAM-LLM \
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
 --module-b-vectors reports/2026-09-16-module-b/decoder_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --output reports/2026-09-16-early-answer-causal/intervention_states.npz \
 --device cuda:0
"$early_answer_python" scripts/analyze_early_answer_causal.py \
 --representations reports/2026-09-16-early-answer-causal/intervention_states.npz \
 --metadata reports/2026-09-16-module-b/metadata.csv \
 --clean-representations reports/2026-09-16-module-b/decoder_states.npz \
 --clean-predictions reports/2026-09-16-answer-content/oof_predictions.csv \
 --clean-analysis reports/2026-09-16-answer-content/analysis.json \
 --output-dir reports/2026-09-16-early-answer-causal/analysis \
 --bootstrap 10000
