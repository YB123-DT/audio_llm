#!/usr/bin/env bash
set -euo pipefail
# Run from repository root with the existing frozen-model runtime.
python scripts/run_source_value_restore.py \
  --manifest artifacts/ravdess_happy_sad_intensity01.csv \
  --ravdess-root /data2/yb/paper/RAVDESS \
  --slam-llm-root /data2/yb/paper/SLAM-LLM \
  --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B \
  --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt \
  --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt \
  --output-csv reports/2026-09-15-source-value/effects.csv \
  --device cuda:0 --seed 1234 --batch-size 1
python scripts/analyze_source_value_restore.py \
  --input-csv reports/2026-09-15-source-value/effects.csv \
  --reference-csv reports/2026-09-15-residual-component/qkv_effects.csv \
  --output-dir reports/2026-09-15-source-value/analysis
