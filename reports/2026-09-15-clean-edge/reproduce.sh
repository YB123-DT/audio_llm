#!/usr/bin/env bash
set -euo pipefail
# Repository root; existing frozen-model runtime. Clean stage runs first.
args=(--manifest artifacts/ravdess_happy_sad_intensity01.csv
 --ravdess-root /data2/yb/paper/RAVDESS --slam-llm-root /data2/yb/paper/SLAM-LLM
 --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B
 --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt
 --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt --device cuda:0 --seed 1234)
python scripts/run_clean_edge_gain.py "${args[@]}" --output-csv reports/2026-09-15-clean-edge/gain.csv
python scripts/analyze_clean_edge_gain.py --input-csv reports/2026-09-15-clean-edge/gain.csv --reference-csv reports/2026-09-15-audio-edge/effects.csv --output-dir reports/2026-09-15-clean-edge/gain_analysis
python scripts/run_block_direct_edge_restore.py "${args[@]}" --output-csv reports/2026-09-15-clean-edge/block_restore.csv
python scripts/analyze_block_direct_edge.py --input-csv reports/2026-09-15-clean-edge/block_restore.csv --reference-csv reports/2026-09-15-audio-edge/effects.csv --output-dir reports/2026-09-15-clean-edge/block_analysis
