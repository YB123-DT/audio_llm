#!/usr/bin/env bash
set -euo pipefail
# Run from repository root in the existing SLAM-Omni runtime.
report_dir=reports/2026-09-15-residual-component
runtime_args=(
  --manifest artifacts/ravdess_happy_sad_intensity01.csv
  --ravdess-root /data2/yb/paper/RAVDESS
  --slam-llm-root /data2/yb/paper/SLAM-LLM
  --qwen-path /data2/yb/paper_runtime_models/Qwen2-0.5B
  --whisper-path /data2/yb/paper_runtime_models/whisper/small.pt
  --checkpoint /data2/yb/paper/SLAM-Omni-0.5B/model.pt
  --device cuda:0 --seed 1234 --batch-size 1
)
python scripts/run_residual_component_restore.py "${runtime_args[@]}" \
  --output-csv "$report_dir/effects.csv"
python scripts/analyze_residual_component_restore.py \
  --input-csv "$report_dir/effects.csv" \
  --attribution-csv "$report_dir/effects_attribution.csv" \
  --output-dir "$report_dir/analysis"
# A failed gate deliberately skips QK/V; an invalid gate file is an error.
gate_result=$(python -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["attention_wins"]))' "$report_dir/analysis/attention_gate.json")
if [[ "$gate_result" == 1 ]]; then
  python scripts/run_residual_component_restore.py "${runtime_args[@]}" \
    --output-csv "$report_dir/qkv_effects.csv" --qkv-layers 18 19 20 21 22 23
  python scripts/analyze_residual_component_restore.py \
    --input-csv "$report_dir/qkv_effects.csv" \
    --attribution-csv "$report_dir/qkv_effects_attribution.csv" \
    --output-dir "$report_dir/qkv_analysis"
fi
