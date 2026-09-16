#!/usr/bin/env bash
set -euo pipefail
cd /data2/yb/paper/audio_llm
omni2_python=/data2/yb/reproduction_envs/light-mer-clean/bin/python
omni2_report=reports/2026-09-16-llama-omni2-confirmation
omni2_dataset=${1:?Choose ravdess or crema-d}
omni2_gpu=${2:-6}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$(dirname "$omni2_python"):$PATH"
command -v ffmpeg >/dev/null
case "$omni2_dataset" in
 ravdess)
  omni2_manifest=artifacts/ravdess_happy_sad_intensity01.csv
  omni2_checksums=$omni2_report/ravdess_audio_checksums.csv
  omni2_audio=/data2/yb/paper/RAVDESS
  ;;
 crema-d)
  omni2_manifest=reports/2026-09-16-crema-confirmation/manifest.csv
  omni2_checksums=reports/2026-09-16-crema-confirmation/audio_checksums.csv
  omni2_audio=/data2/yb/paper/CREMA-D
  ;;
 *) exit 2 ;;
esac
CUDA_VISIBLE_DEVICES="$omni2_gpu" "$omni2_python" scripts/run_model_replication.py \
 --dataset "$omni2_dataset" --manifest "$omni2_manifest" --audio-checksums "$omni2_checksums" \
 --audio-root "$omni2_audio" --source-root /data2/yb/paper/LLaMA-Omni2 \
 --model-path /data2/yb/paper/LLaMA-Omni2-0.5B \
 --whisper-path /data2/yb/paper_runtime_models/whisper/large-v3.pt \
 --output "$omni2_report/$omni2_dataset/intervention_states.npz" --device cuda:0 --resume
"$omni2_python" scripts/analyze_model_replication.py \
 --representations "$omni2_report/$omni2_dataset/intervention_states.npz" \
 --metadata "$omni2_manifest" --dataset "$omni2_dataset" \
 --output-dir "$omni2_report/$omni2_dataset/analysis" --bootstrap 10000
