#!/usr/bin/env bash
set -euo pipefail
# Portable one-GPU paper-configuration example; override with env vars or extra args.
python -m deepmem.train \
  --config "${CONFIG:-configs/paper_r2.json}" \
  --dataset-name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --split train --output-dir "${OUTPUT_DIR:-runs/deepmem_1gpu}" \
  --batch-size "${BATCH_SIZE:-1}" --grad-accum "${GRAD_ACCUM:-256}" \
  --max-steps "${MAX_STEPS:-50000}" --warmup-steps 2000 \
  --lr 3e-4 --min-lr 3e-5 --weight-decay 0.1 \
  --precision auto "$@"
