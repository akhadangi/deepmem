#!/usr/bin/env bash
set -euo pipefail
# Four visible CUDA GPUs, single node, no job scheduler required.
torchrun --standalone --nproc_per_node=4 -m deepmem.train \
  --config "${CONFIG:-configs/paper_r2.json}" \
  --dataset-name "${DATASET_NAME:-HuggingFaceFW/fineweb-edu}" \
  --split train --output-dir "${OUTPUT_DIR:-runs/deepmem_4gpu}" \
  --batch-size "${BATCH_SIZE:-8}" --grad-accum "${GRAD_ACCUM:-8}" \
  --max-steps "${MAX_STEPS:-50000}" --warmup-steps 2000 \
  --lr 3e-4 --min-lr 3e-5 --weight-decay 0.1 \
  --precision fp16 "$@"
