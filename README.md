# DeepMem — Going Deeper With Memories

**Reference PyTorch implementation of DeepMem**, a causal multiscale byte-level language model with an invertible lifting carrier, FastWeight-based analysis path, iterative shared-state bottleneck refinement, and causal attention-based super-resolution decoder.

**Paper:** [Going Deeper With Memories](https://arxiv.org/abs/2611.05255) (arXiv:2611.05255, 2026)  
**Author:** Afshin Khadangi

This repository is designed to run **without SLURM, cluster-specific modules, filesystem paths, or private `seal_m` dependencies**. Use Python for one GPU, or PyTorch `torchrun` for 2–4 (or more) GPUs on one machine. CPU operation is supported for small verification runs.

> **Reproducibility scope:** The main architecture was extracted from the author's supplied DeepMem implementation and its paper configuration was recreated from the supplied training script. The original `seal_m.data.byte_stream` and other `seal_m` runtime modules, experiment checkpoints, dataset snapshots, and benchmark evaluation scripts were **not** supplied. The portable data/training/evaluation implementations here are new, so identical published BPB numbers are **not guaranteed** merely by using the same hyperparameters. The paper configuration has been checked to instantiate exactly **100,158,582 trainable parameters**, including **69,280** parameters in the shared-state module. No pretrained weights are included.

## Features

- Tokenizer-free, next-byte prediction over the **256 possible byte values**.
- Learned pointwise invertible lifting of a fixed 8-bit bipolar carrier; recursive carrier recovery in real arithmetic.
- Multiscale semantic encoder with gated FastWeight recurrence (scalar head gates in the paper).
- **Two** parameter-shared refinement steps at the 16-position bottleneck.
- Causally delayed parent proposals plus an attention-based super-resolution decoder.
- Single-GPU/CPU training, `torchrun`-based DistributedDataParallel (DDP), mixed precision, gradient accumulation, checkpoints and resume.
- Streaming Hugging Face dataset support (optional), and plain local text or JSONL files.
- BPB evaluation, an illustrative zero-shot multiple-choice utility, simple byte sampling, reconstruction and causality checks.

## Setup

Python **3.10+**, PyTorch **2.2+**. Use a PyTorch build appropriate for your local GPU/CUDA environment. For CUDA installation options, see [PyTorch's installation guide](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/akhadangi/deepmem.git
cd deepmem
python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[data,test]"         # installs torch, HF datasets, pytest
# Optional: pip install -e ".[all]"    # includes wandb
```

To install only the architecture and run local `.txt` data, `pip install -e .` is sufficient.

## 1. Verify installation (CPU / any machine)

```bash
python -m deepmem.checks --config configs/tiny_smoke.json --test all --device cpu
pytest -q
```

The tiny configuration is intentionally **not** the 100M paper model. It uses a 32-byte context and small widths so tests finish on a laptop.

Confirm the exact reference architecture (takes more RAM than the tiny checks):

```bash
python -m deepmem.checks --config configs/paper_r2.json --test params --device cpu
```

Expected: `parameters: 100158582`, `bottleneck_parameters: 69280`. You can also check the full carrier and causal graph with `--test all`, but this can be expensive for the paper model.

## 2. Fast local training smoke test

```bash
python -m deepmem.train \
  --config configs/tiny_smoke.json \
  --data-files examples/sample.txt \
  --output-dir runs/smoke \
  --device cpu --precision none \
  --batch-size 1 --grad-accum 1 \
  --max-steps 2 --warmup-steps 1 \
  --log-every 1 --save-every 2 --shuffle-buffer 0
```

Creates `runs/smoke/step_00000002.pt`, `latest.pt`, and `metrics.jsonl`.

## 3. FineWeb-Edu training — one GPU

`configs/paper_r2.json` is the exact **architecture** described by the supplied 4-GPU SLURM script. The example below keeps the same **effective batch size of 256 sequences** as the paper but uses a microbatch of one on a single GPU. This is much slower, and the full model may still exceed the VRAM of smaller cards.

```bash
python -m deepmem.train \
  --config configs/paper_r2.json \
  --dataset-name HuggingFaceFW/fineweb-edu --split train \
  --text-column text --shuffle-buffer 20000 \
  --output-dir runs/deepmem_1gpu \
  --batch-size 1 --grad-accum 256 \
  --max-steps 50000 --warmup-steps 2000 \
  --lr 3e-4 --min-lr 3e-5 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 \
  --grad-clip 1.0 --precision auto \
  --log-every 10 --save-every 1000
```

If the model does not fit, **reduce `--batch-size` first**, then consider changing a copy of the JSON config to set `"grad_checkpointing": true`. Gradient checkpointing trades compute for activation memory. Do not edit the paper reference JSON if you want the exact reference architecture and no-checkpoint training regime.

## 4. FineWeb-Edu training — four GPUs (one node)

No SLURM required. `torchrun` creates the four workers and automatically sets rank, local rank, and world size. Run on a machine with four visible GPUs:

```bash
torchrun --standalone --nproc_per_node=4 -m deepmem.train \
  --config configs/paper_r2.json \
  --dataset-name HuggingFaceFW/fineweb-edu --split train \
  --text-column text --shuffle-buffer 20000 \
  --output-dir runs/deepmem_4gpu \
  --batch-size 8 --grad-accum 8 \
  --max-steps 50000 --warmup-steps 2000 \
  --lr 3e-4 --min-lr 3e-5 \
  --weight-decay 0.1 --beta1 0.9 --beta2 0.95 \
  --grad-clip 1.0 --precision fp16 \
  --log-every 10 --save-every 1000
```

Effective batch size = `batch-size × grad-accum × GPUs` = `8 × 8 × 4 = 256`. At 2,048 bytes/context this is **524,288 input byte presentations per optimizer step**. For 2 GPUs, use `--nproc_per_node=2 --batch-size 8 --grad-accum 16` to retain the same effective batch size.

The original experiment used **4 NVIDIA V100-SXM2 32-GB GPUs**. The repository supports other CUDA GPUs, but the same microbatch and throughput are not guaranteed on different hardware. Multi-GPU support uses DDP, not `DataParallel` or multi-node FSDP.

**Optional W&B**: Add `--wandb-project deepmem --wandb-run-name example_run` after `pip install -e ".[wandb]"`.

## 5. Train on your own data

Local text (UTF-8; each nonempty line is a document) or JSONL (`{"text":"..."}`):

```bash
python -m deepmem.train \
  --config configs/tiny_smoke.json \
  --data-files data/train.jsonl \
  --text-column text --output-dir runs/custom \
  --batch-size 2 --grad-accum 4 --max-steps 100
```

- `--data-files`: a `.txt` / `.jsonl` file, directory, glob, or comma-separated file list.
- `--dataset-name`: instead streams a Hugging Face dataset (requires optional `[data]` package).
- With DDP, packed byte blocks are partitioned across ranks by **block index**. This is generic but can repeat upstream data reads per rank; for very large streams, a dataset-native sharding implementation is recommended.
- Documents are packed into consecutive full-length **non-overlapping** `block_size` windows. A newline separator is inserted between documents unless you pass `--no-document-separator`. Incomplete final fragments are dropped.
- `--shuffle-buffer` controls document-level shuffle for local data and HF streams. Set `0` to preserve order. This implementation does not reproduce the original unseen `seal_m` packing/shuffle logic bit-for-bit.

## 6. Evaluate BPB

For a local test dataset:

```bash
python -m deepmem.evaluate \
  --checkpoint runs/smoke \
  --data-files examples/sample.txt \
  --batch-size 1 --max-batches 4 \
  --output runs/smoke/eval.json
```

Or for an HF test set:

```bash
python -m deepmem.evaluate \
  --checkpoint runs/deepmem_4gpu \
  --dataset-name deepmind/pg19 --split test \
  --text-column text --max-batches 100 \
  --batch-size 1
```

BPB = summed next-byte negative log-likelihood in nats ÷ number of scored bytes ÷ `ln(2)`. Each full block scores positions `1..block_size-1`; incomplete blocks are dropped. Evaluation **does not silently repeat** the dataset to reach `max-batches`.

## Optional illustrative zero-shot multiple-choice evaluation

A **new** (not original) evaluator supports HellaSwag, PIQA, ARC-Easy, and ARC-Challenge:

```bash
python -m deepmem.eval_mc \
  --checkpoint runs/deepmem_4gpu \
  --task piqa --max-samples 100 --scoring sum
```

Use `--task hellaswag`, `--task arc_easy`, or `--task arc_challenge`. A demonstration corruption mode is available via `--perturb repetition` (also: `deletion`, `antspeak`, `uppercase`, `randomcase`). **This utility is illustrative only**: its templates, left-padding, length-handling and corruption rules were written for this release and cannot be assumed to match the paper's unavailable benchmark harness. The model is fixed-context and evaluation can be slow because every candidate uses a full forward pass.

## 7. Generate text

```bash
python -m deepmem.generate \
  --checkpoint runs/deepmem_4gpu \
  --prompt "Once upon a time" --max-new-bytes 64
```

This is a simple correctness demo. Every generated byte invokes a full-context forward pass; there is **no KV/cache optimization**. Since the model requires exactly `block_size` bytes, short prompts are left-padded with ASCII spaces, which may influence output quality. Generation produces **bytes**, so incomplete UTF-8 sequences can be displayed with replacement characters.

## 8. Verify a trained checkpoint

```bash
python -m deepmem.checks \
  --checkpoint runs/deepmem_4gpu \
  --test all --atol 1e-5
```

`reconstruction` checks the algebraic lifting inverse numerically; `causality` compares logits for identical prefixes and changed suffixes. These are **smoke checks**, not an exhaustive correctness proof. Numerical tolerance depends on hardware and precision.

## 9. Resume a run

```bash
python -m deepmem.train \
  --config configs/paper_r2.json \
  --dataset-name HuggingFaceFW/fineweb-edu \
  --output-dir runs/deepmem_1gpu \
  --batch-size 1 --grad-accum 256 \
  --resume runs/deepmem_1gpu \
  --max-steps 50000
```

You can also point `--resume` to a specific `step_XXXXXXXX.pt`. Resume restores weights, optimizer, scaler, step and aggregate byte count. **The streaming dataset cursor and exact per-rank RNG state are not restored**, so this is a practical training resume, not a bitwise-identical continuation. For a full paper reproduction, save dataset snapshots and RNG/dataloader states.

## Architecture settings

The reference JSON config corresponds to:

| Setting | Value |
|---|---|
| Input symbols | 256 UTF-8 bytes |
| Context | 2,048 bytes |
| Sequence lengths | 2048, 1024, 512, 256, 128, 64, 32, 16 |
| Semantic widths | 1024, 512, 256, 128, 128, 128, 128, 128 |
| Carrier widths | 8, 16, 32, 64, 128, 256, 512, 1024 |
| Down / up depths | 2, 2, 2, 3, 3, 3, 4 |
| Encoder | Scalar-gated FastWeight |
| Decoder | Causal self-attention |
| Bottleneck | 16 × 128, shared-state mixer, R = 2 |
| Direct all-scale fusion | Disabled |
| Parameters | 100,158,582 |

The original source also includes **experimental** PRISM and vector-gated mixer options. They are retained in the architecture implementation but are *not* enabled in `configs/paper_r2.json`; they should not be interpreted as evaluated DeepMem paper configurations.

## Project layout

```text
deepmem/
├── configs/
│   ├── paper_r2.json          # paper architecture, 100,158,582 params
│   └── tiny_smoke.json        # lightweight smoke tests
├── examples/sample.txt
├── src/deepmem/
│   ├── model.py               # original architecture adapted to standalone use
│   ├── config.py              # config loading
│   ├── data.py                # local / HF byte stream packing
│   ├── train.py               # 1 GPU / DDP training
│   ├── evaluate.py            # bits per byte
│   ├── eval_mc.py             # illustrative multiple-choice evaluation
│   ├── generate.py            # slow but simple generation
│   ├── checks.py              # causality and inverse lifting smoke checks
│   └── runtime.py             # distributed runtime and checkpoints
├── scripts/                   # scheduler-free 1-/4-GPU examples
├── .github/workflows/tests.yml
├── tests/
├── CITATION.cff
├── LICENSE                   # MIT; verify this is your desired license
└── pyproject.toml
```

## Important limitations

- The release does **not** bundle trained weights, cached datasets, full baseline implementations, or the **original** zero-shot/robustness evaluation harness. Those were not present in the provided source. The included BPB evaluation is functional but is not a claim of exact reproduction of the paper's task-specific formatting and scoring protocols.
- The full 100M model is expensive; CUDA memory and speed depend strongly on batch size, GPU type, gradient checkpointing, and PyTorch version.
- **Algebraic invertibility** holds in exact arithmetic. Floating-point reconstructions may differ slightly.
- Public benchmark results depend on dataset versions, preprocessing, seed assignments, and the exact training implementation. Please report these when publishing independent replications.
- This repository contains an **MIT license** as a suggested default; the author should confirm this is the intended licensing decision before public release.

## Citation

If you use this work, please cite:

```bibtex
@misc{khadangi2026goingdeepermemories,
      title={Going Deeper With Memories},
      author={Afshin Khadangi},
      year={2026},
      eprint={2611.05255},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2611.05255},
}
```
