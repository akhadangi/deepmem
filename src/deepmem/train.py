"""Single-device and torchrun (DDP) training entry point for DeepMem.

Examples:
  python -m deepmem.train --config configs/tiny_smoke.json --data-files examples/sample.txt --output-dir runs/smoke --max-steps 2
  torchrun --standalone --nproc_per_node=4 -m deepmem.train --config configs/paper_r2.json --dataset-name HuggingFaceFW/fineweb-edu ...
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import load_config, save_config
from .data import iter_byte_batches
from .model import LosslessSubbandByteLadderLM, count_parameters
from .runtime import (autocast, cosine_lr, finalize, initialize, load_checkpoint,
                      raw_model, trim_checkpoints, uses_fp16, write_checkpoint)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DeepMem on UTF-8 bytes; supports 1 GPU and torchrun/DDP")
    p.add_argument("--config", required=True, help="Architecture JSON (e.g., configs/paper_r2.json)")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-files", help="Local .txt/.jsonl, directory, glob or comma-separated files")
    source.add_argument("--dataset-name", help="Hugging Face dataset ID (requires deepmem-code[data])")
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="train")
    p.add_argument("--text-column", default="text")
    p.add_argument("--shuffle-buffer", type=int, default=20000)
    p.add_argument("--no-document-separator", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--resume", help="Checkpoint .pt, latest.pt, or directory containing latest.pt")
    p.add_argument("--max-steps", type=int, default=50000)
    p.add_argument("--batch-size", type=int, default=1, help="Microbatch per process/GPU")
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-steps", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--keep-last", type=int, default=3)
    p.add_argument("--wandb-project", help="Optional Weights & Biases project (requires deepmem-code[wandb])")
    p.add_argument("--wandb-run-name")
    return p


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if min(args.batch_size, args.grad_accum, args.max_steps, args.log_every, args.save_every) <= 0:
        raise ValueError("batch-size, grad-accum, max-steps, log-every and save-every must be positive")
    if args.min_lr > args.lr:
        raise ValueError("--min-lr must not exceed --lr")
    ctx = initialize(args.seed, args.device)
    try:
        cfg = load_config(args.config)
        model = LosslessSubbandByteLadderLM(cfg).to(ctx.device)
        if ctx.is_main:
            print(json.dumps({"parameters": count_parameters(model), "world_size": ctx.world_size,
                              "effective_batch_size": args.batch_size * args.grad_accum * ctx.world_size,
                              "config": args.config}), flush=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
        use_scaler = uses_fp16(ctx.device, args.precision)
        if hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:  # Compatibility with older supported PyTorch releases.
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
        start_step = 0
        bytes_seen = 0
        if args.resume:
            state = load_checkpoint(args.resume, device="cpu")
            if state["model_config"] != __import__("dataclasses").asdict(cfg):
                raise ValueError("Resume checkpoint architecture differs from --config")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if "scaler" in state:
                scaler.load_state_dict(state["scaler"])
            # Move optimizer tensors to local device after loading CPU checkpoint.
            for entry in optimizer.state.values():
                for key, value in entry.items():
                    if torch.is_tensor(value):
                        entry[key] = value.to(ctx.device)
            start_step = int(state["step"])
            bytes_seen = int(state.get("bytes_seen", 0))
            if ctx.is_main:
                print(f"Resuming model and optimizer at step {start_step}. Data stream starts at a new epoch.", flush=True)
        if ctx.distributed:
            model = DDP(model, device_ids=[ctx.local_rank] if ctx.device.type == "cuda" else None,
                        find_unused_parameters=(cfg.disable_carrier_to_sem or cfg.disable_multiscale))
        output = Path(args.output_dir)
        if ctx.is_main:
            output.mkdir(parents=True, exist_ok=True)
            save_config(cfg, output / "model_config.json")
            (output / "run_args.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
            log = (output / "metrics.jsonl").open("a", encoding="utf-8")
            if args.wandb_project:
                try:
                    import wandb
                except ImportError as e:
                    raise ImportError("Install optional support: pip install 'deepmem-code[wandb]'") from e
                wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        else:
            log = None
            wandb = None
        if ctx.distributed:
            dist.barrier()
        iterator = iter_byte_batches(data_files=args.data_files, dataset_name=args.dataset_name,
                                     dataset_config=args.dataset_config, split=args.split,
                                     text_column=args.text_column, block_size=cfg.block_size,
                                     batch_size=args.batch_size, seed=args.seed + start_step,
                                     rank=ctx.rank, world_size=ctx.world_size,
                                     shuffle_buffer=args.shuffle_buffer, repeat=True,
                                     add_document_separator=not args.no_document_separator)
        model.train()
        started = time.monotonic()
        for step in range(start_step, args.max_steps):
            lr = cosine_lr(step, args.max_steps, args.warmup_steps, args.lr, args.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=ctx.device)
            for micro in range(args.grad_accum):
                batch = next(iterator).to(ctx.device, non_blocking=True)
                no_sync = model.no_sync() if ctx.distributed and micro != args.grad_accum - 1 else nullcontext()
                with no_sync, autocast(ctx.device, args.precision):
                    loss = model(batch, targets=batch)["loss"]
                    scaled_loss = loss / args.grad_accum
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                loss_sum += scaled_loss.detach()
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip) if args.grad_clip > 0 else torch.tensor(0.0, device=ctx.device)
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            bytes_seen += args.batch_size * cfg.block_size * ctx.world_size * args.grad_accum
            if (step + 1) % args.log_every == 0 or step == start_step:
                if ctx.distributed:
                    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
                    loss_sum /= ctx.world_size
                elapsed = max(1e-9, time.monotonic() - started)
                loss_value = float(loss_sum.item())
                record = {"step": step + 1, "train_loss_nats": loss_value,
                          "train_bpb": loss_value / math.log(2), "lr": lr,
                          "grad_norm": float(norm.detach().cpu()), "bytes_seen": bytes_seen,
                          "elapsed_sec": elapsed, "world_size": ctx.world_size}
                if ctx.is_main:
                    print(json.dumps(record), flush=True)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if args.wandb_project:
                        wandb.log(record, step=step + 1)
            if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
                if ctx.distributed:
                    dist.barrier()
                if ctx.is_main:
                    write_checkpoint(output / f"step_{step + 1:08d}.pt", model=model, optimizer=optimizer,
                                     scaler=scaler, cfg=cfg, step=step + 1, bytes_seen=bytes_seen,
                                     args=vars(args))
                    trim_checkpoints(output, args.keep_last)
                if ctx.distributed:
                    dist.barrier()
        if ctx.is_main:
            log.close()
            if args.wandb_project:
                wandb.finish()
    finally:
        finalize(ctx)


if __name__ == "__main__":
    main()
