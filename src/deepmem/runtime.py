"""Hardware-independent distributed, precision, and checkpoint helpers."""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .model import LosslessSubbandByteLadderConfig, LosslessSubbandByteLadderLM


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize(seed: int, device_name: str = "auto") -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA GPU is available")
    if device_name.startswith("cuda"):
        torch.cuda.set_device(local_rank if world_size > 1 else (int(device_name.split(":")[-1]) if ":" in device_name else 0))
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo", init_method="env://")
    random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
    return DistributedContext(rank, local_rank, world_size, device, world_size > 1)


def finalize(ctx: DistributedContext) -> None:
    if ctx.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def autocast(device: torch.device, precision: str):
    if precision == "auto":
        precision = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else ("fp16" if device.type == "cuda" else "none")
    if precision == "none":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("fp16/bf16 mixed precision requires CUDA; use --precision none on CPU")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    return torch.autocast("cuda", dtype=dtype)


def uses_fp16(device: torch.device, precision: str) -> bool:
    return device.type == "cuda" and (precision == "fp16" or precision == "auto" and not torch.cuda.is_bf16_supported())


def cosine_lr(step: int, max_steps: int, warmup_steps: int, peak: float, minimum: float) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def raw_model(model):
    return model.module if isinstance(model, DDP) else model


def write_checkpoint(path: Path, *, model, optimizer, scaler, cfg, step: int, bytes_seen: int, args: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": raw_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "model_config": asdict(cfg),
        "step": step,
        "bytes_seen": bytes_seen,
        "training_args": args,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    tmp = path.with_name(path.name + ".incomplete")
    torch.save(payload, tmp)
    tmp.replace(path)
    latest = path.parent / "latest.pt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(path.name)
    except OSError:
        # On Windows or restricted filesystems: resolve latest through latest.json.
        (path.parent / "latest.json").write_text(json.dumps({"checkpoint": path.name}), encoding="utf-8")


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if path.is_dir():
        latest = path / "latest.pt"
        if latest.exists():
            return latest
        pointer = path / "latest.json"
        if pointer.exists():
            return path / json.loads(pointer.read_text(encoding="utf-8"))["checkpoint"]
        raise FileNotFoundError(f"No latest checkpoint in {path}")
    return path


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict:
    path = resolve_checkpoint(path)
    # Only load trusted checkpoints. weights_only avoids arbitrary code execution during unpickling.
    return torch.load(path, map_location=device, weights_only=True)


def build_from_checkpoint(path: str | Path, device: torch.device) -> tuple[LosslessSubbandByteLadderLM, dict]:
    payload = load_checkpoint(path, device="cpu")
    cfg = LosslessSubbandByteLadderConfig(**payload["model_config"])
    model = LosslessSubbandByteLadderLM(cfg)
    model.load_state_dict(payload["model"])
    return model.to(device), payload


def trim_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("step_*.pt"))
    for path in checkpoints[:-keep_last]:
        path.unlink(missing_ok=True)
