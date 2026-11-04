"""Evaluate next-byte negative log-likelihood (bits per byte) from a checkpoint."""
from __future__ import annotations

import argparse
import json
import math
import time
import torch
import torch.nn.functional as F

from .data import iter_byte_batches
from .runtime import autocast, build_from_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description="Byte-level perplexity / BPB evaluation")
    p.add_argument("--checkpoint", required=True, help="Checkpoint .pt, latest.pt, or run directory")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-files", help="Local .txt/.jsonl evaluation set")
    source.add_argument("--dataset-name", help="Hugging Face dataset ID (requires [data])")
    p.add_argument("--dataset-config")
    p.add_argument("--split", default="test")
    p.add_argument("--text-column", default="text")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=200, help="Max batches, capped by available data")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--precision", choices=["auto", "fp16", "bf16", "none"], default="auto")
    p.add_argument("--no-document-separator", action="store_true")
    p.add_argument("--output", help="Optional JSON output path")
    args = p.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_from_checkpoint(args.checkpoint, device)
    model.eval()
    iterator = iter_byte_batches(data_files=args.data_files, dataset_name=args.dataset_name,
                                 dataset_config=args.dataset_config, split=args.split,
                                 text_column=args.text_column, block_size=model.cfg.block_size,
                                 batch_size=args.batch_size, seed=args.seed,
                                 shuffle_buffer=0, repeat=False,
                                 add_document_separator=not args.no_document_separator)
    nll = 0.0
    n_scored = 0
    batches = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in iterator:
            if batches >= args.max_batches:
                break
            batch = batch.to(device)
            with autocast(device, args.precision):
                logits = model(batch)["logits"]
                total = F.cross_entropy(logits[:, :-1].reshape(-1, 256).float(),
                                        batch[:, 1:].reshape(-1), reduction="sum")
            nll += float(total.item())
            n_scored += batch.size(0) * (batch.size(1) - 1)
            batches += 1
    if n_scored == 0:
        raise ValueError("No full-context evaluation batches; provide more data or lower batch size")
    result = {"bpb": nll / n_scored / math.log(2), "nll_nats": nll / n_scored,
              "bytes_scored": n_scored, "batches": batches,
              "elapsed_sec": time.perf_counter() - started,
              "checkpoint": str(args.checkpoint), "split": args.split}
    print(json.dumps(result, indent=2))
    if args.output:
        from pathlib import Path
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
