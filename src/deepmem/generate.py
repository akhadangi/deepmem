"""Simple (non-cached) demonstration of autoregressive UTF-8 byte generation."""
from __future__ import annotations

import argparse
import torch

from .runtime import build_from_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate bytes using a trained DeepMem checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="The ")
    p.add_argument("--max-new-bytes", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args(argv)
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_from_checkpoint(args.checkpoint, device)
    model.eval()
    output = bytearray(args.prompt.encode("utf-8"))
    context = model.cfg.block_size
    with torch.inference_mode():
        for _ in range(args.max_new_bytes):
            # This architecture expects a full block. Left-padding with spaces is
            # an inference convenience, not a special trained padding token.
            window = bytes(output[-context:]).rjust(context, b" ")
            batch = torch.tensor(list(window), dtype=torch.long, device=device)[None]
            logits = model(batch)["logits"][0, -1].float() / args.temperature
            chosen = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
            output.append(chosen)
    print(output.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
