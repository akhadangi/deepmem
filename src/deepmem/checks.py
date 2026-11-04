"""Architecture, invertibility, and future-perturbation smoke tests."""
from __future__ import annotations

import argparse
import json
import torch

from .config import load_config
from .model import LosslessSubbandByteLadderLM, count_parameters
from .runtime import build_from_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description="DeepMem verification")
    p.add_argument("--config", default="configs/tiny_smoke.json")
    p.add_argument("--checkpoint", help="Optional trained checkpoint (overrides --config)")
    p.add_argument("--test", choices=["params", "reconstruction", "causality", "all"], default="all")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args(argv)
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    if args.checkpoint:
        model, _ = build_from_checkpoint(args.checkpoint, device)
    else:
        model = LosslessSubbandByteLadderLM(load_config(args.config)).to(device)
    model.eval()
    results = {"parameters": count_parameters(model), "bottleneck_parameters": model.parameter_breakdown()["bottleneck_shared_state_reasoner"]}
    with torch.no_grad():
        if args.test in {"reconstruction", "all"}:
            x = torch.randint(0, 256, (1, model.cfg.block_size), device=device)
            c0 = model.carrier(x)
            c = c0
            for layer in model.downs:
                c, _ = layer.carrier_lift(c)
            error = float((model.reconstruct_carrier_from_bottom(c) - c0).abs().max().item())
            results["reconstruction_max_error"] = error
            results["reconstruction_passed"] = error <= args.atol
        if args.test in {"causality", "all"}:
            x = torch.randint(0, 256, (1, model.cfg.block_size), device=device)
            cutoff = model.cfg.block_size // 2
            alt = x.clone()
            alt[:, cutoff + 1:] = (alt[:, cutoff + 1:] + 1) % 256
            pred = model(x)["logits"].float()
            pred_alt = model(alt)["logits"].float()
            diff = float((pred[:, :cutoff + 1] - pred_alt[:, :cutoff + 1]).abs().max().item())
            results["causality_max_prefix_logit_diff"] = diff
            results["causality_passed"] = diff <= args.atol
    print(json.dumps(results, indent=2))
    if any(key.endswith("_passed") and not value for key, value in results.items()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
