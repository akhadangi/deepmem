"""Illustrative zero-shot multiple-choice evaluator for HellaSwag, PIQA, and ARC.

IMPORTANT: This is NOT the original benchmark harness used in the paper.
Prompt templates, answer normalization, preprocessing and corruption protocols
may differ from the reported experiments. Do not claim exact score reproduction.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .runtime import autocast, build_from_checkpoint


DATASETS = {
    "hellaswag": ("Rowan/hellaswag", None, "validation"),
    "piqa": ("ybisk/piqa", None, "validation"),
    "arc_easy": ("allenai/ai2_arc", "ARC-Easy", "validation"),
    "arc_challenge": ("allenai/ai2_arc", "ARC-Challenge", "validation"),
}


def candidates_and_answer(row: dict, task: str) -> tuple[str, list[str], int]:
    if task == "hellaswag":
        prefix = str(row.get("ctx", ""))
        options = [str(x) for x in row["endings"]]
        correct = int(row["label"])
    elif task == "piqa":
        prefix = "Goal: " + str(row["goal"]) + "\nSolution: "
        options = [str(row["sol1"]), str(row["sol2"])]
        correct = int(row["label"])
    elif task in {"arc_easy", "arc_challenge"}:
        prefix = "Question: " + str(row["question"]) + "\nAnswer: "
        choices = row["choices"]
        options = [str(x) for x in choices["text"]]
        keys = [str(x) for x in choices["label"]]
        answer = str(row["answerKey"])
        if answer not in keys:
            raise ValueError(f"ARC answerKey {answer!r} not in {keys!r}")
        correct = keys.index(answer)
    else:
        raise ValueError(task)
    if not options or not 0 <= correct < len(options):
        raise ValueError("Example has no valid gold answer")
    return prefix, options, correct


def perturb_text(text: str, name: str, *, rng: random.Random, rate: float) -> str:
    if name == "clean":
        return text
    if name == "uppercase":
        return text.upper()
    if name == "randomcase":
        return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)
    if name == "deletion":
        result = "".join(c for c in text if c.isspace() or rng.random() >= rate)
        return result if result else " "
    if name == "repetition":
        return "".join(c * (2 if not c.isspace() and rng.random() < rate else 1) for c in text)
    if name == "antspeak":
        return " ".join(text)
    raise ValueError(name)


def _encode_candidate(prompt: str, continuation: str, block_size: int) -> tuple[torch.Tensor, int]:
    pre = prompt.encode("utf-8") or b" "
    cont = continuation.encode("utf-8")
    if len(cont) == 0:
        raise ValueError("Empty candidate is not supported")
    if len(cont) >= block_size:
        raise ValueError("Continuation is too long for a single DeepMem context block")
    # Discard old prompt bytes if needed, but keep at least one prefix byte.
    keep = min(len(pre), block_size - len(cont))
    sequence = pre[-keep:] + cont
    left_pad = block_size - len(sequence)
    sequence = b" " * left_pad + sequence
    start = block_size - len(cont)
    return torch.tensor(list(sequence), dtype=torch.long), start


@torch.inference_mode()
def score_candidates(model, prompt: str, candidates: list[str], *, precision: str, length_norm: bool) -> list[float]:
    device = next(model.parameters()).device
    result = []
    for candidate in candidates:
        ids, start = _encode_candidate(prompt, candidate, model.cfg.block_size)
        ids = ids.to(device)[None]
        with autocast(device, precision):
            logits = model(ids)["logits"][0, start - 1:-1].float()
        targets = ids[0, start:]
        scores = F.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).sum().item()
        if length_norm:
            scores /= len(targets)
        result.append(float(scores))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Illustrative DeepMem zero-shot multiple-choice evaluation")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", choices=list(DATASETS), required=True)
    p.add_argument("--dataset-name", help="Override the default HF dataset ID")
    p.add_argument("--dataset-config", help="Override default HF dataset subset")
    p.add_argument("--split", help="Override default validation split")
    p.add_argument("--max-samples", type=int, default=100)
    p.add_argument("--precision", choices=["auto", "none", "fp16", "bf16"], default="auto")
    p.add_argument("--scoring", choices=["sum", "mean"], default="sum",
                   help="sum=total candidate log-likelihood, mean=per-byte normalization")
    p.add_argument("--perturb", choices=["clean", "deletion", "repetition", "antspeak", "uppercase", "randomcase"], default="clean")
    p.add_argument("--perturb-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--output", help="Optional JSON output path")
    args = p.parse_args(argv)
    if not 0 <= args.perturb_rate <= 1:
        raise ValueError("--perturb-rate must be in [0,1]")
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError("Install Hugging Face datasets: pip install 'deepmem-code[data]'") from e
    name, subset, split = DATASETS[args.task]
    name = args.dataset_name or name
    subset = args.dataset_config if args.dataset_config else subset
    split = args.split or split
    stream = load_dataset(name, subset, split=split, streaming=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = build_from_checkpoint(args.checkpoint, device)
    model.eval()
    correct = 0
    evaluated = 0
    rng = random.Random(args.seed)
    skipped = 0
    for row in stream:
        if evaluated >= args.max_samples:
            break
        try:
            prompt, options, answer = candidates_and_answer(row, args.task)
            if args.perturb != "clean":
                prompt = perturb_text(prompt, args.perturb, rng=rng, rate=args.perturb_rate)
                options = [perturb_text(t, args.perturb, rng=rng, rate=args.perturb_rate) for t in options]
            likelihoods = score_candidates(model, prompt, options, precision=args.precision,
                                           length_norm=(args.scoring == "mean"))
            prediction = max(range(len(likelihoods)), key=likelihoods.__getitem__)
            correct += int(prediction == answer)
            evaluated += 1
        except (ValueError, KeyError, TypeError):
            skipped += 1
    if not evaluated:
        raise RuntimeError("No usable examples; check the dataset split and fields")
    result = {"task": args.task, "dataset": name, "split": split, "accuracy": correct / evaluated,
              "correct": correct, "evaluated": evaluated, "skipped": skipped,
              "scoring": args.scoring, "perturb": args.perturb, "perturb_rate": args.perturb_rate,
              "warning": "Illustrative prompts and perturbations; not the original paper evaluation harness"}
    print(json.dumps(result, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
