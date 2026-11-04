"""Portable UTF-8 document packing for local text/JSONL and HF datasets.

No site-specific paths or file-system dependencies. Streams are assigned to
ranks by document index; within a rank, documents are concatenated into
independent block_size-byte windows for next-byte prediction.
"""
from __future__ import annotations

import glob
import itertools
import json
import random
from collections import deque
from pathlib import Path
from typing import Iterator, Optional

import torch


def _local_paths(data_files: str) -> list[Path]:
    paths = []
    for spec in data_files.split(","):
        spec = spec.strip()
        if not spec:
            continue
        path = Path(spec).expanduser()
        if path.is_dir():
            paths.extend(x for x in path.rglob("*") if x.suffix.lower() in {".txt", ".jsonl"} and x.is_file())
        else:
            paths.extend(Path(x) for x in glob.glob(str(path), recursive=True) if Path(x).is_file())
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No .txt or .jsonl files found for {data_files!r}")
    return paths


def _read_local_documents(files: list[Path], text_column: str, *, rng: random.Random, shuffle: bool) -> Iterator[str]:
    files = files.copy()
    if shuffle:
        rng.shuffle(files)
    for path in files:
        if path.suffix.lower() not in {".txt", ".jsonl"}:
            raise ValueError(f"Only .txt and .jsonl are supported locally: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip("\n\r")
                if not line:
                    continue
                if path.suffix.lower() == ".jsonl":
                    record = json.loads(line)
                    text = record.get(text_column)
                    if not isinstance(text, str):
                        continue
                else:
                    text = line
                if text:
                    yield text


def _shuffle_stream(source: Iterator[str], buffer_size: int, rng: random.Random) -> Iterator[str]:
    if buffer_size <= 1:
        yield from source
        return
    it = iter(source)
    buffer = list(itertools.islice(it, buffer_size))
    if not buffer:
        return
    for value in it:
        j = rng.randrange(len(buffer))
        yield buffer[j]
        buffer[j] = value
    rng.shuffle(buffer)
    yield from buffer


def iter_documents(
    *, data_files: Optional[str] = None, dataset_name: Optional[str] = None,
    dataset_config: Optional[str] = None, split: str = "train", text_column: str = "text",
    seed: int = 1337, rank: int = 0, world_size: int = 1,
    shuffle_buffer: int = 0, repeat: bool = False,
) -> Iterator[str]:
    if (bool(data_files) + bool(dataset_name)) != 1:
        raise ValueError("Specify exactly one of --data-files or --dataset-name")
    if rank < 0 or rank >= world_size:
        raise ValueError("invalid rank/world_size")
    paths = _local_paths(data_files) if data_files else None
    epoch = 0
    while True:
        rng = random.Random(seed + epoch)
        if paths is not None:
            raw = _read_local_documents(paths, text_column, rng=rng, shuffle=bool(shuffle_buffer))
        else:
            try:
                from datasets import load_dataset
            except ImportError as exc:
                raise ImportError("Hugging Face streaming needs: pip install 'deepmem-code[data]'") from exc
            data = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
            if shuffle_buffer > 1:
                data = data.shuffle(seed=seed + epoch, buffer_size=shuffle_buffer)
            raw = (row[text_column] for row in data if isinstance(row.get(text_column), str))
        if shuffle_buffer > 1 and paths is not None:
            raw = _shuffle_stream(raw, shuffle_buffer, rng)
        n = 0
        for i, doc in enumerate(raw):
            if i % world_size == rank:
                n += 1
                yield doc
        if n == 0:
            raise RuntimeError(f"No documents assigned to rank {rank} in {split!r}; use more data")
        if not repeat:
            return
        epoch += 1


def iter_byte_blocks(
    documents: Iterator[str], *, block_size: int, add_document_separator: bool = True,
) -> Iterator[torch.Tensor]:
    if block_size < 2:
        raise ValueError("block_size must be >= 2")
    # Small buffer: never accumulate the full dataset in memory.
    buffer = bytearray()
    separator = b"\n" if add_document_separator else b""
    for document in documents:
        buffer.extend(document.encode("utf-8", errors="strict"))
        buffer.extend(separator)
        while len(buffer) >= block_size:
            block = bytes(buffer[:block_size])
            del buffer[:block_size]
            yield torch.tensor(list(block), dtype=torch.long)
    # Any trailing fragment smaller than one full context is deliberately discarded.


def iter_byte_batches(
    *, data_files: Optional[str], dataset_name: Optional[str], dataset_config: Optional[str],
    split: str, text_column: str, block_size: int, batch_size: int, seed: int,
    rank: int = 0, world_size: int = 1, shuffle_buffer: int = 0,
    repeat: bool = False, add_document_separator: bool = True,
) -> Iterator[torch.Tensor]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    # Shard *packed blocks*, not documents: a single long local document must
    # still work with 2 or 4 GPUs. Each rank replays the same stream in the
    # same seed order and takes a disjoint block subset.
    docs = iter_documents(
        data_files=data_files, dataset_name=dataset_name, dataset_config=dataset_config,
        split=split, text_column=text_column, seed=seed, rank=0,
        world_size=1, shuffle_buffer=shuffle_buffer, repeat=repeat,
    )
    full_stream = iter_byte_blocks(docs, block_size=block_size, add_document_separator=add_document_separator)
    blocks = itertools.islice(full_stream, rank, None, world_size)
    while True:
        xs = list(itertools.islice(blocks, batch_size))
        if len(xs) < batch_size:
            return
        yield torch.stack(xs)
