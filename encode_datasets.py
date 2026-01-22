#!/usr/bin/env python3
"""Encode datasets into uint16 token IDs.

This script loads BPE tokenizers and encodes the specified datasets
into NumPy .npy arrays (dtype=uint16) for efficient training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Iterable

import numpy as np

from cs336_basics.bpe import BPETokenizer


def iter_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        yield from f


def count_tokens(tokenizer: BPETokenizer, path: Path) -> int:
    total = 0
    for _ in tokenizer.encode_iterable(iter_lines(path)):
        total += 1
    return total


def encode_to_memmap(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    total_tokens: int,
    chunk_size: int = 1_000_000,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    memmap = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.uint16,
        shape=(total_tokens,),
    )

    index = 0
    buffer: list[int] = []
    for token_id in tokenizer.encode_iterable(iter_lines(input_path)):
        buffer.append(token_id)
        if len(buffer) >= chunk_size:
            arr = np.asarray(buffer, dtype=np.uint16)
            memmap[index : index + len(arr)] = arr
            index += len(arr)
            buffer.clear()

    if buffer:
        arr = np.asarray(buffer, dtype=np.uint16)
        memmap[index : index + len(arr)] = arr
        index += len(arr)

    if index != total_tokens:
        raise RuntimeError(f"Token count mismatch: expected {total_tokens}, wrote {index}")


def load_tokenizer(tokenizer_dir: Path, special_tokens: list[str]) -> BPETokenizer:
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Tokenizer files not found in {tokenizer_dir}. Expected bpe_vocab.json and bpe_merges.txt."
        )
    return BPETokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)


def ensure_uint16_compatible(tokenizer: BPETokenizer) -> None:
    max_id = max(tokenizer.vocab.keys()) if tokenizer.vocab else 0
    if max_id > np.iinfo(np.uint16).max:
        raise ValueError(f"Tokenizer vocab max id {max_id} exceeds uint16 range; use uint32 instead.")


def encode_dataset(
    name: str,
    train_path: Path,
    valid_path: Path,
    tokenizer_dir: Path,
    out_dir: Path,
    special_tokens: list[str],
    chunk_size: int,
) -> None:
    print(f"\n==> Encoding dataset: {name}")
    print(f"Tokenizer: {tokenizer_dir}")

    tokenizer = load_tokenizer(tokenizer_dir, special_tokens)
    ensure_uint16_compatible(tokenizer)

    for split_name, split_path in ("train", train_path), ("valid", valid_path):
        if not split_path.exists():
            raise FileNotFoundError(f"{name} {split_name} file not found: {split_path}")
        output_path = out_dir / f"{name}_{split_name}_ids_uint16.npy"
        print(f"Counting tokens for {split_name}...")
        total_tokens = count_tokens(tokenizer, split_path)
        print(f"Total tokens: {total_tokens:,}")
        print(f"Writing {output_path}...")
        encode_to_memmap(tokenizer, split_path, output_path, total_tokens, chunk_size=chunk_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode datasets to uint16 token IDs.")
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "owt", "all"],
        default="tinystories",
        help="Which dataset(s) to encode.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/encoded"),
        help="Output directory for .npy files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Number of token IDs to buffer before writing to disk.",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=["<|endoftext|>"],
        help="Special tokens to keep intact (repeatable).",
    )
    parser.add_argument(
        "--tinystories-tokenizer-dir",
        type=Path,
        default=Path("models/tinystories_train_tokenizer"),
        help="Tokenizer directory for TinyStories.",
    )
    parser.add_argument(
        "--owt-tokenizer-dir",
        type=Path,
        default=Path("models/owt_train_tokenizer"),
        help="Tokenizer directory for OpenWebText.",
    )
    parser.add_argument(
        "--skip-missing-tokenizer",
        action="store_true",
        help="Skip dataset if tokenizer files are missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "tinystories": {
            "train": Path("data/TinyStoriesV2-GPT4-train.txt"),
            "valid": Path("data/TinyStoriesV2-GPT4-valid.txt"),
            "tokenizer_dir": args.tinystories_tokenizer_dir,
        },
        "owt": {
            "train": Path("data/owt_train.txt"),
            "valid": Path("data/owt_valid.txt"),
            "tokenizer_dir": args.owt_tokenizer_dir,
        },
    }

    target = ["tinystories", "owt"] if args.dataset == "all" else [args.dataset]

    for name in target:
        config = datasets[name]
        tokenizer_dir = config["tokenizer_dir"]
        if not tokenizer_dir.exists() and args.skip_missing_tokenizer:
            print(f"Skipping {name}: tokenizer dir not found at {tokenizer_dir}")
            continue
        encode_dataset(
            name=name,
            train_path=config["train"],
            valid_path=config["valid"],
            tokenizer_dir=tokenizer_dir,
            out_dir=out_dir,
            special_tokens=args.special_token,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
