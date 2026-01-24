"""Train a Transformer language model on TinyStories with memmap datasets."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.adamw import AdamW
from cs336_basics.bpe import BPETokenizer
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.transformer_lm import Transformer_LM


@dataclass
class TrainConfig:
    train_data: Path
    valid_data: Path
    tokenizer_dir: Path
    vocab_size: int | None
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float
    batch_size: int
    grad_accum_steps: int
    precision: str
    max_iters: int
    lr: float
    betas: tuple[float, float]
    weight_decay: float
    grad_clip: float
    log_interval: int
    eval_interval: int
    eval_iters: int
    save_interval: int
    checkpoint_dir: Path
    resume_from: Path | None
    device: str
    seed: int
    wandb_project: str | None
    wandb_run_name: str | None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a Transformer LM on TinyStories using memmap datasets.")
    parser.add_argument(
        "--train-data",
        type=Path,
        default=Path("models/tinystories_train_uint16.npy"),
        help="Path to training .npy file (uint16 token IDs).",
    )
    parser.add_argument(
        "--valid-data",
        type=Path,
        default=Path("models/tinystories_valid_uint16.npy"),
        help="Path to validation .npy file (uint16 token IDs).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("models/tinystories_train_tokenizer"),
        help="Tokenizer directory containing bpe_vocab.json and bpe_merges.txt.",
    )
    parser.add_argument("--vocab-size", type=int, default=None, help="Override tokenizer vocab size.")
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--d-ff", type=int, default=2048)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=1,
        help="Number of micro-steps to accumulate before optimizer step.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=("fp32", "fp16", "bf16"),
        help="Mixed precision mode (fp32, fp16, bf16).",
    )
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/tinystories"),
        help="Directory to store checkpoints and config.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, cuda).",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--wandb-project", type=str, default=None, help="Weights & Biases project name.")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name.")

    args = parser.parse_args()
    return TrainConfig(
        train_data=args.train_data,
        valid_data=args.valid_data,
        tokenizer_dir=args.tokenizer_dir,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
    grad_accum_steps=args.grad_accum_steps,
    precision=args.precision,
        max_iters=args.max_iters,
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_vocab_size(tokenizer_dir: Path) -> int:
    vocab_path = tokenizer_dir / "bpe_vocab.json"
    merges_path = tokenizer_dir / "bpe_merges.txt"
    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_dir}.")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    return max(tokenizer.vocab.keys()) + 1 if tokenizer.vocab else 0


def load_memmap(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run encode_datasets.py to create memmap .npy files.")
    return np.load(path, mmap_mode="r")


def evaluate(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    eval_iters: int,
    vocab_size: int,
    autocast_context: Any,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(data, batch_size, context_length, device=str(device))
            with autocast_context:
                logits = model(inputs)
                loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def get_autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def maybe_init_wandb(config: TrainConfig) -> Any:
    if not config.wandb_project:
        return None
    import wandb

    run = wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=asdict(config),
    )
    return run


def save_config(config: TrainConfig) -> None:
    config_path = config.checkpoint_dir / "train_config.json"
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_data = load_memmap(config.train_data)
    valid_data = load_memmap(config.valid_data)
    if len(train_data) <= config.context_length + 1:
        raise ValueError("Training data is too small for the chosen context length.")

    vocab_size = config.vocab_size or load_vocab_size(config.tokenizer_dir)
    model = Transformer_LM(
        vocab_size,
        config.context_length,
        config.d_model,
        config.num_layers,
        config.num_heads,
        config.d_ff,
        config.rope_theta,
        device=device,
    )
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    autocast_context = get_autocast_context(device, config.precision)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda" and config.precision == "fp16"))

    start_iter = 0
    if config.resume_from:
        start_iter = load_checkpoint(config.resume_from, model, optimizer)
        print(f"Resumed from {config.resume_from} at iteration {start_iter}.")

    save_config(config)
    wandb_run = maybe_init_wandb(config)

    print(
        "Training config:\n"
        f"  Device: {device}\n"
        f"  Vocab size: {vocab_size}\n"
        f"  Precision: {config.precision}\n"
        f"  Grad accum steps: {config.grad_accum_steps}\n"
        f"  Train tokens: {len(train_data):,}\n"
        f"  Valid tokens: {len(valid_data):,}\n"
    )

    model.train()
    last_log_time = time.time()

    for step in range(start_iter, config.max_iters):
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        try:
            for _ in range(config.grad_accum_steps):
                inputs, targets = get_batch(train_data, config.batch_size, config.context_length, device=str(device))
                with autocast_context:
                    logits = model(inputs)
                    micro_loss = cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
                accum_loss += micro_loss.item()
                micro_loss = micro_loss / config.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(micro_loss).backward()
                else:
                    micro_loss.backward()
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                print(
                    "CUDA OOM: try reducing --batch-size, increasing --grad-accum-steps, "
                    "or using --precision bf16/fp16."
                )
                torch.cuda.empty_cache()
            raise exc

        if config.grad_clip > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            gradient_clipping(model.parameters(), config.grad_clip)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        avg_loss = accum_loss / config.grad_accum_steps

        if (step + 1) % config.log_interval == 0:
            now = time.time()
            step_time = now - last_log_time
            tokens_per_sec = (
                config.batch_size
                * config.context_length
                * config.log_interval
                * config.grad_accum_steps
                / max(step_time, 1e-9)
            )
            last_log_time = now
            log_msg = (
                f"Iter {step + 1}/{config.max_iters} | "
                f"loss {avg_loss:.4f} | "
                f"lr {config.lr:.2e} | "
                f"tok/s {tokens_per_sec:,.0f}"
            )
            print(log_msg)
            if wandb_run:
                wandb_run.log(
                    {
                        "train/loss": avg_loss,
                        "train/tokens_per_sec": tokens_per_sec,
                        "iteration": step + 1,
                    }
                )

        if (step + 1) % config.eval_interval == 0:
            val_loss = evaluate(
                model,
                valid_data,
                config.batch_size,
                config.context_length,
                device,
                config.eval_iters,
                vocab_size,
                autocast_context,
            )
            print(f"Validation @ {step + 1}: loss {val_loss:.4f}")
            if wandb_run:
                wandb_run.log({"valid/loss": val_loss, "iteration": step + 1})

        if (step + 1) % config.save_interval == 0:
            config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = config.checkpoint_dir / f"checkpoint_{step + 1}.pt"
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
