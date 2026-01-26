from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from cs336_basics.bpe import BPETokenizer
from cs336_basics.transformer_lm import Transformer_LM


@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float


def load_model_config(config_path: Path) -> ModelConfig:
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab_size = raw.get("vocab_size", 0) or 0
    return ModelConfig(
        vocab_size=vocab_size,
        context_length=int(raw["context_length"]),
        d_model=int(raw["d_model"]),
        num_layers=int(raw["num_layers"]),
        num_heads=int(raw["num_heads"]),
        d_ff=int(raw["d_ff"]),
        rope_theta=float(raw["rope_theta"]),
    )


def load_tokenizer(vocab_path: Path, merges_path: Path) -> tuple[BPETokenizer, int | None]:
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])
    eos_token_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    return tokenizer, eos_token_id


def infer_vocab_size(tokenizer: BPETokenizer) -> int:
    return max(tokenizer.vocab.keys()) + 1 if tokenizer.vocab else 0


class Inference:
    def __init__(
        self,
        vocab_path: str | Path,
        merges_path: str | Path,
        model_path: str | Path,
        config_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        vocab_path = Path(vocab_path)
        merges_path = Path(merges_path)
        model_path = Path(model_path)
        config_path = Path(config_path) if config_path else model_path.parent / "train_config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")

        self.tokenizer, self.eos_token_id = load_tokenizer(vocab_path, merges_path)
        config = load_model_config(config_path)
        if not config.vocab_size:
            config.vocab_size = infer_vocab_size(self.tokenizer)

        checkpoint = torch.load(model_path, map_location=self.device)
        model = Transformer_LM(
            int(config.vocab_size),
            int(config.context_length),
            int(config.d_model),
            int(config.num_layers),
            int(config.num_heads),
            int(config.d_ff),
            float(config.rope_theta),
            device=self.device,
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.context_length = config.context_length

    def generate_text(self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_p: float = 0.0) -> str:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        input_ids = self.tokenizer.encode(prompt)
        generated_ids = list(input_ids)

        self.model.eval()
        with torch.no_grad():
            for _ in range(max_length):
                context_ids = generated_ids[-self.context_length :]
                input_tensor = torch.tensor(context_ids, dtype=torch.long, device=self.device).unsqueeze(0)
                logits = self.model(input_tensor)
                next_token_logits = logits[0, -1, :] / temperature

                if top_p > 0.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float("inf")

                probabilities = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1).item()

                generated_ids.append(next_token_id)
                if self.eos_token_id is not None and next_token_id == self.eos_token_id:
                    break

        return self.tokenizer.decode(generated_ids)


inference = Inference


if __name__ == "__main__":
    tinystories_inference = Inference(
        vocab_path="models/tinystories_train_tokenizer/bpe_vocab.json",
        merges_path="models/tinystories_train_tokenizer/bpe_merges.txt",
        model_path="/home/hw/learn/llm-from-scratch-assignment1-basics/checkpoints/tinystories_exp3/checkpoint_10000.pt",
        device="cpu",
    )
    output = tinystories_inference.generate_text(
        prompt="Once upon a time in a land far away,",
        max_length=100,
        temperature=0.8,
        top_p=0.9,
    )
    print(output)
