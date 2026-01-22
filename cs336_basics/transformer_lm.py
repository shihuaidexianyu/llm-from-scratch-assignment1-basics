"""Given the weights of a Transformer language model and input indices,
return the output of running a forward pass on the input indices.

This function should use RoPE.

Args:
    vocab_size (int): The number of unique items in the output vocabulary to be predicted.
    context_length (int): The maximum number of tokens to process at once.
    d_model (int): The dimensionality of the model embeddings and sublayer outputs.
    num_layers (int): The number of Transformer layers to use.
    num_heads (int): Number of heads to use in multi-headed attention. `d_model` must be
        evenly divisible by `num_heads`.
    d_ff (int): Dimensionality of the feed-forward inner layer (section 3.3).
    rope_theta (float): The RoPE $\\Theta$ parameter.
    weights (dict[str, Tensor]):
        State dict of our reference implementation. {num_layers} refers to an
        integer between `0` and `num_layers - 1` (the layer index).
        The keys of this dictionary are:
        - `token_embeddings.weight`
            Token embedding matrix. Shape is (vocab_size, d_model).
        - `layers.{num_layers}.attn.q_proj.weight`
            The query projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.q_proj.weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.k_proj.weight`
            The key projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_k),
            so `attn.k_proj.weight == torch.cat([k_heads.0.weight, ..., k_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.v_proj.weight`
            The value projections for all `num_heads` attention heads.
            Shape is (num_heads * (d_model / num_heads), d_model).
            The rows are ordered by matrices of shape (num_heads, d_v),
            so `attn.v_proj.weight == torch.cat([v_heads.0.weight, ..., v_heads.N.weight], dim=0)`.
        - `layers.{num_layers}.attn.output_proj.weight`
            Weight of the multi-head self-attention output projection
            Shape is ((d_model / num_heads) * num_heads, d_model).
        - `layers.{num_layers}.ln1.weight`
            Weights of affine transform for the first RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `layers.{num_layers}.ffn.w1.weight`
            Weight of the first linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `layers.{num_layers}.ffn.w2.weight`
            Weight of the second linear transformation in the FFN.
            Shape is (d_ff, d_model).
        - `layers.{num_layers}.ffn.w3.weight`
            Weight of the third linear transformation in the FFN.
            Shape is (d_model, d_ff).
        - `layers.{num_layers}.ln2.weight`
            Weights of affine transform for the second RMSNorm
            applied in the transformer block.
            Shape is (d_model,).
        - `ln_final.weight`
            Weights of affine transform for RMSNorm applied to the output of the final transformer block.
            Shape is (d_model, ).
        - `lm_head.weight`
            Weights of the language model output embedding.
            Shape is (vocab_size, d_model).
    in_indices (Int[Tensor, "batch_size sequence_length"]) Tensor with input indices to run the language model on. Shape is (batch_size, sequence_length), where
        `sequence_length` is at most `context_length`.

Returns:
    Float[Tensor, "batch_size sequence_length vocab_size"]: Tensor with the predicted unnormalized
    next-word distribution for each token.
"""

from .transformer_block import TransformerBlock
from torch import Tensor
import torch
import torch.nn as nn
from .embedding import MyEmbedding
from .rmsnorm import MyRMSNorm
from .linear import MyLinear


class Transformer_LM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.token_embeddings = MyEmbedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, rope_theta, context_length) for _ in range(num_layers)]
        )
        self.ln_final = MyRMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = MyLinear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
