from .scaled_dot_product_attention import scaled_dot_product_attention
from torch import nn
import torch
from .linear import MyLinear

"""
    Given the key, query, and value projection weights of a naive unbatched
    implementation of multi-head attention, return the output of an optimized batched
    implementation. This implementation should handle the key, query, and value projections
    for all heads in a single matrix multiply.
    This function should not use RoPE.
    See section 3.2.2 of Vaswani et al., 2017.

    Args:
        d_model (int): Dimensionality of the feedforward input and output.
        num_heads (int): Number of heads to use in multi-headed attention.
        max_seq_len (int): Maximum sequence length to pre-cache if your implementation does that.
        q_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the Q projection
        k_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the K projection
        v_proj_weight (Float[Tensor, "d_k d_in"]): Weights for the V projection
        o_proj_weight (Float[Tensor, "d_model d_v"]): Weights for the output projection
        in_features (Float[Tensor, "... sequence_length d_in"]): Tensor to run your implementation on.

    Returns:
        Float[Tensor, " ... sequence_length d_out"]: Tensor with the output of running your optimized, batched multi-headed attention
        implementation with the given QKV projection weights and input features.
"""


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = MyLinear(d_model, d_model)
        self.k_proj = MyLinear(d_model, d_model)
        self.v_proj = MyLinear(d_model, d_model)
        self.out_proj = MyLinear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool))
        mask = mask.view(1, 1, seq_length, seq_length)
        mask = mask.to(x.device)
        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        pos = torch.arange(seq_length, device=x.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, seq_length)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_proj(attn_output)
        return output


class MultiHeadSelfAttentionWithRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = MyLinear(d_model, d_model)
        self.k_proj = MyLinear(d_model, d_model)
        self.v_proj = MyLinear(d_model, d_model)
        self.out_proj = MyLinear(d_model, d_model)
        from .rope import RotaryPositionalEmbedding

        self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        seq_length = x.size(1)
        mask = torch.tril(torch.ones(seq_length, seq_length, device=x.device, dtype=torch.bool))
        mask = mask.view(1, 1, seq_length, seq_length)
        mask = mask.to(x.device)
        Q = self.q_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(batch_size, seq_length, self.num_heads, self.head_dim)
        pos = torch.arange(seq_length, device=x.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand(batch_size, seq_length)
        # 1. 变形: (Batch, Seq, Heads, Dim) -> (Batch, Heads, Seq, Dim) -> (Batch*Heads, Seq, Dim)
        # 注意：这里最好先 transpose 把 Heads 放到前面，这样内存是连续的逻辑
        Q_flat = Q.transpose(1, 2).reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        K_flat = K.transpose(1, 2).reshape(batch_size * self.num_heads, seq_length, self.head_dim)
        # 2. 扩展 Position: (Batch, Seq) -> (Batch*Heads, Seq)
        # 这一步是必须的！因为你的输入变成了 B*H 个样本，位置索引也要跟上
        pos_flat = pos.repeat_interleave(self.num_heads, dim=0)
        # 3. 传入未修改的 RoPE (它只接受 3D 输入)
        # RoPE 看到的是：我有 B*H 个句子，每个句子长 Seq，特征 Dim
        Q_rotated_flat = self.rope(Q_flat, pos_flat)
        K_rotated_flat = self.rope(K_flat, pos_flat)
        # 4. 拼回去: (Batch*Heads, Seq, Dim) -> (Batch, Heads, Seq, Dim)
        Q = Q_rotated_flat.view(batch_size, self.num_heads, seq_length, self.head_dim)
        K = K_rotated_flat.view(batch_size, self.num_heads, seq_length, self.head_dim)
        V = V.transpose(1, 2)
        attn_output = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        output = self.out_proj(attn_output)

        return output
