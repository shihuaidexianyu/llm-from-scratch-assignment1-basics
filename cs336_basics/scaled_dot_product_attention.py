import torch
import math

"""
    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA
    """


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.size(-1)
    # 1. 计算注意力分数
    # scores: (..., queries, keys)
    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(d_k)

    # 2. 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))  # 对于掩码位置(False)，设置为负无穷

    # 3. 计算注意力权重
    attn_weights = torch.softmax(scores, dim=-1)

    # 4. 计算加权值
    output = torch.einsum("...qk,...kd->...qd", attn_weights, V)

    return output
