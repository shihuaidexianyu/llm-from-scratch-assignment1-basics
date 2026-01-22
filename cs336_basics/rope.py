import torch
import torch.nn as nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=torch.device | None):
        super().__init__()
        self.dim = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        # 1. 计算频率 (inv_freq)
        # 结果形状: (dim // 2,)
        # 这里的公式对应: theta ** -(2i / d)
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))

        # 将其注册为 buffer，这样它会随模型保存，但不会作为参数更新
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Dim)

        # 1. 计算频率 (逻辑不变)
        # 注意：这里需要确保维度对齐，einsum 写法最稳
        # freqs: (Batch, Seq, Dim/2)
        freqs = torch.einsum("..., d -> ...d", token_positions.float(), self.inv_freq)

        # 2. 准备 cos/sin (逻辑不变)
        cos = freqs.cos().to(x.dtype)
        sin = freqs.sin().to(x.dtype)

        # 3. 构造旋转矩阵 (逻辑不变)
        # rot_mat: (Batch, Seq, Dim/2, 2, 2)
        row0 = torch.stack([cos, -sin], dim=-1)
        row1 = torch.stack([sin, cos], dim=-1)
        rot_mat = torch.stack([row0, row1], dim=-2)

        # -----------------------------------------------------------
        # 4. 准备输入对 (修改这里！！！)
        # 改为【相邻配对】：将最后一维 reshape 成 (Dim/2, 2)
        # x_pairs: (Batch, Seq, Dim/2, 2)
        # -----------------------------------------------------------
        x_pairs = x.view(*x.shape[:-1], self.dim // 2, 2)

        # 5. Einsum 旋转 (逻辑不变)
        # rot_mat: ...ij (Batch, Seq, Dim/2, 2, 2)
        # x_pairs: ...j  (Batch, Seq, Dim/2, 2)
        # output:  ...i  (Batch, Seq, Dim/2, 2)
        x_rotated_pairs = torch.einsum("...ij, ...j -> ...i", rot_mat, x_pairs)

        # -----------------------------------------------------------
        # 6. 还原形状 (修改这里！！！)
        # 只需要把最后两维 flatten 回去即可
        # -----------------------------------------------------------
        x_rotated = x_rotated_pairs.flatten(-2)

        return x_rotated
