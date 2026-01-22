import torch
import torch.nn as nn
from torch import Tensor


class MyEmbedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        embedding layer
        """
        super().__init__()

        def _init_parameter(tensor: Tensor) -> None:
            nn.init.trunc_normal_(tensor, mean=0.0, std=1, a=-3 * 1, b=3 * 1)

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        _init_parameter(self.weight)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]
