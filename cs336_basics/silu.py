"""Given a tensor of inputs, return the output of applying SiLU
to each element.

Args:
    in_features(Float[Tensor, "..."]): Input features to run SiLU on. Shape is arbitrary.

Returns:
    Float[Tensor,"..."]: of with the same shape as `in_features` with the output of applying
    SiLU to each element.
"""

import torch


def silu(in_features: torch.Tensor) -> torch.Tensor:
    return in_features * torch.sigmoid(in_features)
