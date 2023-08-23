import torch
from torch import Tensor, nn


class PositionalEncoder(nn.Module):
    """
    Modified from https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/positional_encoding.py
    """
    def __init__(self, d: int, T: int = 1000, repeat: int | None = None, offset: int = 0):
        super().__init__()
        self.d = d
        self.T = T
        self.repeat = repeat
        self.denom = torch.pow(
            T, 2 * torch.div(torch.arange(offset, offset + d).float(), 2, rounding_mode='floor') / d
        )
        self.updated_location = False

    def forward(self, batch_positions: Tensor) -> Tensor:
        if not self.updated_location:
            self.denom = self.denom.to(batch_positions.device)
            self.updated_location = True

        # B x T x C, where B is equal to batch_size * H * W
        sinusoid_table = (
            batch_positions[:, :, None] / self.denom[None, None, :]
        )
        sinusoid_table[:, :, 0::2] = torch.sin(sinusoid_table[:, :, 0::2])  # dim 2i
        sinusoid_table[:, :, 1::2] = torch.cos(sinusoid_table[:, :, 1::2])  # dim 2i+1

        if self.repeat is not None:
            sinusoid_table = torch.cat(
                [sinusoid_table for _ in range(self.repeat)], dim=-1
            )

        return sinusoid_table
