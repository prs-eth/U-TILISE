"""
Based on the following implementations:
https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/ltae.py
https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
"""

import copy
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .make_layers import get_activation, get_group_gn
from .parameters import LTAENormType
from .positional_encoding import PositionalEncoder


class LTAEtransformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        n_head: int = 4,
        d_k: int = 4,
        bias_qk: bool = False,
        positional_encoding: bool = True,
        T: int = 1000,
        mlp: List[int] = [128, 128],
        activation: str | Tuple[str, float] = 'relu',
        norm: Literal['group', 'layer'] = 'group',
        num_groups: int = 4,
        dim_per_group: int = -1,
        group_norm_eps: float = 1e-05,
        norm_first: bool = True,
        dropout: float = 0.1,
        attn_dropout: float = 0.1,
        return_att: bool = False
    ):
        """
        Transformer-inspired Lightweight Temporal Attention Encoder (L-TAE) for sequence-to-sequence modeling.
        A shared L-TAE is applied to all pixel positions of the image sequence.
        Args:
            in_channels:         int, number of channels of the input embeddings.
            n_head:              int, number of attention heads.
            d_k:                 int, dimension of the keys and queries.
            bias_qk:             bool, True to augment the keys and queries with a learned bias, False otherwise.
            positional_encoding: bool, activate (True) or deactivate (False) the sinusoidal positional encoding.
            T:                   int, period to be used for the sinusoidal positional encoding.
            mlp:                 list of int, dimensions of the MLP layers in the feed-forward block.
            activation:          str or (str, float), non-linear activation in the feed-forward block.
            norm:                str, type of normalization layer to use. Choose among:
                                    'group': GroupNorm
                                    'layer': LayerNorm
            num_groups:          int, number of groups in the GroupNorm layer. This parameter is mutually exclusive
                                 with the `dim_per_group` parameter.
                                 Specify num_groups = -1 to use `dim_per_group` instead.
            dim_per_group:       int, number of dimensions per group in the GroupNorm layer. This parameter is mutually
                                 exclusive with the `num_groups` parameter.
                                 Specify dim_per_group = -1 to use `num_groups` instead.
            group_norm_eps:      float, eps value used in the GroupNorm layer.
            norm_first:          bool, True to normalize the data before the multi-head self-attention and the
                                 feed-forward blocks (pre-norm transformer); False to normalize the data after the
                                 multi-head self-attention and the feed-forward blocks (post-norm transformer).
                                 Cf. Figure 1 in https://arxiv.org/pdf/2002.04745.pdf
            dropout:             float, dropout rate. Applied after the multi-head self-attention and the feed-forward
                                 blocks.
            attn_dropout:        float, dropout rate. Applied to the attention masks within the multi-head
                                 self-attention block.
            return_att:          bool, True to return the attention masks along with the output embeddings;
                                 False otherwise.
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_head = n_head
        self.d_k = d_k
        self.bias_qk = bias_qk
        self.mlp = copy.deepcopy(mlp)
        self.attn_dropout = attn_dropout
        self.activation = get_activation(activation)
        self.norm = LTAENormType(norm)
        self.norm_first = norm_first
        self.num_groups = num_groups
        self.dim_per_group = dim_per_group
        self.group_norm_eps = group_norm_eps
        self.return_att = return_att

        assert self.mlp[0] == self.in_channels

        if positional_encoding:
            self.positional_encoder = PositionalEncoder(
                self.in_channels // n_head, T=T, repeat=n_head
            )
        else:
            self.positional_encoder = None

        self.attention_heads = MultiHeadAttention(n_head=self.n_head, d_k=self.d_k, d_in=self.in_channels,
                                                  bias_qk=self.bias_qk, attn_dropout=self.attn_dropout)

        if self.norm == LTAENormType.GROUP:
            self.norm1 = nn.GroupNorm(num_channels=self.in_channels,
                                      num_groups=get_group_gn(self.in_channels, self.dim_per_group, self.num_groups),
                                      eps=self.group_norm_eps
                                      )
            self.norm2 = nn.GroupNorm(num_channels=mlp[-1],
                                      num_groups=get_group_gn(self.mlp[-1], self.dim_per_group, self.num_groups),
                                      eps=self.group_norm_eps
                                      )
        elif self.norm == LTAENormType.LAYER:
            self.norm1 = nn.LayerNorm(self.in_channels)
            self.norm2 = nn.LayerNorm(self.in_channels)
        else:
            NotImplementedError
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Feed-forward block
        layers = []
        for i in range(len(self.mlp) - 1):
            layers.extend(
                [
                    nn.Linear(self.mlp[i], self.mlp[i + 1]),
                    self.activation
                ]
            )

        self.mlp = nn.Sequential(*layers)

    def _sa_block(
            self, x: Tensor, pad_mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Sequence-to-sequence multi-head self-attention.

        Args:
            x:          (B x H x W) x T x C
            pad_mask:   (B x H x W) x T

        Returns:
            out:        (B x H x W x T) x C
            attn:       self.n_head x (B x H x W) x T x T
        """

        sz_b, seq_len, _ = x.shape
        out, attn = self.attention_heads(x, pad_mask=pad_mask)

        # Concatenate heads
        out = (out.permute(1, 2, 0, 3).contiguous().view(sz_b, seq_len, -1))  # (B x H x W) x T x C
        out = self.dropout1(out.view(sz_b * seq_len, -1))

        return out, attn

    def _ff_block(self, x: Tensor) -> Tensor:
        """Feed-forward block."""
        out = self.dropout2(self.mlp(x))
        return out

    def forward(
            self, x: Tensor, batch_positions: Optional[Tensor] = None, pad_mask: Optional[Tensor] = None
    ) -> Tensor | Tuple[Tensor, Tensor]:
        sz_b, seq_len, c, h, w = x.shape
        if pad_mask is not None:
            pad_mask = (
                pad_mask.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # B x T x H x W
            pad_mask = (
                pad_mask.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)  # (B x H x W) x T
            )

        # B x T x C x H x W -> (B x H x W) x T x C
        out = x.permute(0, 3, 4, 1, 2).contiguous().view(sz_b * h * w, seq_len, c)

        if self.positional_encoder is not None and batch_positions is not None:
            bp = (
                batch_positions.unsqueeze(-1)
                .repeat((1, 1, h))
                .unsqueeze(-1)
                .repeat((1, 1, 1, w))
            )  # B x T x H x W
            bp = bp.permute(0, 2, 3, 1).contiguous().view(sz_b * h * w, seq_len)  # (B x H x W) x T
            out = out + self.positional_encoder(bp)   # (B x H x W) x T x C

        if self.norm_first:
            out = out.view(sz_b * h * w * seq_len, -1)
            buffer, attn = self._sa_block(self.norm1(out).view(sz_b * h * w, seq_len, -1), pad_mask=pad_mask)
            out = out + buffer
            out = out + self._ff_block(self.norm2(out))
        else:
            buffer, attn = self._sa_block(out, pad_mask=pad_mask)
            out = self.norm1(out.view(sz_b * h * w * seq_len, -1) + buffer)
            out = self.norm2(out + self._ff_block(out))

        # B x T x C x H x W
        out = out.view(sz_b, h, w, seq_len, -1).permute(0, 3, 4, 1, 2)

        # n_head x B x T x T x H x W
        attn = attn.view(self.n_head, sz_b, h, w, seq_len, seq_len).permute(0, 1, 4, 5, 2, 3)

        if self.return_att:
            return out.contiguous(), attn.contiguous()
        return out.contiguous()


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, n_head: int, d_k: int, d_in: int, bias_qk: bool = True, attn_dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_in = d_in
        self.attn_dropout = attn_dropout

        self.fc_k = nn.Linear(d_in, n_head * d_k, bias=bias_qk)
        self.fc_q = nn.Linear(d_in, n_head * d_k, bias=bias_qk)
        nn.init.normal_(self.fc_k.weight, mean=0, std=np.sqrt(2.0 / d_k))
        nn.init.normal_(self.fc_q.weight, mean=0, std=np.sqrt(2.0 / d_k))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=self.attn_dropout)

    def forward(
            self, v: Tensor, pad_mask: Optional[Tensor] = None, return_comp: bool = False
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Optional[Tensor]]:
        d_k, d_in, n_head = self.d_k, self.d_in, self.n_head
        sz_b, seq_len, _ = v.size()

        # Perform linear operation and split into heads
        k = self.fc_k(v).view(sz_b, seq_len, n_head, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n_head x B x H x W) x T x d_k

        # One query per date and sequence
        q = self.fc_q(v).view(sz_b, seq_len, n_head, d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)  # (n_head x B x H x W) x T x d_k

        if pad_mask is not None:
            pad_mask = pad_mask.repeat(
                (n_head, 1)
            )  # replicate pad_mask for each head (nxb) x lk

        v = torch.stack(v.split(v.shape[-1] // n_head, dim=-1)).view(
            n_head * sz_b, seq_len, -1
        )   # (B x H x W) x T x C  ->  (n_head x B x H x W) x T x (C // n_head)
        if return_comp:
            output, attn, comp = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
        else:
            output, attn = self.attention(
                q, k, v, pad_mask=pad_mask, return_comp=return_comp
            )
            comp = None

        attn = attn.view(n_head, sz_b, seq_len, seq_len)                # n_head x (B x H x W) x T x T
        output = output.view(n_head, sz_b, seq_len, d_in // n_head)     # n_head x (B x H x W) x T x (C // n_head)

        if return_comp:
            return output, attn, comp
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention.

    Modified from github.com/jadore801120/attention-is-all-you-need-pytorch
    """

    def __init__(self, temperature: float, attn_dropout: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self, q: Tensor, k: Tensor, v: Tensor, pad_mask: Optional[Tensor] = None, return_comp: bool = False
    ) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, Optional[Tensor]]:
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if pad_mask is not None:
            attn = attn.masked_fill(pad_mask.unsqueeze(1), -1e3)
        comp = attn if return_comp else None

        attn = self.softmax(attn)        # attn: (n_head x B x H x W) x T_out x T_in
        attn = self.dropout(attn)
        output = torch.matmul(attn, v)   # v: (n_head x B x H x W) x T_in x (C // n_head)

        if return_comp:
            return output, attn, comp
        return output, attn
