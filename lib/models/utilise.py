"""
Based on https://github.com/VSainteuf/utae-paps/blob/main/src/backbones/utae.py

Garnot, V. S. F., Landrieu, L., 2021. Panoptic segmentation of satellite image time series with convolutional temporal
attention networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 4872-4881).
"""

from typing import List, Literal, Optional, Tuple

import torch
from torch import Tensor, nn

from .ltae_transformer import LTAEtransformer
from .make_layers import get_activation, get_group_gn, str2ActivationType
from .parameters import (
    ActivationType,
    NormType,
    TemporalAggregationMode,
    UpConvType
)


class UTILISE(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: Optional[int] = None,
            encoder_widths: List[int] = [64, 64, 64, 128],
            decoder_widths: List[int] = [32, 32, 64, 128],
            upconv_type: Literal['transpose', 'bilinear'] = 'transpose',
            encoder_norm: Optional[Literal['group', 'batch', 'instance']] = None,
            decoder_norm: Optional[Literal['group', 'batch', 'instance']] = None,
            skip_norm: Optional[Literal['group', 'batch', 'instance']] = None,
            activation: str | Tuple[str, float] = 'relu',
            str_conv_k: int = 4,
            str_conv_s: int = 2,
            str_conv_p: int = 1,
            str_conv_k_up: Optional[int] = 2,
            str_conv_p_up: Optional[int] = 0,
            padding_mode: str = 'reflect',
            skip_attention: bool = False,
            positional_encoding: bool = True,
            n_temporal_encoding_layers: int = 1,
            agg_mode: Optional[Literal['att_group', 'att_mean']] = 'att_group',
            n_head: int = 4,
            d_k: int = 4,
            bias_qk: bool = False,
            attn_dropout: float = 0.1,
            dropout: float = 0.1,
            n_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05,
            ltae_norm: str = 'group',
            ltae_activation: str | Tuple[str, float] = 'relu',
            norm_first: bool = True,
            pad_value: Optional[float] = 0,
            output_activation: str | Tuple[str, float] | bool | None = 'sigmoid',
            return_maps: bool = False
    ):
        """
        U-TILISE (U-Net Temporal Imputation Lightweight Image Sequence Encoder) architecture for spatio-temporal
        sequence modeling of satellite image time series.
        Args:
            input_dim:           int, number of channels in the input image time series.
            output_dim:          int, number of channels in the output image time series.
            encoder_widths:      list of int, list specifying the channel dimension per encoding level of the UNet.
                                 The channel dimensions are given from top to bottom, i.e., from the highest to the
                                 lowest spatial resolution. `encoder_widths[0]` denotes the channel dimension of the
                                 input convolutional block and `encoder_widths[1:]` the channel dimension(s) of the
                                 DownConvBlock block(s). I.e., this argument also defines the number of downsampling
                                 operations, where the number of downsampling operations is equal to
                                 len(encoder_widths) - 1.
            decoder_widths:      list of int, list specifying the channel dimension per decoding level of the UNet.
                                 The channel dimensions are given from top to bottom, i.e., from the highest to the
                                 lowest spatial resolution. If this argument is not specified, the channel dimensions
                                 of the decoding layers are symmetric to the ones of the corresponding encoding layers
                                 (i.e., equal to `encoder_widths`).
            upconv_type:         str, upsampling method. Choose among:
                                    'transpose': transpose convolution
                                    'bilinear':  bilinear upsampling followed by convolution
            encoder_norm:        str, type of normalization layer to use in the encoding branch. Choose among:
                                    'group':    GroupNorm
                                    'batch':    BatchNorm
                                    'instance': InstanceNorm
                                    None:       without normalization layer
            decoder_norm:        str, type of normalization layer to use in the decoding branch.
                                 Same options as for `encoder_norm`.
            skip_norm:           str, type of normalization layer to use in the UNet skip connections.
                                 Same options as for `encoder_norm`.
            activation:          str or (str, float), non-linear activation type.
            str_conv_k:          int, kernel size of the strided convolutions.
            str_conv_s:          int, stride of the strided convolutions.
            str_conv_p:          int, padding of the strided convolutions.
            str_conv_k_up:       int, kernel size of the transpose convolutions. If None, `str_conv_k_up` is
                                 set to `str_conv_k`.
            str_conv_p_up:       int, padding of the transpose convolutions. If None, `str_conv_p_up` is set to `str_conv_p`.
            padding_mode:        str, spatial padding strategy for convolutional layers (passed to nn.Conv2d).
            skip_attention:      bool; if True, the model is a standard UNet without temporal attention. If False,
                                 the bottleneck of the UNet is equipped with a sequence-to-sequence variant of the
                                 L-TAE temporal encoding module (use `agg_mode` to specify the temporal weighting
                                 strategy applied in the UNet skip connections).
            positional_encoding: bool, activate (True) or deactivate (False) sinusoidal positional encoding.
            n_temporal_encoding_layers:  int, number of temporal encoding layers in the UNet bottleneck.
            agg_mode:            str, aggregation mode used in the UNet skip connections. Choose among:
                                    'att_group':   Attention weighted temporal feature maps, using the same channel
                                                   grouping strategy as in the seq2seq L-TAE. The attention masks are
                                                   bilinearly resampled to the resolution of the skipped feature maps.
                                    'att_mean':    Attention weighted temporal feature maps, using the average attention
                                                   scores across heads for each time step.
                                    None:          Default UNet skip connection without temporal weighting.
            n_head:              int, number of attention heads in the seq2seq L-TAE module.
            d_k:                 int, dimension of the keys and queries.
            bias_qk:             bool, True to augment the keys and queries with a learned bias, False otherwise.
            attn_dropout:        float, dropout rate. Applied to the attention masks within the multi-head
                                 self-attention block.
            dropout:             float, dropout rate. Applied after the multi-head self-attention and the feed-forward
                                 blocks.
            n_groups:            int, number of groups in the GroupNorm layer. This parameter is mutually exclusive
                                 with the `dim_per_group` parameter.
                                 Specify num_groups = -1 to use `dim_per_group` instead.
            dim_per_group:       int, number of dimensions per group in the GroupNorm layer. This parameter is mutually
                                 exclusive with the `num_groups` parameter.
                                 Specify dim_per_group = -1 to use `num_groups` instead.
            group_norm_eps:      float, eps value used in the GroupNorm layer.
            ltae_norm:           str, type of normalization layer to use in the seq2seq L-TAE module. Choose among:
                                    'group':    GroupNorm
                                    'layer':    LayerNorm
            norm_first:          bool, True to normalize the data before the multi-head self-attention and the
                                 feed-forward blocks (pre-norm transformer); False to normalize the data after the
                                 multi-head self-attention and the feed-forward blocks (post-norm transformer).
                                 Cf. Figure 1 in https://arxiv.org/pdf/2002.04745.pdf
            pad_value:           float, value used by the data loader for temporal padding.
            output_activation:   str, type of non-linear activation after the last convolutional layer in the decoder.
                                 If None, `output_activation` is set to `activation`. If False, identity mapping is
                                 applied (i.e., no output activation).
            return_maps:         bool, True to additionally return the feature maps, False otherwise.
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        self.n_stages = len(encoder_widths)
        self.return_maps = return_maps
        self.encoder_widths = encoder_widths

        if decoder_widths is not None:
            self.decoder_widths = decoder_widths
            assert len(self.encoder_widths) == len(self.decoder_widths)
            assert self.encoder_widths[-1] == self.decoder_widths[-1]
        else:
            self.decoder_widths = self.encoder_widths

        self.out_conv_widths = [self.decoder_widths[0], self.output_dim]
        self.str_conv_k = str_conv_k
        self.str_conv_s = str_conv_s
        self.str_conv_p = str_conv_p
        self.agg_mode = TemporalAggregationMode(agg_mode)
        self.upconv_type = UpConvType(upconv_type)
        self.encoder_norm = NormType(encoder_norm)
        self.decoder_norm = NormType(decoder_norm)
        self.skip_norm = NormType(skip_norm)
        self.activation = str2ActivationType(activation)

        if output_activation is None or output_activation is True:
            self.output_activation = self.activation
        elif output_activation is False:
            self.output_activation = None
        else:
            self.output_activation = str2ActivationType(output_activation)

        self.n_head = n_head
        self.d_k = d_k
        self.bias_qk = bias_qk
        self.attn_dropout = attn_dropout
        self.dropout = dropout
        self.pad_value = pad_value
        self.padding_mode = padding_mode
        self.skip_attention = skip_attention
        self.positional_encoding = positional_encoding
        self.n_temporal_encoding_layers = n_temporal_encoding_layers
        self.n_groups = n_groups
        self.dim_per_group = dim_per_group
        self.group_norm_eps = group_norm_eps
        self.ltae_norm = ltae_norm
        self.ltae_activation = ltae_activation
        self.str_conv_k_up = str_conv_k_up if str_conv_k_up is not None else self.str_conv_k
        self.str_conv_p_up = str_conv_p_up if str_conv_p_up is not None else self.str_conv_p
        self.norm_first = norm_first

        if self.skip_attention:
            self.agg_mode = TemporalAggregationMode.NONE

        self.in_conv = ConvBlock(
            n_kernels=[self.input_dim] + [self.encoder_widths[0], self.encoder_widths[0]],
            pad_value=self.pad_value,
            norm=self.encoder_norm,
            num_groups=self.n_groups,
            dim_per_group=self.dim_per_group,
            group_norm_eps=self.group_norm_eps,
            activation=self.activation,
            padding_mode=self.padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                d_in=self.encoder_widths[i],
                d_out=self.encoder_widths[i + 1],
                k=self.str_conv_k,
                s=self.str_conv_s,
                p=self.str_conv_p,
                pad_value=self.pad_value,
                norm=self.encoder_norm,
                num_groups=self.n_groups,
                dim_per_group=self.dim_per_group,
                group_norm_eps=self.group_norm_eps,
                activation=self.activation,
                padding_mode=self.padding_mode,
            )
            for i in range(self.n_stages - 1)
        )
        self.up_blocks = nn.ModuleList(
            UpConvBlock(
                d_in=self.decoder_widths[i],
                d_out=self.decoder_widths[i - 1],
                d_skip=self.encoder_widths[i - 1],
                k=self.str_conv_k_up,
                s=self.str_conv_s,
                p=self.str_conv_p_up,
                upconv_type=self.upconv_type,
                norm_conv=self.decoder_norm,
                norm_skip=self.skip_norm,
                num_groups=self.n_groups,
                dim_per_group=self.dim_per_group,
                group_norm_eps=self.group_norm_eps,
                activation=self.activation,
                pad_value=self.pad_value,
                padding_mode=self.padding_mode
            )
            for i in range(self.n_stages - 1, 0, -1)
        )
        if self.skip_attention is False:
            layers = []

            for _ in range(self.n_temporal_encoding_layers):
                t_encoder = LTAEtransformer(
                    in_channels=self.encoder_widths[-1],
                    n_head=self.n_head,
                    d_k=self.d_k,
                    bias_qk=self.bias_qk,
                    positional_encoding=self.positional_encoding,
                    mlp=[self.encoder_widths[-1], self.encoder_widths[-1]],
                    activation=self.ltae_activation,
                    norm=self.ltae_norm,
                    num_groups=self.n_head,
                    dim_per_group=-1,
                    group_norm_eps=self.group_norm_eps,
                    norm_first=self.norm_first,
                    dropout=self.dropout,
                    attn_dropout=self.attn_dropout,
                    return_att=True
                )
                layers.append(t_encoder)

            self.temporal_encoder = nn.Sequential(*layers)

        self.temporal_aggregator = TemporalAggregator(mode=self.agg_mode)
        self.out_conv = ConvBlock(
            n_kernels=[self.decoder_widths[0]] + self.out_conv_widths,
            pad_value=pad_value,
            norm=self.decoder_norm,
            num_groups=self.n_groups,
            dim_per_group=self.dim_per_group,
            group_norm_eps=self.group_norm_eps,
            activation=self.activation,
            padding_mode=self.padding_mode,
            activation_last_layer=self.output_activation
        )

    def forward(
            self, x: Tensor, batch_positions: Optional[Tensor] = None, return_att: bool = False
    ) -> Tensor | Tuple[Tensor, Tensor] | Tuple[Tensor, Optional[List[Tensor]]] | Tuple[
        Tensor, Optional[Tensor], Optional[List[Tensor]]
    ]:
        pad_mask = (
            (x == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
        )  # BxT pad mask

        # Fully masked frames should not be treated as padded frames
        if batch_positions is not None:
            pad_mask = torch.logical_and(pad_mask, batch_positions == 0)  # BxT pad mask

        out = self.in_conv.smart_forward(x, pad_mask=pad_mask)
        feature_maps = [out]
        # SPATIAL ENCODER
        for i in range(self.n_stages - 1):
            out = self.down_blocks[i].smart_forward(feature_maps[-1], pad_mask=pad_mask)
            feature_maps.append(out)
        # TEMPORAL ENCODER
        if self.skip_attention:
            att = None
        else:
            for layer in self.temporal_encoder:
                # att.shape: n_head x B x T x T x h x w
                out, att = layer(out, batch_positions=batch_positions, pad_mask=pad_mask)

        # SPATIAL DECODER
        maps = [out] if self.return_maps else None

        for i in range(self.n_stages - 1):
            skip = self.temporal_aggregator(
                feature_maps[-(i + 2)], pad_mask=pad_mask, attn_mask=att
            )
            out = self.up_blocks[i](out, skip)
            if self.return_maps:
                # pyre-ignore[16]: `Optional` has no attribute `append`.
                maps.append(out)

        out = self.out_conv.smart_forward(out, pad_mask=pad_mask)
        if return_att and self.return_maps:
            return out, att, maps
        if return_att:
            return out, att
        if self.return_maps:
            return out, maps
        return out


class TemporallySharedBlock(nn.Module):
    """
    Helper module for convolutional encoding blocks that are shared across a sequence.
    This module adds the self.smart_forward() method to a block of network layers.
    smart_forward will combine the batch and temporal dimension of an input tensor
    if it is 5-D and apply the shared convolutions to all the (batch x temp) positions.
    """

    def __init__(self, pad_value: Optional[float] = None):
        super().__init__()
        self.out_shape = None
        self.pad_value = pad_value

    def smart_forward(self, x: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        if len(x.shape) == 4:
            return self.forward(x)

        b, t, c, h, w = x.shape

        if self.pad_value is not None:
            dummy = torch.zeros(x.shape, device=x.device).float()
            self.out_shape = self.forward(dummy.view(b * t, c, h, w)).shape
            del dummy

        out = x.view(b * t, c, h, w)
        if self.pad_value is not None:
            if pad_mask is None:
                pad_mask = (out == self.pad_value).all(dim=-1).all(dim=-1).all(dim=-1)
            elif len(pad_mask.shape) == 2:
                pad_mask = pad_mask.view(b * t)

            if pad_mask.any():
                temp = (
                        torch.ones(
                            self.out_shape, device=x.device, requires_grad=False
                        )
                        * self.pad_value
                )
                temp[~pad_mask] = self.forward(out[~pad_mask])
                out = temp
            else:
                out = self.forward(out)
        else:
            out = self.forward(out)
        _, c, h, w = out.shape
        out = out.view(b, t, c, h, w)
        return out


class ConvLayer(nn.Module):
    def __init__(
            self,
            n_kernels: List[int],
            norm: NormType = NormType.BATCH,
            activation: ActivationType | Tuple[ActivationType, float] = ActivationType.RELU,
            k: int = 3,
            s: int = 1,
            p: int = 1,
            num_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05,
            padding_mode: str = 'reflect',
            activation_last_layer: ActivationType | Tuple[ActivationType, float] | bool | None = True
    ):
        super().__init__()
        layers = []
        if norm == NormType.BATCH:
            nl = nn.BatchNorm2d
        elif norm == NormType.INSTANCE:
            nl = nn.InstanceNorm2d
        elif norm == NormType.GROUP:
            nl = lambda num_features: nn.GroupNorm(
                num_channels=num_features,
                num_groups=get_group_gn(num_features, dim_per_group, num_groups),
                eps=group_norm_eps
            )
        else:
            nl = None

        activation_last: Optional[ActivationType | Tuple[ActivationType, float]] = None

        if activation_last_layer is True:
            activation_last = activation
        elif activation_last_layer is False:
            pass
        elif activation_last_layer is not None:
            activation_last = activation_last_layer

        for i in range(len(n_kernels) - 1):
            layers.append(
                nn.Conv2d(
                    in_channels=n_kernels[i],
                    out_channels=n_kernels[i + 1],
                    kernel_size=k,
                    padding=p,
                    stride=s,
                    padding_mode=padding_mode,
                )
            )
            if nl is not None:
                layers.append(nl(n_kernels[i + 1]))

            if i < len(n_kernels) - 2:
                layers.append(get_activation(activation))
            elif activation_last is not None:
                layers.append(get_activation(activation_last))

        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class UpConvLayer(TemporallySharedBlock):
    def __init__(
            self,
            n_kernels: List[int],
            pad_value: Optional[float] = None,
            norm: NormType = NormType.BATCH,
            upconv_type: UpConvType = UpConvType.TRANSPOSE,
            activation: ActivationType | Tuple[ActivationType, float] = ActivationType.RELU,
            k: int = 4,
            s: int = 2,
            p: int = 1,
            num_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05,
            activation_last_layer: bool = True
    ):
        super().__init__(pad_value=pad_value)
        layers = []
        if norm == NormType.BATCH:
            nl = nn.BatchNorm2d
        elif norm == NormType.INSTANCE:
            nl = nn.InstanceNorm2d
        elif norm == NormType.GROUP:
            nl = lambda num_features: nn.GroupNorm(
                num_channels=num_features,
                num_groups=get_group_gn(num_features, dim_per_group, num_groups),
                eps=group_norm_eps
            )
        else:
            nl = None

        if upconv_type == UpConvType.TRANSPOSE:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=n_kernels[0],
                    out_channels=n_kernels[1],
                    kernel_size=k,
                    stride=s,
                    padding=p
                )
            )
        elif upconv_type == UpConvType.BILINEAR:
            layers.append(nn.Upsample(mode='bilinear', scale_factor=2))
            layers.append(nn.Conv2d(n_kernels[0], n_kernels[1], kernel_size=1, stride=1))

        if nl is not None:
            layers.append(nl(n_kernels[-1]))
        if activation_last_layer:
            layers.append(get_activation(activation))

        self.convtrans = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.convtrans(x)


class ConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            n_kernels: List[int],
            k: int = 3,
            p: int = 1,
            pad_value: Optional[float] = None,
            norm: NormType = NormType.BATCH,
            activation: ActivationType | Tuple[ActivationType, float] = ActivationType.RELU,
            activation_last_layer: ActivationType | Tuple[ActivationType, float] | bool | None = True,
            padding_mode: str = 'reflect',
            num_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05
    ):
        super().__init__(pad_value=pad_value)
        self.conv = ConvLayer(
            n_kernels=n_kernels,
            norm=norm,
            activation=activation,
            k=k,
            p=p,
            activation_last_layer=activation_last_layer,
            padding_mode=padding_mode,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class DownConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            k: int,
            s: int,
            p: int,
            pad_value: Optional[float] = None,
            norm: NormType = NormType.BATCH,
            num_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05,
            activation: ActivationType | Tuple[ActivationType, float] = ActivationType.RELU,
            padding_mode: str = 'reflect'
    ):
        super().__init__(pad_value=pad_value)
        self.down = ConvLayer(
            n_kernels=[d_in, d_in],
            norm=norm,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation,
            k=k,
            s=s,
            p=p,
            padding_mode=padding_mode
        )
        self.conv1 = ConvLayer(
            n_kernels=[d_in, d_out],
            norm=norm,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation,
            padding_mode=padding_mode
        )
        self.conv2 = ConvLayer(
            n_kernels=[d_out, d_out],
            norm=norm,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation,
            padding_mode=padding_mode,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.down(x)
        out = self.conv1(out)
        out = out + self.conv2(out)
        return out


class UpConvBlock(TemporallySharedBlock):
    def __init__(
            self,
            d_in: int,
            d_out: int,
            k: int,
            s: int,
            p: int,
            d_skip: Optional[int] = None,
            upconv_type: UpConvType = UpConvType.TRANSPOSE,
            norm_conv: NormType = NormType.BATCH,
            norm_skip: NormType = NormType.BATCH,
            num_groups: int = 4,
            dim_per_group: int = -1,
            group_norm_eps: float = 1e-05,
            activation: ActivationType | Tuple[ActivationType, float] = ActivationType.RELU,
            padding_mode: str = 'reflect',
            pad_value: Optional[float] = None
    ):
        super().__init__(pad_value=pad_value)
        d = d_out if d_skip is None else d_skip

        self.skip_conv = ConvBlock(
            n_kernels=[d, d],
            pad_value=pad_value,
            k=1,
            p=0,
            norm=norm_skip,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation
        )

        self.up = UpConvLayer(
            n_kernels=[d_in, d_out],
            pad_value=pad_value,
            k=k,
            s=s,
            p=p,
            norm=norm_conv,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            upconv_type=upconv_type,
            activation=activation
        )

        self.conv1 = ConvBlock(
            n_kernels=[d_out + d, d_out],
            pad_value=pad_value,
            norm=norm_conv,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation,
            padding_mode=padding_mode
        )

        self.conv2 = ConvBlock(
            n_kernels=[d_out, d_out],
            pad_value=pad_value,
            norm=norm_conv,
            num_groups=num_groups,
            dim_per_group=dim_per_group,
            group_norm_eps=group_norm_eps,
            activation=activation,
            padding_mode=padding_mode
        )

    def forward(self, x: Tensor, skip: Tensor, pad_mask: Optional[Tensor] = None) -> Tensor:
        out = self.up.smart_forward(x, pad_mask=pad_mask)
        out = torch.cat([out, self.skip_conv.smart_forward(skip, pad_mask=pad_mask)], dim=2)
        out = self.conv1.smart_forward(out, pad_mask=pad_mask)
        out = out + self.conv2.smart_forward(out, pad_mask=pad_mask)
        return out


class TemporalAggregator(nn.Module):
    def __init__(self, mode: TemporalAggregationMode = TemporalAggregationMode.ATT_GROUP):
        super().__init__()
        self.mode = mode

    def forward(self, x: Tensor, pad_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None) -> Tensor:

        if self.mode is TemporalAggregationMode.NONE:
            return x

        if pad_mask is not None and pad_mask.any() and attn_mask is not None:
            if self.mode == TemporalAggregationMode.ATT_GROUP:
                n_heads, b, t, _, h, w = attn_mask.shape
                attn = attn_mask.view(n_heads * b * t, t, h, w)

                if x.shape[-2] > w:
                    attn = nn.Upsample(
                        size=tuple(x.shape[-2:]), mode='bilinear', align_corners=False
                    )(attn)
                else:
                    attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)

                attn = attn.view(n_heads, b, t, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[None, :, None, :, None, None]

                out = torch.stack(x.chunk(n_heads, dim=2))  # n_heads x B x T x (C/n_heads) x H x W
                out = attn[:, :, :, :, None, :, :] * out[:, :, None, :, :, :, :]
                out = out.sum(dim=3)  # n_heads x B x T x (C/n_heads) x H x W
                out = torch.cat([group for group in out], dim=2)  # B x T x C x H x W
                return out
            if self.mode == TemporalAggregationMode.ATT_MEAN:
                n_heads, b, t, _, h, w = attn_mask.shape
                attn = attn_mask.mean(dim=0)  # average over heads -> B x T x T x H x W
                attn = attn.view(b * t, t, h, w)

                attn = nn.Upsample(
                    size=tuple(x.shape[-2:]), mode='bilinear', align_corners=False
                )(attn)

                attn = attn.view(b, t, t, *x.shape[-2:])
                attn = attn * (~pad_mask).float()[:, None, :, None, None]
                out = (x[:, None, :, :, :, :] * attn[:, :, :, None, :, :]).sum(dim=2)
                return out

        else:
            if attn_mask is not None:
                if self.mode == TemporalAggregationMode.ATT_GROUP:
                    n_heads, b, t, _, h, w = attn_mask.shape
                    attn = attn_mask.view(n_heads * b * t, t, h, w)
                    if x.shape[-2] > w:
                        attn = nn.Upsample(
                            size=tuple(x.shape[-2:]), mode='bilinear', align_corners=False
                        )(attn)
                    else:
                        attn = nn.AvgPool2d(kernel_size=w // x.shape[-2])(attn)
                    attn = attn.view(n_heads, b, t, t, *x.shape[-2:])  # n_heads x B x T x T x H x W
                    out = torch.stack(x.chunk(n_heads, dim=2))  # n_heads x B x T x (C/n_heads) x H x W
                    out = attn[:, :, :, :, None, :, :] * out[:, :, None, :, :, :, :]
                    out = out.sum(dim=3)  # n_heads x B x T x (C/n_heads) x H x W
                    out = torch.cat([group for group in out], dim=2)  # -> B x T x C x H x W
                    return out
                if self.mode == TemporalAggregationMode.ATT_MEAN:
                    n_heads, b, t, _, h, w = attn_mask.shape
                    attn = attn_mask.mean(dim=0)  # average over heads -> B x T x T x H x W
                    attn = attn.view(b * t, t, h, w)

                    attn = nn.Upsample(
                        size=tuple(x.shape[-2:]), mode='bilinear', align_corners=False
                    )(attn)

                    attn = attn.view(b, t, t, *x.shape[-2:])
                    out = (x[:, None, :, :, :, :] * attn[:, :, :, None, :, :]).sum(dim=2)
                    return out

        return x
