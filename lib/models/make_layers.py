from typing import Tuple

from torch import nn

from .parameters import ActivationType


def str2ActivationType(activation: str | Tuple[str, float]) -> ActivationType | Tuple[ActivationType, float]:

    if isinstance(activation, str):
        activation = activation.replace("(", '').replace(")", '').split(',')

    if len(activation) == 2:
        return ActivationType(activation[0]), float(activation[1])
    return ActivationType(activation[0])


def get_activation(activation: ActivationType | Tuple[ActivationType, float] | str | Tuple[str, float]):
    """
    activation:  str or tuple (str, float)
                 Examples: 'relu', ('lrelu', 0.1), 'prelu', 'gelu', 'mish'
    """

    if isinstance(activation, str) or (isinstance(activation, tuple) and isinstance(activation[0], str)):
        # Conversion:
        # a) string -> ActivationType
        # b) (str, float) -> (ActivationType, float)
        activation = str2ActivationType(activation)

    if isinstance(activation, tuple) and activation[0] == ActivationType.LRELU:
        return nn.LeakyReLU(activation[1]) if isinstance(activation[1], float) else nn.LeakyReLU(0.01)

    activation_list = nn.ModuleDict({
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU(0.01),
        'prelu': nn.PReLU(),
        'mish': nn.Mish(),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    })

    return activation_list[activation.value]


def get_group_gn(dim, dim_per_gp, num_groups):
    """
    Get the number of groups used by GroupNorm, based on the number of channels.

    Example 1:
    get_group_gn(64, -1, 32) returns 32, i.e., the channel dimension will be divided into 32 groups

    Example 2:
    get_group_gn(64, 4, -1) returns 16, i.e., the channel dimension will be divided into 16 groups with 4 channels
    per group

    Modified from:
    https://github.com/facebookresearch/MultiplexedOCR/blob/0ad0c3fb099082e944f2711a00a5a7b9e7ffea5c/multiplexer/modeling/make_layers.py#L14
    """

    assert (dim_per_gp == -1 and num_groups != -1) or (dim_per_gp != -1 and num_groups == -1), \
        "GroupNorm: can only specify G or C/G."

    # Use only one group if the given number of groups is greater than the number of channels
    if num_groups != -1 and dim <= num_groups:
        num_groups = 1
    if dim_per_gp != -1 and dim <= dim_per_gp:
        dim_per_gp = dim

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, f"dim: {dim}, dim_per_gp: {dim_per_gp}"
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, f"dim: {dim}, num_groups: {num_groups}"
        group_gn = num_groups

    return group_gn
