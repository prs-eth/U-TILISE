from typing import Literal, Optional, Tuple

import numpy as np
import torch
from kornia import morphology as morph
from torch import Tensor


def dilate_masks(
        masks: Tensor, kernel: Tensor = Tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]]), iterations: int = 1
) -> Tensor:
    """
    Returns the dilated `masks` using the kernel `kernel`. Dilation is performed `iterations` times.

    Args:
        masks:      torch.Tensor, input mask(s), (B x C x H x W) or (C x H x W).
        kernel:     torch.Tensor, kernel.
        iterations: int, number of iterations.

    Returns:
        masks:      torch.Tensor, dilated mask(s), (B x C x H x W) or (C x H x W).
    """

    if masks.dim() == 3:
        masks = masks.unsqueeze(0)
        reduce_dim = True
    else:
        reduce_dim = False

    for _ in range(iterations):
        masks = morph.dilation(masks, kernel)

    return masks.squeeze(0) if reduce_dim else masks


def masks_init_filling(
        seq: Tensor,
        masks: Tensor,
        mask_valid: Optional[Tensor] = None,
        fill_type: Literal['fill_value', 'white_noise', 'mean'] = 'fill_value',
        fill_value: float = 0,
        dilate_cloud_masks: Optional[bool] = False
) -> Tuple[Tensor, Tensor]:
    """
    Overlays the input satellite image time series `seq` with the sequenc of masks `masks`. The pixel values of masked
    pixels are defined by the strategy `fill_type`.

    Args:
        seq:                torch.Tensor, satellite image time series to be masked, (T x C x H x W).
        masks:              torch.Tensor, mask sequence, (T x 1 x H x W), where a pixel value of 1 indicates a masked
                            pixel and 0 an unmasked pixel.
        mask_valid:         torch.Tensor, flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid.
        fill_type:          str, strategy used to initialize masked pixels. Choose among:
                                'fill_value':   Fill masked pixels with the value specified by `fill_val`.
                                'white_noise':  White noise per channel with mean 0 and standard deviation 0.5.
                                'mean':         Mean value per channel.
        fill_value:         float, pixel value assigned to masked pixels (used if fill_type == 'fill_value').
        dilate_cloud_masks: bool, True to dilate the cloud masks before masking; False to use the original cloud mask
                            shapes.

    Returns:
        masked_seq:  torch.Tensor, masked satellite image time series, where masked pixel have been replaced by the
                     value `fill_type`.
        masks:       torch.Tensor, sequence of cloud masks.
    """

    assert fill_type in ['fill_value', 'white_noise', 'mean'], 'Invalid mask initialization.'

    if dilate_cloud_masks:
        masks = dilate_masks(masks)

    num_channels = seq.shape[1]
    masked_seq = seq.clone()
    flag = (masks == 1.).expand_as(seq)

    if fill_type == 'fill_value':
        masked_seq[flag] = fill_value

    elif fill_type == 'white_noise':
        noise = torch.normal(mean=0, std=0.5, size=seq.shape)
        noise[noise < 0] = 0
        noise[noise > 1] = 1
        masked_seq[flag] += noise[flag]

    elif fill_type == 'mean':
        for c in range(num_channels):
            # Compute the mean per channel (across all time steps)
            masked_seq[:, c, :, :][flag[:, 0, :, :]] = \
                torch.mean(seq[mask_valid == 1, c, :, :][(~flag)[mask_valid == 1, 0, :, :]])
    else:
        raise NotImplementedError(f'Unknown mask fill type: {fill_type}\n')

    return masked_seq, masks


def overlay_seq_with_clouds(
        images: Tensor, cloud_masks: Tensor, t_masked: np.ndarray | None = None, fill_value: int = 0,
        dilate_cloud_masks: Optional[bool] = False
) -> Tuple[Tensor, Tensor]:
    """
    Masks the given satellite image time series `images` with cloud masks stored in `cloud_masks`.

    Args:
        images:             torch.Tensor, (T1 x C x H x W), input satellite image time series.
        cloud_masks:        torch.Tensor, (T2 x 1 x H x W), masks, where T2 denotes the number of images to be masked
                            (T2 <= T1).
        t_masked:           np.ndarray, (T2, ), time steps to be masked; time indices are given w.r.t. `images`.
        fill_value:         int, pixel value for masked pixels.
        dilate_cloud_masks: bool, True to dilate the cloud masks before masking; False to use the original cloud mask
                            shapes.

    Returns:
        images_masked:      torch.Tensor, (T1 x C x H x W), masked satellite image time series.
        masks:              torch.Tensor, (T1 x C x H x W), corresponding sequence of cloud masks. A value of 0 denotes
                            an observed pixel and 1 an occluded pixel.
    """

    assert cloud_masks.shape[0] <= images.shape[0]

    if cloud_masks.shape[0] < images.shape[0]:
        if t_masked is None:
            # Randomly sample the images to be masked
            t_masked = np.random.choice(np.arange(0, images.shape[0]), cloud_masks.shape[0], replace=False)

        masks = torch.zeros((images.shape[0], 1, *images.shape[-2:]))
        masks[t_masked, :, :, :] = cloud_masks
    else:
        masks = cloud_masks.clone()

    # Image time series with overlaid cloud masks filled with value `fill_value`
    images_masked, masks = masks_init_filling(
        seq=images, masks=masks, fill_type='fill_value', fill_value=fill_value, dilate_cloud_masks=dilate_cloud_masks
    )

    return images_masked, masks
