import math
import matplotlib
import os
import torch

from enum import Enum
from matplotlib import pyplot as plt
from torch import Tensor, nn
from typing import Any, Dict, List, Literal, Optional, Tuple

from lib import config_utils, data_utils, utils, visutils
from lib.models import MODELS
from lib.visutils import COLORMAPS


class Method(Enum):
    UTILISE = 'utilise'
    TRIVIAL = 'trivial'

    
class Mode(Enum):
    LAST = 'last'
    NEXT = 'next'
    CLOSEST = 'closest'
    LINEAR_INTERPOLATION = 'linear_interpolation'
    NONE = None


class Imputation:
    def __init__(
            self,
            config_file_train: str | None,
            method: Literal['utilise', 'trivial'] = 'utilise',
            mode: Literal['last', 'next', 'closest', 'linear_interpolation'] | None = None,
            checkpoint: str | None = None
    ):
        
        self.method = Method(method)
        self.mode = Mode(mode)
        self.checkpoint = checkpoint
        self.config_file_train = config_file_train

        if self.method == Method.TRIVIAL and self.mode == Mode.NONE:
            raise ValueError(f'No mode specified. Choose among {[mode.value for mode in Mode]}.')
        
        if self.method == Method.UTILISE:
            if self.checkpoint is None:
                raise ValueError('No checkpoint specified.\n')
                
            if self.config_file_train is None:
                raise ValueError('No training configuration file specified.\n')
            
            if not os.path.isfile(self.config_file_train):
                raise FileNotFoundError(f'Cannot find the configuration file used during training: {self.config_file_train}\n')

            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(f'Cannot find the model weights: {self.checkpoint}\n')
                
            # Read the configuration file used during training
            self.config = config_utils.read_config(self.config_file_train)

            # Extract the temporal window size and the number of channels used during training
            self.temporal_window = self.config.data.max_seq_length
            self.num_channels = data_utils.get_dataset(self.config, phase=self.config.misc.run_mode).num_channels
        

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        _ = torch.set_grad_enabled(False)

        # Get the model
        if self.method == Method.UTILISE:
            self.model, _ = utils.get_model(self.config, self.num_channels)
            self._resume()
            self.model.to(self.device).eval()
        else:
            self.model = MODELS['ImageSeriesInterpolator'](mode=self.mode.value)

    def impute_sample(
            self,
            batch: Dict[str, Any],
            t_start: Optional[int] = None,
            t_end: Optional[int] = None,
            return_att: Optional[bool] = False
    ) -> Tuple[Dict[str, Any], Tensor, Tensor] | Tuple[Dict[str, Any], Tensor]:

        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch['x'] = batch['x'][:, t_start:t_end, ...]
            
            for key in ['y', 'masks', 'cloud_mask', 'masks_valid_obs']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]
                    
            for key in ['days', 'position_days']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        if isinstance(self.model, MODELS['utilise']):
            batch = data_utils.to_device(batch, self.device)
            if return_att:
                y_pred, att = impute_sequence(self.model, batch, self.temporal_window, return_att=True)
                if att is not None:
                    att = att.cpu()
            else:
                y_pred = impute_sequence(self.model, batch, self.temporal_window, return_att=False)
            batch = data_utils.to_device(batch, 'cpu')
            y_pred = y_pred.cpu()
        else:
            y_pred = self.model(batch['x'], cloud_mask=batch['masks'], days=batch['days'])

        if return_att:
            return batch, y_pred, att
        return batch, y_pred

    def _resume(self) -> None:
        checkpoint = torch.load(self.checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint \'{self.checkpoint}\' loaded.')
        print(f"Chosen epoch: {checkpoint['epoch']}\n")
        del checkpoint


def impute_sequence(
        model, batch: Dict[str, Any], temporal_window: int, return_att: bool = False
) -> Tensor | Tuple[Tensor, Tensor]:
    """
    Sliding-window imputation of satellite image time series.

    Assumption: `batch` consists of a single sample.
    """

    x = batch['x']
    positions = batch['position_days']
    y_pred: Tensor
    att: Tensor

    if temporal_window is None or x.shape[1] <= temporal_window:
        # Process the entire sequence in one go
        if return_att:
            y_pred, att = model(x, batch_positions=positions, return_att=True)
        else:
            y_pred = model(x, batch_positions=positions)
    else:
        if return_att:
            att = None
            
        t_start = 0
        t_end = temporal_window
        t_max = x.shape[1]
        cloud_coverage = torch.mean(batch['masks'], dim=(0, 2, 3, 4))
        reached_end = False

        while not reached_end:
            y_pred_chunk = model(x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end])

            if t_start == 0:
                # Initialize the full-length output sequence
                B, T, _, H, W = x.shape
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)

                y_pred[:, t_start:t_end] = y_pred_chunk

                # Move the temporal window
                t_start_old = t_start
                t_end_old = t_end
                t_start, t_end = move_temporal_window_next(t_start, t_max, temporal_window, cloud_coverage)
            else:
                # Find the indices of those frames that have been processed by both the previous and the current
                # temporal window
                t_candidates = torch.Tensor(
                    list(set(torch.arange(t_start_old, t_end_old).tolist()) & set(
                        torch.arange(t_start, t_end).tolist()))
                ).long().to(x.device)

                # Find the frame for which the difference between the previous and the current prediction is
                # the lowest:
                # use this frame to switch from the previous imputation results to the current imputation results
                error = torch.mean(
                    torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start]),
                    dim=(0, 2, 3, 4)
                )
                t_switch = error.argmin().item() + t_start
                y_pred[:, t_switch:t_end] = y_pred_chunk[:, (t_switch - t_start)::]

                if t_end == t_max:
                    reached_end = True
                else:
                    # Move the temporal window
                    t_start_old = t_start
                    t_end_old = t_end
                    t_start, t_end = move_temporal_window_next(
                        t_start_old, t_max, temporal_window, cloud_coverage
                    )

    if return_att:
        return y_pred, att
    return y_pred


def move_temporal_window_end(t_max: int, temporal_window: int) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation such that the last frame of the temporal window coincides with the
    last frame of the image sequence.

    Args:
        t_max:              int, sequence length of the image sequence
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    t_start = t_max - temporal_window
    t_end = t_max

    return t_start, t_end


def move_temporal_window_next(
        t_start: int, t_max: int, temporal_window: int, cloud_coverage: Tensor
) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation by half of the temporal window size (= stride).
    If the first frame within the new temporal window is cloudy (cloud coverage above 10%), the temporal window is
    shifted by at most half the stride (backward or forward) such that the first frame is as least cloudy as
    possible.

    Args:
        t_start:            int, frame index, start of the subsequence for processing
        t_max:              int, frame index, t_max - 1 is the last frame of the subsequence for processing
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing
        cloud_coverage:     torch.Tensor, (T,), cloud coverage [-] per frame

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    stride = temporal_window // 2
    t_start += stride

    if t_start + temporal_window > t_max:
        # Reduce the stride such that the end of the temporal window coincides with the end of the entire sequence
        t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    else:
        # Check if the start of the next temporal window is mostly cloud-free
        if cloud_coverage[t_start] <= 0.1:
            # Keep the default stride and ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)
        else:
            # Find the least cloudy frame within [t_start + stride - dt, t_start + stride + dt]
            dt = math.ceil(stride / 2)
            left = max(0, t_start - dt)
            right = min(t_start + dt + 1, t_max)

            # Frame(s) with the lowest cloud coverage within [t_start + stride - dt, t_start + stride + dt]
            t_candidates = (cloud_coverage[left:right] == cloud_coverage[left:right].min()).nonzero(as_tuple=True)[
                               0] + left

            # Take the frame closest to the standard stride
            t_start = t_candidates[torch.abs(t_candidates - t_start).argmin()].item()

            # Ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)

    return t_start, t_end


def upsample_att_maps(att: Tensor, target_shape: Tuple[int, int]) -> Tensor:
    """Upsamples the attention masks `att` to the spatial resolution `target_shape`."""

    n_heads, b, t_out, t_in, h, w = att.shape
    attn = att.view(n_heads * b * t_out, t_in, h, w)

    attn = nn.Upsample(
        size=target_shape, mode="bilinear", align_corners=False
    )(attn)

    return attn.view(n_heads, b, t_out, t_in, *target_shape)


def visualize_att_for_one_head_across_time(
        seq: Tensor,
        att: Tensor,
        head: int,
        batch: int = 0,
        upsample_att: bool = True,
        indices_rgb: List[int] | List[float] | Tensor | None = None,
        brightness_factor: float = 1,
        fontsize: int = 10,
        scale_individually: bool = False
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks learned by the `head`.th attention head across time.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        head:                   int, index of the attention head to be visualized.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    seq_length = seq.shape[1]
    figsize = (7, 1 + seq_length)
    fig, axes = plt.subplots(nrows=seq_length + 1, ncols=1, figsize=figsize)

    # Plot satellite image time series
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)
    axes[0].imshow(grid, COLORMAPS['rgb'])
    axes[0].set_title('Input sequence', fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention mask for attention head `head` across all time steps
    for t in range(seq_length):
        grid = visutils.gallery(att[head, batch, t, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[t + 1].imshow(grid, COLORMAPS['att'], vmin=vmin, vmax=vmax)
        axes[t + 1].set_title(f'Attention mask, head {head}, target frame {t}', fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig


def visualize_att_for_target_t_across_heads(
        seq: Tensor,
        att: Tensor,
        t_target: int,
        batch: int = 0,
        upsample_att: bool = True,
        indices_rgb: List[int] | List[float] | Tensor | None = None,
        brightness_factor: float = 1,
        figsize: Tuple[float, float] = (10, 7),
        dpi: int = 200,
        fontsize: int = 10,
        scale_individually: bool = False,
        highlight_t_target: bool = True
) -> matplotlib.figure.Figure:
    """
    Visualizes the attention masks of all attention heads w.r.t. to the time step `t_target`.

    Args:
        seq:                    torch.Tensor, B x T x C x H x W, satellite image time series.
        att:                    torch.Tensor, n_head x B x T x T x h x w, attention masks.
        t_target:               int, time step (temporal coordinate) to visualize.
        batch:                  int, batch index to visualize.
        upsample_att:           bool, True to upsample the attention masks to the spatial resolution of the satellite
                                image time series; False to keep the native spatial resolution of the attention masks.
        indices_rgb:            list of int or list of float or torch.Tensor, indices of the RGB channels.
        brightness_factor:      float, brightness factor applied to all images in the sequence.
        figsize:                (float, float), figure size.
        dpi:                    int, dpi of the figure.
        fontsize:               int, font size.
        scale_individually:     bool, True to scale the attention masks for each time step individually; False to
                                maintain a common scale across all attention masks and time.
        highlight_t_target:     bool, True to highlight the target time step by drawing a red frame around the
                                respective image in the time series.

    Returns:
        matplotlib.pyplot.
    """

    indices_rgb = [0, 1, 2] if indices_rgb is None else indices_rgb

    if upsample_att:
        target_shape = seq.shape[-2:]
        att = upsample_att_maps(att, target_shape)

    n_heads = att.shape[0]
    fig, axes = plt.subplots(nrows=n_heads + 1, ncols=1, figsize=figsize, dpi=dpi)

    # Plot input sequence
    grid = visutils.gallery(seq[batch, :, indices_rgb, :, :], brightness_factor=brightness_factor)

    if highlight_t_target:
        # Create a red frame to highlight the target frame
        border_thickness = 2
        H, W = seq.shape[-2:]
        H += 2 * border_thickness
        W += 2 * border_thickness
        frame_color = torch.Tensor([1, 0, 0]).type(grid.dtype)

        if t_target < 0:
            t_target = seq.shape[1] - abs(t_target)

        grid[0:(2 * border_thickness + 1), t_target * W:(t_target + 1) * W, :] = frame_color
        grid[-2 * border_thickness::, t_target * W:(t_target + 1) * W, :3] = frame_color
        grid[:, t_target * W - border_thickness:t_target * W + border_thickness, :] = frame_color
        grid[:, ((t_target + 1) * W - border_thickness):(t_target + 1) * W + border_thickness, :] = frame_color

        if t_target == seq.shape[1] - 1:
            grid[:, ((t_target + 1) * W - 2 * border_thickness):(t_target + 1) * W + border_thickness, :] = frame_color
        elif t_target == 0:
            grid[:, 0:2 * border_thickness, :] = frame_color

    axes[0].imshow(grid, COLORMAPS['rgb'])
    axes[0].set_title('Input sequence', fontsize=fontsize)

    if scale_individually:
        vmin = None
        vmax = None
    else:
        vmin = 0
        vmax = 1

    # Plot attention masks per head for frame `t_target`
    for head in range(n_heads):
        grid = visutils.gallery(att[head, batch, t_target, :, :, :].unsqueeze(1), brightness_factor=1)
        axes[head + 1].imshow(grid, COLORMAPS['att'], vmin=vmin, vmax=vmax)
        axes[head + 1].set_title(f'Attention mask, head {head}, target frame {t_target}', fontsize=fontsize)

    for ax in axes.ravel():
        ax.set_axis_off()
    plt.tight_layout()

    return fig
