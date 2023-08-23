import datetime as dt
import math
import random
from itertools import compress
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from torch import Tensor

# Launch date of Sentinel-2A
REFERENCE_DATE: dt.date = dt.datetime(*map(int, '2015-06-23'.split("-")), tzinfo=None).date()

# Strategies for positional encoding
PE_STRATEGIES = ['day-of-year', 'day-within-sequence', 'absolute', 'enumeration']

# Mask types for synthetically generating data gaps in cloud-free satellite image time series
MASK_TYPES = ['random_clouds', 'real_clouds']

def str2date(date_string: str) -> dt.date:
    """Converts a date in string format to datetime format."""
    return dt.datetime.strptime(date_string, '%Y-%m-%d').date()


def detect_impaired_frames(
        seq: Tensor, cloud_prob: Optional[Tensor], cloud_mask: Tensor, increased_filter_strength: bool = False
) -> Tuple[List[int], Dict[str, Tensor]]:
    """
    Returns the indices of unavailable or cloudy/foggy images within a given image time series `seq`.

    Args:
        seq:                        torch.Tensor, T x C x H x W.
        cloud_prob:                 torch.Tensor, cloud masks, T x 1 x H x W, per-pixel cloud probabilities
                                    in percentage [%].
        cloud_mask:                 torch.Tensor, binary cloud masks, T x 1 x H x W.
        increased_filter_strength:  bool, True to detect smaller clouds (at the expense of a higher false positive
                                    rate), False for a more conservative cloud filtering.

    Returns:
        idx_impaired_frames:        list of int, indices of impaired frames (unavailable or cloudy/foggy frames).
        debug:                      dict, intermediate results, used for debugging purposes.
    """

    # Sequence length and spatial dimensions
    seq_length, _, H, W = seq.shape

    if cloud_prob is not None:
        # Percentage of cloudy pixels per frame
        clouds_cumsum = (torch.sum(cloud_mask, dim=(-2, -1)) / (H * W) * 100).flatten()

        # Percentage of pixels with cloud probability >`p1`% per frame
        p1 = 1
        cc_cumsum = (torch.sum(cloud_prob > p1, dim=(-2, -1)) / (H * W) * 100).flatten()

        # Criterion 1: a frame not available if all pixels are NaN
        not_avail = torch.Tensor([torch.all(torch.isnan(seq[i, ...])) for i in range(seq_length)])

        # Criterion 2: a frame is considered as cloudy/foggy if >=`q`% of its pixels are cloudy
        # (w.r.t. the binary cloud mask)
        q = 40
        clouds_status = clouds_cumsum >= q

        # Criterion 3: a frame is considered as cloudy/foggy if
        # (i) at least `p2`% of the pixels have a cloud probability of >`p1`% AND
        # (ii) the percentage of cloudy pixels (according to the binary cloud mask) exceeds `p3`%
        # => NOTE: two sets of thresholds for `p2` and `p4`
        p2 = 1
        p3 = 10
        cc_status1 = torch.logical_and(cc_cumsum > p2, clouds_cumsum > p3)
        p2_2 = 4
        p3_2 = 3.5 if increased_filter_strength else 8
        cc_status2 = torch.logical_and(cc_cumsum > p2_2, clouds_cumsum > p3_2)
        cc_status = torch.logical_or(cc_status1, cc_status2)

        # Combine all criteria: criterion 1 OR criterion 2 OR criterion 3
        frame_impaired = torch.logical_or(not_avail, torch.logical_or(cc_status, clouds_status))
    else:
        # Percentage of cloudy pixels per frame
        clouds_cumsum = (torch.sum(cloud_mask, dim=(-2, -1)) / (H * W) * 100).flatten()

        # Criterion 1: a frame not available if all pixels are NaN
        not_avail = torch.Tensor([torch.all(torch.isnan(seq[i, ...])) for i in range(seq_length)])

        # Criterion 2: a frame is considered as cloudy/foggy if >=`q`% of its pixels are cloudy
        # (w.r.t. the binary cloud mask)
        q = 5
        clouds_status = clouds_cumsum >= q

        # Combine both criteria: criterion 1 OR criterion 2
        frame_impaired = torch.logical_or(not_avail, clouds_status)

    # Extract the indices of the impaired frames
    idx_impaired_frames = list(compress(range(seq_length), frame_impaired))

    debug_info = {
        'clouds_cumsum': clouds_cumsum,
        'clouds_status': clouds_status,
        'frame_impaired': frame_impaired
    }
    if cloud_prob is not None:
        debug_info['cc_cumsum'] = cc_cumsum
        debug_info['cc_status'] = cc_status

    return idx_impaired_frames, debug_info

    
def get_position_for_positional_encoding(dates: List[dt.date], strategy: str) -> Tensor:
    """
    Extracts the position index for every observation in an image time series, expressed as the number of days since
    a given reference date. The position indices will be used for sinusoidal positional encoding in the temporal encoder
    of U-TILISE.

    Args:
        dates:     list of datetime.date dates, acquisition dates for every observation in the sequence.
        strategy:  str, specifies the reference date. Choose among:
                        'day-of-year':          The position of each observation is expressed as the number of days
                                                since the 1st of January of the respective year, where the
                                                1st of January equals position 0 (i.e, seasonal information is
                                                implicitly encoded in the position).
                        'day-within-sequence':  The position of each observation is expressed relative to the first
                                                observation in the sequence, i.e., the first observation in the sequence
                                                is encoded as position 0 (i.e, seasonal information is not encoded in
                                                the position).
                        'absolute':             The position of each observation is expressed as the number of days
                                                since the reference date `REFERENCE_DATE`.
                        'enumeration':          Simple enumeration of the observations, i.e., 0, 1, 2, 3, etc.

    Returns:
        position:  torch.Tensor, number of days since a given reference date for every observation in the sequence.
    """

    if strategy == 'enumeration':
        position = torch.arange(0, len(dates))
    elif strategy == 'day-of-year':
        position = Tensor([(date - dt.date(date.year, 1, 1)).days for date in dates])
    elif strategy == 'day-within-sequence':
        position = Tensor([(date - dates[0]).days for date in dates])
    elif strategy == 'absolute':
        position = Tensor([(date - REFERENCE_DATE).days for date in dates])
    else:
        raise NotImplementedError(f'Unknown positional encoding strategy {strategy}.\n')

    return position


def sample_indices_masked_frames(idx_valid_input_frames: np.ndarray,
                                 ratio_masked_frames: float = 0.5,
                                 ratio_fully_masked_frames: float = 0.0,
                                 non_masked_frames: Optional[List[int]] = None,
                                 fixed_masking_ratio: bool = True) -> Dict[str, np.ndarray]:
    """
    Generates a sequence of `masks` to synthetically mask an image time series. masks[t1, 0, y1, x1] == 1 will mask the
    spatio-temporal location (t1, y1, x1), whereas masks[t2, 0, y2, x2] == 0 will retain the observed reflectance at
    the spatio-temporal location (t2, y2,x2) (w.r.t. all spectral channels).

    Args:
        idx_valid_input_frames:      np.ndarray, indices of those frames that are available for masking.
        ratio_masked_frames:         float, ratio of (partially or fully) masked frames.
        ratio_fully_masked_frames:   float, ratio of fully masked frames.
        non_masked_frames:           list of int, indices of those frames that should be excluded from masking
                                     (e.g., first frame).
        fixed_masking_ratio:         bool, True to enforce the same ratio of masked frames across image time sequences,
                                     False to vary the ratio of masked frames across image time sequences.
                                     For varying sampling ratios: `ratio_masked_frames` and `ratio_fully_masked_frames`
                                     define upper bounds.

    Returns:
        dict, defines two mutually exclusive sets of frame indices sampled from `idx_valid_input_frames`:
            'indices_masked':        np.ndarray, indices of (partially) masked frames.
            'indices_fully_masked':  np.ndarray, indices of fully masked frames.
    """

    assert ratio_fully_masked_frames <= ratio_masked_frames, "Masking parameter `ratio_fully_masked_frames` needs to " \
                                                             "be smaller or equal to `ratio_masked_frames.`"

    # Upper bound: Maximum number of masked input frames (partially or fully masked)
    num_total = len(idx_valid_input_frames)

    if not fixed_masking_ratio:
        # Vary the sampling ratio by adjusting the number of frames available for masking
        # (at least one frame has to be masked)
        num_total = random.randint(1, num_total)

    # Number of masked frames (partially or fully masked)
    num_masked = math.ceil(ratio_masked_frames * num_total)

    # Number of fully masked frames
    num_fully_masked = math.ceil(ratio_fully_masked_frames * num_total)

    # Randomly select the indices of those frames that will be masked (partially or fully)
    if non_masked_frames is not None:
        non_masked_frames = np.asarray(non_masked_frames)
        if np.any(non_masked_frames < 0):
            # Account for negative indices
            indices_pos = non_masked_frames[non_masked_frames >= 0]
            indices_neg = idx_valid_input_frames[non_masked_frames[non_masked_frames < 0]]
            non_masked_frames = np.concatenate((indices_pos, indices_neg), axis=0)
        else:
            non_masked_frames = idx_valid_input_frames[non_masked_frames]
        list_frames = np.setdiff1d(idx_valid_input_frames, non_masked_frames)
        indices_masked = np.random.choice(list_frames, min(num_masked, list_frames.size), replace=False)
    else:
        indices_masked = np.random.choice(idx_valid_input_frames, num_masked, replace=False)

    # Randomly selected the frame indices of the fully masked frames
    indices_fully_masked = np.random.choice(indices_masked, num_fully_masked, replace=False)

    return {'indices_masked': indices_masked, 'indices_fully_masked': indices_fully_masked}


def get_mask_sampling_id_hdf5(mask_args: DictConfig) -> Tuple[str, str]:
    """
    Returns the storage location of pre-computed masks relative to a parent hdf5 group (data sample).

    Args:
        mask_args: dict, mask settings.

    Returns:
        mask_dir:   string, hdf5 group name.
        mask_name:  string, hd5f dataset name, mask sequence (stored in `mask_dir`).
    """

    if 'ratio_fully_masked_frames' in mask_args and mask_args.ratio_fully_masked_frames > 0.:
        mask_dir = f'ratio_masked_{mask_args.ratio_masked_frames}_fully_masked_{mask_args.ratio_fully_masked_frames}'
    else:
        mask_dir = f'ratio_masked_{mask_args.ratio_masked_frames}'

    mask_name = f'masks_{mask_args.mask_type}'

    return mask_dir, mask_name
