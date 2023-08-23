import datetime as dt
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.utils.data
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import Tensor
from torchvision import transforms

from lib import torch_transforms
from lib.datasets import dataset_tools
from lib.datasets.dataset_tools import MASK_TYPES, PE_STRATEGIES
from lib.datasets.mask_generation import (
    masks_init_filling,
    overlay_seq_with_clouds
)

SPLITS = ['train', 'iid', 'ood', 'extreme', 'seasonal']
CHANNEL_CONFIG = ['rgb', 'bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']

# Sequence length per data split
split2seq_length = {
    'train': 30,
    'iid_test_split': 30,
    'ood_test_split': 30,
    'extreme_test_split': 60,
    'seasonal_test_split': 210
}


class EarthNet2021Dataset(torch.utils.data.Dataset):
    """
    torch.utils.data.Dataset class for the EarthNet2021 dataset.

    Dataset source:
    C. Requena-Mesa, V. Benson, M. Reichstein, J. Runge, and J. Denzler, “EarthNet2021: A large-scale dataset and
    challenge for earth surface forecasting as a guided video prediction task”, in Proceedings of th IEEE/CVF Conference
    on Computer Vision and Pattern Recognition (CVPR) Workshops, 2021, pp. 1132-1142.

    Args:
        root:                  str, data root directory (must contain a hdf5 file storing the EarthNet2021 dataset).
        hdf5_file:             None or str:
                                   None:   The hdf5 filename is identical to the naming convention of the data split.
                                           E.g., train.hdf5
                                   str:    hdf5 filepath w.r.t. the `root` directory.
        preprocessed:          bool, True to use preprocessed validation/test data, i.e., the temporal trimming
                               and mask generation are loaded from the hdf5 file instead of computed on-the-fly
                               (automatically switched off for training data).

        split:                 str, data split, choose among ['train', 'iid', 'ood', 'extreme', 'seasonal'].
        mode:                  str, data split mode for the training set, choose among ['train', 'val', 'all'].
        channels:              str, channels to be extracted, choose among
                               ['rgb', 'bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask'].

        filter_settings:       dict, filtering settings to remove cloudy/foggy/unavailable images from the satellite
                               image time series. The dictionary must contain the following key-value pairs:
                                    'type': None or str:
                                          None:                     No filter applied. Retains all images per sequence.
                                          'cloud-free':             Retains all cloud-free and available images
                                                                    per sequence; the output `masks_valid_obs[t]`
                                                                    equals to 1 if the t.th image is cloud-free and
                                                                    available; 0 otherwise.
                                          'cloud-free_consecutive': Finds the longest subsequence of cloud-free
                                                                    and available images per sequence; the output
                                                                    `masks_valid_obs[t]` equals to 1 for all images
                                                                    that belong to the longest cloud-free subsequence;
                                                                    0 otherwise.

                                    'min_length':              int, minimal sequence length. Removes cloud-free image
                                                               sequences consisting of less than `min_length` images.
                                    'return_valid_obs_only':   bool, True to return only those images per sequence for
                                                               which `masks_valid_obs[t]` == 1;
                                                               CONSEQUENCE: variable sequence lengths across samples.
                                    'max_num_consec_invalid': int, maximum number of consecutive invalid images per
                                                              sequence. Sequences with more than
                                                              `max_num_consec_invalid` consecutive invalid images
                                                              (cloudy/unavailable) are ignored.

        crop_settings:         dict, settings to spatially crop the satellite image time series.
                               The dictionary must contain the following key-value pairs:
                                   'enabled':  bool, True to activate cropping, False otherwise.
                                   'type':     str, Crop location. Choose from ['random', 'center'].
                                   'shape':    (int, int), image size after cropping.

        pe_strategy:           str, strategy for positional encoding. Choose among:
                                    'day-of-year':          The position of each observation is expressed as the number
                                                            of days since the 1st of January of the respective year,
                                                            where the 1st of January equals position 0 (i.e, seasonal
                                                            information is implicitly encoded in the position).
                                    'day-within-sequence':  The position of each observation is expressed relative to
                                                            the first observation in the sequence, i.e., the first
                                                            observation in the sequence is encoded as position 0 (i.e,
                                                            seasonal information is not encoded in the position).
                                    'absolute':             The position of each observation is expressed as the number
                                                            of days since the reference date `REFERENCE_DATE`, defined
                                                            in dataset_tools.py.
                                    'enumeration':          Simple enumeration of the images, i.e., 0, 1, 2, 3, etc.

        mask_kwargs:           dict, settings for generating synthetically masked satellite image series. The dictionary
                               can include the following key-value pairs (set `mask_kwargs` to None if the image
                               sequences should not be masked):
                                    'mask_type':                    str, type of masks for synthetic data gap generation.
                                                                    Choose among ['random_clouds', 'real_clouds'].
                                    'ratio_masked_frames':          float, ratio of partially/fully masked images in a
                                                                    satellite image time series
                                                                    (upper bound if `fixed_masking_ratio` is True).
                                    'ratio_fully_masked_frames':    float, ratio of fully masked images in a satellite
                                                                    image time series
                                                                    (upper bound if `fixed_masking_ratio` is True).
                                    'fixed_masking_ratio':          bool, True to vary the masking ratio across
                                                                    satellite image time series, False otherwise.
                                    'non_masked_frames':            list of int, time steps to be excluded from masking.
                                    'intersect_real_cloud_masks':   bool, True to intersect randomly sampled cloud masks
                                                                    with the actual cloud mask sequence, False otherwise.
                                    'dilate_cloud_masks':           bool, True to dilate the cloud masks before masking,
                                                                    False otherwise.
                                    'fill_type':                    str, strategy for initializing masked pixels.
                                                                    See lib/datasets/mask_generation.py,
                                                                    masks_init_filling() for details.
                                    'fill_value':                   float, pixel value of masked pixels.

        render_occluded_above_p: float, mark an entire image as occluded if its cloud coverage exceeds
                                 `render_occluded_above_p` [-] (i.e., the cloud mask of the respective image is
                                 overwritten). Set `render_occluded_above_p` to None to retain the original cloud masks.
        return_cloud_prob:       bool, True to return the Sen2Cor cloud mask (cloud probabilities [0-1]) as
                                 `cloud_prob`.
        return_class_map:        bool, True to return the ESA Scene Classification map as `classification`.
        return_cloud_mask:       bool, True to return the EarthNet2021 binary cloud mask (0 if non-cloudy, 1 if cloudy)
                                 as `cloud_mask`.
        augment:                 bool, True to activate data augmentation (random rotation by multiples of 90 degrees
                                 as well as random flipping along the horizontal and vertical axes), False otherwise.
        max_seq_length:          int, randomly selects a subsequence consisting of `max_seq_length` images.
                                 Image time series shorter than `max_seq_length` remain unmodified.
                                 Set `max_seq_length` to None to retain the original sequences after cloud filtering.
        to_export:               bool, True to return additional parameters used in the simulation for reproducibility
                                 purposes, False otherwise.
    """

    def __init__(self,
                 root: str,
                 hdf5_file: str | Path | None = None,
                 preprocessed: bool = False,
                 split: str = 'train',
                 mode: Optional[str] = 'train',
                 channels: str = 'bgr-nir',
                 filter_settings: Optional[Dict | DictConfig] = None,
                 crop_settings: Optional[Dict | DictConfig] = None,
                 pe_strategy: str = 'day-of-year',
                 mask_kwargs: Optional[Dict | DictConfig] = None,
                 render_occluded_above_p: Optional[float] = None,
                 return_cloud_prob: bool = False,
                 return_class_map: bool = False,
                 return_cloud_mask: bool = True,
                 augment: bool = False,
                 max_seq_length: Optional[int] = None,
                 verbose: int = 0,
                 to_export: bool = False
                 ):

        if filter_settings is None:
            filter_settings = {'type': None, 'min_length': 5, 'return_valid_obs_only': False}
        if crop_settings is None:
            crop_settings = {'enabled': False, 'type': 'random', 'shape': (64, 64)}

        # -------------------------------------- Verify input parameters -------------------------------------- #

        if not os.path.exists(root):
            raise FileNotFoundError(f"Invalid `root`. Root directory does not exist: {root}")

        if split not in SPLITS:
            raise ValueError(f"Invalid `split` parameter. Choose among {SPLITS} to specify `split`.\n")

        if split != 'train':
            split = split + '_test_split'

        if hdf5_file is None:
            hdf5_file = Path(root) / (split + '.hdf5')
        else:
            hdf5_file = Path(root) / hdf5_file

        if not os.path.exists(hdf5_file):
            raise FileNotFoundError(f"Cannot find the hdf5 file: {hdf5_file}")

        if split == 'train' and mode not in ['train', 'val', 'all']:
            raise ValueError(
                "Invalid `mode` parameter. Choose among ['train', 'val', 'all'] to specify `mode`.\n")

        if isinstance(filter_settings, dict):
            filter_settings = OmegaConf.create(filter_settings)

        if isinstance(crop_settings, dict):
            crop_settings = OmegaConf.create(crop_settings)

        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if not isinstance(filter_settings, DictConfig) or 'type' not in filter_settings or \
                filter_settings.type not in [None, 'cloud-free', 'cloud-free_consecutive'] or \
                ('min_length' in filter_settings and not isinstance(filter_settings.min_length, int)) or \
                ('max_num_consec_invalid' in filter_settings and not isinstance(filter_settings.max_num_consec_invalid,
                                                                                int)):
            raise RuntimeError(
                "Invalid `filter_settings` parameter. Define a dictionary with the following keys and value options:\n"
                "'type': None or str,          # sequence filter type, choose from [None, 'cloud-free', "
                "'cloud-free_consecutive']\n "
                "'min_length': int,            # minimum number of frames per sequence\n"
                "'return_valid_obs_only': bool # True to return filtered frames only, False otherwise\n "
                "'max_num_consec_invalid': int # maximum number of consecutive invalid frames per sequence\n")

        if not isinstance(crop_settings, DictConfig):
            raise ValueError(
                "Invalid `crop_settings` parameter. Define a dictionary with the following keys and value options:\n"
                "'on': bool,                   # True to spatially crop the image time series\n "
                "'type': str,                  # rectangular crop type, choose from ['random', 'center']\n"
                "'shape': (int, int)           # crop shape\n")

        if pe_strategy not in PE_STRATEGIES:
            raise ValueError(f"Invalid `pe_strategy` parameter. Choose among {PE_STRATEGIES} to specify the strategy "
                             "used for positional encoding.\n")

        if 'enabled' not in crop_settings or not isinstance(crop_settings.enabled, bool):
            raise ValueError("Invalid `crop_settings['enabled']` parameter. Specify a boolean.")

        if crop_settings.enabled:
            if 'type' not in crop_settings or crop_settings.type not in ['random', 'center']:
                raise ValueError("Invalid `crop_settings['type']` parameter. Choose among ['random', 'center'] to "
                                 "specify `crop_settings['type']`.\n")

            if 'shape' not in crop_settings or not \
                    (isinstance(crop_settings.shape, (tuple, ListConfig)) and
                     len(crop_settings.shape) == 2 and isinstance(crop_settings.shape[0], int) and
                     isinstance(crop_settings.shape[1], int)):
                raise RuntimeError("Invalid `crop_settings['shape']` parameter. Specify a tuple (int, int).\n")

        if '-mask' in channels and mask_kwargs is None:
            raise RuntimeError("Cannot concatenate the image time series with associated masks. "
                               "Please provide a masking strategy. \n")

        if mask_kwargs is not None and 'mask_type' in mask_kwargs and mask_kwargs.mask_type not in MASK_TYPES:
            raise ValueError(f"Invalid `mask_type` parameter. Choose among {MASK_TYPES} to specify the type of masks"
                             "used for synthetically generating data gaps in cloud-free satellite image time series.\n")

        if render_occluded_above_p is not None and not isinstance(render_occluded_above_p, float):
            raise TypeError("Invalid `render_occluded_above_p` parameter. Specify a float (or None).")

        if not isinstance(return_cloud_prob, bool):
            raise TypeError("Invalid `return_cloud_prob` parameter. Specify a boolean.")

        if not isinstance(return_class_map, bool):
            raise TypeError("Invalid `return_class_map` parameter. Specify a boolean.")

        if not isinstance(return_cloud_mask, bool):
            raise TypeError("Invalid `return_cloud_mask` parameter. Specify a boolean.")

        if not isinstance(augment, bool):
            raise TypeError("Invalid `augment` parameter. Specify a boolean.")

        # -------------------------------------- Data and split -------------------------------------- #
        self.hdf5_file = hdf5_file
        self.root = root
        self.split = split
        self.mode = mode if self.split == 'train' else None

        # Fixed sequence length? If yes, the `collate_fn` function of the data loader pads samples to the same temporal
        # length before collating them to a batch
        if filter_settings and filter_settings.get('type', None) is not None:
            self.variable_seq_length = filter_settings.return_valid_obs_only
        else:
            self.variable_seq_length = False

        # Parameters used for creating synthetic data gaps
        if mask_kwargs is not None:
            mask_kwargs.mask_type = mask_kwargs.get('mask_type', 'random_clouds')
            mask_kwargs.ratio_masked_frames = mask_kwargs.get('ratio_masked_frames', 0.5)
            mask_kwargs.ratio_fully_masked_frames = mask_kwargs.get('ratio_fully_masked_frames', 0.0)
            mask_kwargs.non_masked_frames = mask_kwargs.get('non_masked_frames', [])

            self.fill_type = mask_kwargs.get('fill_type', 'fill_value')
            self.fill_value = mask_kwargs.get('fill_value', 1)
            self.fixed_masking_ratio = mask_kwargs.get('fixed_masking_ratio', False)
            self.intersect_real_cloud_masks = mask_kwargs.get('intersect_real_cloud_masks', False)
            self.dilate_cloud_masks = mask_kwargs.get('dilate_cloud_masks', False)
            self.mask_kwargs = mask_kwargs
        else:
            self.mask_kwargs = None

        # Strategy for positional encoding
        self.pe_strategy = pe_strategy

        # Fully occlude images with high cloud cover
        self.render_occluded_above_p = render_occluded_above_p

        self.preprocessed = preprocessed
        self.to_export = to_export

        # -------------------------------------- Channel settings -------------------------------------- #
        # Image channels and/or composites
        if channels not in CHANNEL_CONFIG:
            raise ValueError(f"Unknown channel configuration `{channels}`. Choose among {CHANNEL_CONFIG} to "
                             "specify `channels`.\n")

        self.channels = channels

        # Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
        # self.s2_channels: used to extract the relevant channels from the hdf5 file
        # self.c_index_rgb and self.c_index_nir: indices of the RGB and NIR channels, w.r.t. the output of
        # the self.__getitem__() call
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [0, 1, 2]
        else:
            # self.channels == 'rgb'
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([0, 1, 2]).long()
            self.s2_channels = [2, 1, 0]

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [3]
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        if '-mask' in self.channels:
            # self.channels in ['bgr-mask', 'bgr-nir-mask']
            self.num_channels += 1

        # -------------------------------------- Spatial cropping settings -------------------------------------- #
        self.crop_settings = OmegaConf.create(crop_settings)

        # Image size
        self.image_size = crop_settings.shape if self.crop_settings.enabled else (128, 128)

        # Cropping function
        if self.crop_settings.enabled:
            if self.crop_settings.type == 'random':
                self.crop_function = transforms.RandomCrop(self.image_size)
            elif self.crop_settings.type == 'center':
                self.crop_function = transforms.CenterCrop(self.image_size)

        # -------------------------------------- Filtering settings -------------------------------------- #
        self.filter_settings = OmegaConf.create(filter_settings)
        self.filter_settings.max_num_consec_invalid = self.filter_settings.get('max_num_consec_invalid', None)
        self.filter_settings.min_length = self.filter_settings.get('min_length', 0)
        self.max_seq_length = max_seq_length

        # Get the sequence length of the (temporally trimmed) satellite image time series
        # (used to compute the size of the model, cf. write_model_structure_to_file() in lib/utils.py)
        self.seq_length = split2seq_length[self.split] if self.max_seq_length is None else self.max_seq_length

        # Temporal resolution
        self.t_frequency = 5

        # Offset between the first meteorological observation and the first S2 observation
        self.S2_day_offset = 4

        # -------------------------------------- Auxiliary data -------------------------------------- #
        # Save whether auxiliary data should be returned
        self.return_cloud_prob = return_cloud_prob if self.split == 'train' else False
        self.return_class_map = return_class_map if self.split == 'train' else False
        self.return_cloud_mask = return_cloud_mask

        # -------------------------------------- Data augmentation -------------------------------------- #
        if (self.split == 'train' and mode == 'val') or self.split != 'train':
            self.augment = False
        else:
            self.augment = augment
            self.augmentation_function = transforms.Compose([
                torch_transforms.Rotate(),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ])

        # -------------------------------------- List of samples -------------------------------------- #
        self.verbose = verbose

        # Open hdf5 file
        self.f = h5py.File(self.hdf5_file, 'r', libver='latest', swmr=True)

        # Get the paths of all data samples (including train/val split) and optionally remove short sequences
        self.paths, self.tiles2samples = self._get_data_samples()

        # Number of satellite image time series (= data samples)
        self.num_samples = len(self.paths)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(
            self, idx: int, t_sampled: Optional[Tensor] = None, t_masked: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        # For simulation:
        # t_sampled as additional input to make the temporal trimming deterministic
        # t_masked as additional input to make the random masking deterministic

        cc: Optional[Tensor] = None
        class_maps:  Optional[Tensor] = None

        if not self.preprocessed or (self.split == 'train' and self.mode in ['train', 'all']):
            # a) TRAINING sequence
            # b) VALIDATION/TEST sequence which is not yet preprocessed and dumped to disk
            sample = self.f[self.paths[idx]]

            # Load the entire S2 image time series to apply the same data augmentation to cloud-free and cloudy images,
            # H x W x C x T
            data = sample['highresdynamic'][:]

            # Reshape from (H x W x C x T) to (T x C x H x W)
            data = torch.from_numpy(np.transpose(data, (3, 2, 0, 1)))

            if self.augment:
                data = self.augmentation_function(data)

            if self.crop_settings.enabled:
                data = self.crop_function(data)

            # Extract the desired satellite image channels
            images = data[:, self.s2_channels, :, :].float()

            # Preprocess the Sentinel-2 images
            images[torch.isnan(images)] = 0
            images[images < 0] = 0
            images[images > 1] = 1

            # Temporally subsample/trim the sequence
            if t_sampled is None:
                t_sampled, masks_valid_obs = self._subsample_sequence(sample, seq_length=images.shape[0])
            else:
                masks_valid_obs = torch.ones(len(t_sampled),)

            frames_input = images[t_sampled, :, :, :].clone()
            frames_target = images[t_sampled, :, :, :].clone()

            # Extract auxiliary data: cloud probabilities, classification maps, cloud masks
            if self.return_cloud_prob:
                cc = data[t_sampled, 4, :, :].unsqueeze(1).float() / 100.0
            if self.return_class_map:
                class_maps = data[t_sampled, 5, :, :].unsqueeze(1)
            cloud_mask = data[t_sampled, -1, :, :].unsqueeze(1).float()

            if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
                cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

            # Extract the acquisition dates of the subsampled S2 observations in the sequence
            dates = self._get_dates_S2(self.paths[idx])
            dates = [dates[idx] for idx in t_sampled]

            # Generate masks
            if self.mask_kwargs is not None:
                t_masked, frames_input, masks = self._generate_masks(sample, frames_input, cloud_mask, t_masked)
            else:
                masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))

        else:
            sample = self.f[self.paths[idx]]

            # Load the sequence, T x C x H x W
            frames_target = torch.from_numpy(sample['frames_target'][:]).float()
            frames_input = frames_target.clone()
            masks_valid_obs = torch.ones(frames_input.shape[0],)

            # Load the acquisition dates
            dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['dates']]

            if self.return_cloud_prob:
                cc = torch.from_numpy(sample['cloud_prob'][:]).float()
            if self.return_class_map:
                class_maps = torch.from_numpy(sample['classification'][:])
            cloud_mask = torch.from_numpy(sample['cloud_mask'][:]).float()

            # Load the precomputed masks
            if self.mask_kwargs is not None:
                frames_input, masks = self._get_precomputed_masks(sample, frames_input, cloud_mask)
            else:
                masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))

        if '-mask' in self.channels and self.mask_kwargs is not None:
            frames_input = torch.cat((frames_input, masks), dim=1)

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = dataset_tools.get_position_for_positional_encoding(dates, 'day-within-sequence')

        # Get positions for positional encoding
        position_days = dataset_tools.get_position_for_positional_encoding(dates, self.pe_strategy)

        # Assemble output
        out = {
            'x': frames_input,  # (synthetically masked) satellite image time series, (T x C x H x W)
            'y': frames_target,  # observed/target satellite image time series, (T x C x H x W)
            'masks': masks,  # masks applied to `x`, (T x 1 x H x W); pixel with value 1 is masked, 0 otherwise
            'masks_valid_obs': masks_valid_obs,  # flag to indicate valid time steps, (T, ); 1 if valid, 0 if invalid
            'position_days': position_days,
            'days': days,    # temporal sampling, number of days since the first observation in the sequence, (T, )
            'sample_index': idx,
            'filepath': self.paths[idx],
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir
        }

        if self.return_cloud_prob:
            out['cloud_prob'] = cc  # Sen2Cor Cloud Mask (CLD), cloud probability per pixel [0-1], (T x 1 x H x W)
        if self.return_class_map:
            out['classification'] = class_maps  # ESA Scene Classification (SCL), 0-11 categories, (T x 1 x H x W)
        if self.return_cloud_mask:
            out['cloud_mask'] = cloud_mask  # 0: non-occluded pixel, 1: occluded pixel, (T x 1 x H x W), w.r.t. `y`

        if self.to_export:
            # List of str, acquisition date for every observation in the sequence
            out['dates'] = [date.strftime('%Y-%m-%d') for date in dates]

            out['to_export'] = {'t_sampled': t_sampled}
            if self.mask_kwargs is not None and t_masked is not None:
                # pyre-ignore[16]: `Optional` has no attribute `__getitem__`.
                out['to_export']['indices_masked'] = torch.Tensor(t_masked['indices_masked'])
                if self.mask_kwargs.get('ratio_fully_masked_frames', 0) > 0:
                    out['to_export']['indices_fully_masked'] = torch.Tensor(t_masked['indices_fully_masked'])

        return out

    def _generate_masks(
            self,
            sample: h5py._hl.group.Group,
            frames_input: Tensor,
            cloud_mask_input: Tensor,
            t_masked: Dict[str, np.ndarray] | None = None
    ) -> Tuple[Dict[str, np.ndarray], Tensor, Tensor]:
        """
        Uses a sequence of masks (randomly generated or actual cloud mask sequence) to synthetically generate data gaps
        in the given satellite image time series.

        Args:
            sample:             h5py group.
            frames_input:       torch.Tensor, (T x C x H x W), temporally trimmed (subsampled) input image time series.
            cloud_mask_input:   torch.Tensor, (T x 1 x H x W), cloud masks associated with `frames_input`.
            t_masked:           dict, optional, defines two mutually exclusive sets of frame indices:
                                    'indices_masked':        np.ndarray, indices of the (partially) masked frames.
                                    'indices_fully_masked':  np.ndarray, indices of fully masked frames.

        Returns:
            t_masked:           dict, defines two mutually exclusive sets of frame indices:
                                    'indices_masked':        np.ndarray, indices of (partially) masked frames.
                                    'indices_fully_masked':  np.ndarray, indices of fully masked frames.
            frames_input:       torch.Tensor, (T x C x H x W), randomly masked image time series `frames_input`.
            masks:              torch.Tensor, (T x C x H x W), corresponding sequence of masks.
        """

        if t_masked is None and self.mask_kwargs.mask_type != 'real_clouds':
            # Indices of the frames to be masked w.r.t. the temporally trimmed sequence
            t_masked = dataset_tools.sample_indices_masked_frames(
                idx_valid_input_frames=np.arange(0, frames_input.shape[0]),
                ratio_masked_frames=self.mask_kwargs.ratio_masked_frames,
                ratio_fully_masked_frames=self.mask_kwargs.ratio_fully_masked_frames,
                non_masked_frames=self.mask_kwargs.non_masked_frames,
                fixed_masking_ratio=self.fixed_masking_ratio
            )

        if self.mask_kwargs.mask_type == 'random_clouds':
            # Randomly sample cloud masks
            sampled_clouds = self._sample_cloud_masks_from_tiles(sample, len(t_masked['indices_masked']))

            # Generate a sequence of masks
            masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))
            masks[t_masked['indices_masked'], :, :, :] = sampled_clouds

            # Intersect the randomly generated sequence of cloud masks with the actual cloud masks of the sequence
            if self.intersect_real_cloud_masks:
                masks = self._intersect_masks(masks, cloud_mask_input)

            # Apply masking
            frames_input, masks = overlay_seq_with_clouds(
                frames_input, masks, t_masked=None, fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks
            )

        elif self.mask_kwargs.mask_type == 'real_clouds':
            # Use the real cloud masks for masking
            frames_input, masks = masks_init_filling(
                frames_input, cloud_mask_input, None, fill_type='fill_value', fill_value=self.fill_value,
                dilate_cloud_masks=self.dilate_cloud_masks
            )
        else:
            raise NotImplementedError

        return t_masked, frames_input, masks

    def _get_precomputed_masks(
            self, sample: h5py._hl.group.Group, frames_input: Tensor, cloud_mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Loads precomputed masks previously dumped to a hdf5 file and generates the masked image time series.

        Args:
            sample:         h5py group.
            frames_input:   torch.Tensor, (T x C x H x W), input image time series.
            cloud_mask:     torch.Tensor, (T x 1 x H x W), time series of cloud masks.

        Returns:
            frames_input:   torch.Tensor, (T x C x H x W), masked image time series.
        """

        mask_dir, mask_name = dataset_tools.get_mask_sampling_id_hdf5(self.mask_kwargs)
        masks = torch.from_numpy(sample[mask_dir][mask_name][:]).float()

        # Intersect the randomly generated cloud mask sequence with the actual cloud masks of the input image time
        # series
        if self.mask_kwargs.mask_type == 'random_clouds' and self.intersect_real_cloud_masks:
            masks = self._intersect_masks(masks, cloud_mask)

        # Mask the input image time series
        frames_input, masks = overlay_seq_with_clouds(
            frames_input, masks, t_masked=None, fill_value=self.fill_value, dilate_cloud_masks=self.dilate_cloud_masks
        )

        return frames_input, masks

    def _get_data_samples(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Returns the paths of all data samples stored in the given hdf5 file `self.f`, where short cloud-free sequences
        are ignored (if `self.filter_settings.min_length` is specified and `self.filter_settings.type` is not None).

        Returns:
            paths:          list of str, paths of the valid data samples.
            tiles2samples:  dict, mapping from tiles to valid data samples:
                                keys:    str, tile names.
                                values:  list of str, paths of the valid data samples within the respective tile.
        """

        name = self.mode if self.split == 'train' else self.split

        if self.filter_settings is not None and self.filter_settings.type == 'cloud-free' and \
                f'path_samples_{name}' in self.f.keys():
            # Load the list of sample paths previously dumped to the hdf5 file
            paths = [path.decode("utf-8") for path in self.f[f'path_samples_{name}']]
        else:
            groups = []
            paths = []

            if self.split == 'train':
                if self.mode in ['train', 'val']:
                    if 'samples' in self.f[self.mode]:
                        groups.append(self.f[self.mode]['samples'])
                    else:
                        groups.append(self.f[self.mode])
                elif self.mode == 'all':
                    # Iterate over all samples from both the training and the validation split
                    if 'samples' in self.f['train']:
                        groups.append(self.f['train']['samples'])
                        groups.append(self.f['val']['samples'])
                    else:
                        groups.append(self.f['train'])
                        groups.append(self.f['val'])
            else:
                if 'samples' in self.f[name]:
                    groups.append(self.f[name]['samples'])
                else:
                    groups.append(self.f[name])

            for grp in groups:
                # Get the paths of all data samples belonging to `grp`
                paths = paths + self._get_sample_paths_from_tiles(grp)

        # Pre-compute the mapping from tiles to valid samples
        tiles2samples = {}
        for path in paths:
            parent_name = str(Path(path).parent)

            if parent_name not in tiles2samples:
                tiles2samples[parent_name] = [path]
            else:
                tiles2samples[parent_name].append(path)

        return paths, tiles2samples

    def _get_sample_paths_from_tiles(self, grp: h5py._hl.group.Group) -> List[str]:
        """
        Recursively visits every key of the h5py group `grp`. Returns a list of all keys that refer to a valid data
        sample. Definition of a valid sample: see documentation of _valid_sample().

        Args:
            grp:    h5py group.

        Returns:
            paths:  list of str, keys of valid data samples.
        """

        paths = []
        grp.visit(lambda key: paths.append(grp[key].name) if self._valid_sample(grp[key]) else None)

        return paths

    def _get_dates_S2(self, sample_path: str) -> List[dt.date]:
        """
        Extracts the acquisition date for every observation in the sequence.

        Args:
            sample_path: str, sample path.

        Returns:
            dates:       list of datetime.date dates, acquisition date of every observation in the sequence.
        """

        # Extract the acquisition date of the first and the last S2 observation in the sequence
        start_date, end_date = os.path.basename(sample_path).split('_')[1:3]
        start_date = dataset_tools.str2date(start_date)
        end_date = dataset_tools.str2date(end_date)

        # Start date w.r.t. meteorological variables; the first S2 observation has an offset of `self.S2_day_offset`
        # days
        start_date = start_date + dt.timedelta(days=self.S2_day_offset)

        # Get the acquisition dates of all intermediate S2 observations
        dates = [
            start_date + dt.timedelta(days=i) for i in range(0, (end_date - start_date).days + 1, self.t_frequency)
        ]

        return dates

    def _intersect_masks(self, masks: torch.Tensor, cloud_mask: torch.Tensor) -> torch.Tensor:
        """
        Intersects a randomly generated sequence of cloud masks `masks` with the actual cloud mask sequence of the
        image time series to be masked.

        Args:
            masks:      torch.Tensor, (T x 1 x H x W), sequence of randomly sampled cloud masks.
            cloud_mask: torch.Tensor, (T x 1 x H x W), actual time series of cloud masks.

        Returns:
            masks:      torch.Tensor, (T x 1 x H x W), intersection of `masks` with `cloud_mask`.
        """

        assert masks[0].shape == cloud_mask[0].shape, 'Cannot intersect two sequences of masks with unequal temporal ' \
                                                      'shape.'
        assert masks[-2:].shape == cloud_mask[-2:].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spatial shape.'
        assert masks[1].shape == cloud_mask[1].shape, 'Cannot intersect two sequences of masks with unequal ' \
                                                      'spectral shape.'

        masks[torch.logical_or(masks > 0., cloud_mask == 1)] = 1

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            masks = self._mask_images_with_cloud_coverage_above_p(masks)

        return masks

    def _mask_images_with_cloud_coverage_above_p(self, cloud_mask: torch.Tensor) -> torch.Tensor:
        """
        Marks all pixels of an image as occluded if its cloud coverage exceeds `self.render_occluded_above_p` [-].

        Args:
            cloud_mask: torch.Tensor, (T x 1 x H x W), time series of cloud masks.

        Returns:
            cloud_mask: torch.Tensor, (T x 1 x H x W), updated time series of cloud masks.
        """

        coverage = torch.mean(cloud_mask, dim=(1, 2, 3))
        cloud_mask[coverage > self.render_occluded_above_p, :, :, :] = 1

        return cloud_mask


    def _sample_cloud_masks_from_tiles(self, sample: h5py._hl.group.Group, n: int, p: float = 0.1) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            sample:  h5py group.
            n:       int, number of cloud masks to be sampled.
            p:       float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """

        # Extract all samples that originate from the same tile as the given input sample
        samples = self.tiles2samples[sample.parent.name]

        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn image time series, H x W x 1 x T
            seq = torch.from_numpy(self.f[random.choice(samples)]['highresdynamic'][:, :, [-1], :]).float()

            if self.crop_settings.enabled:
                seq = self.crop_function(seq)

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(0, 1, 2))

            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[:, :, :, np.random.choice(indices)])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=3).permute(3, 2, 0, 1)

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask


    def _subsample_sequence(self, sample: h5py._hl.group.Group, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) removes invalid time steps (i.e., no observation, black image),
        3) trims the sequence to a maximum temporal length.

        Args:
            sample:           h5py group.
            seq_length:       int, temporal length of the sample.

        Returns:
            t_sampled:        torch.Tensor, length T.
            masks_valid_obs:  torch.Tensor, (T, ).
        """

        # Generate a mask to exclude invalid frames:
        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        if self.filter_settings.type == 'cloud-free':
            # Indices of available and cloud-free images
            masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])

        elif self.filter_settings.type == 'cloud-free_consecutive':
            subseq = self._longest_consecutive_seq(sample)
            masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])
            masks_valid_obs[:subseq['start']] = 0
            masks_valid_obs[subseq['end'] + 1:] = 0
        else:
            masks_valid_obs = torch.ones(seq_length, )

        if self.filter_settings.get('return_valid_obs_only', True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled, masks_valid_obs[t_sampled]


    def _valid_sample(self, sample: h5py._hl.group.Group) -> bool:
        """
        Determines whether the h5py group `sample` defines a valid data sample. The following conditions have to be met:
        (i)  The h5py group stores satellite imagery, i.e., it has a key 'highresdynamic' that refers to a h5py Dataset.
        (ii) If cloud filtering is enabled: the satellite image time series has a minimal sequence length, where
             the minimal length is given by self.filter_settings.min_length.

        Args:
            sample: h5py group.

        Returns:
            bool, True if the h5py group `sample` is a valid data sample, False otherwise.
        """

        if isinstance(sample, h5py.Group) and 'highresdynamic' in sample.keys():
            if self.filter_settings.type is None:
                return True
            if self.filter_settings.type == 'cloud-free':
                seq_length = sample['idx_good_frames'].size

                # Check number of valid frames
                if seq_length >= self.filter_settings.min_length:
                    if self.filter_settings.max_num_consec_invalid is not None:
                        # Check number of consecutive invalid frames
                        max_num_consec_invalid = self._count_max_num_consecutive_invalid_frames(sample)

                        if max_num_consec_invalid <= self.filter_settings.max_num_consec_invalid:
                            # (i) Minimal sequence length is ok, and (ii) max. number of consecutive invalid frames
                            # is below the threshold
                            return True
                        if self.verbose == 1:
                            print(f"Too many consecutive invalid frames within the sequence "
                                  f"({max_num_consec_invalid} < {self.filter_settings.max_num_consec_invalid}): "
                                  f"{sample.name}")
                        return False

                    # Minimal sequence length is ok
                    return True
                if self.verbose == 1:
                    print(f"Too short sequence ({seq_length} < {self.filter_settings.min_length}): {sample.name}")
                return False
            if self.filter_settings.type == 'cloud-free_consecutive':
                seq_length = self._longest_consecutive_seq(sample)['len']
                if seq_length >= self.filter_settings.min_length:
                    return True
                if self.verbose == 1:
                    print(f"Too short sequence ({seq_length} < {self.filter_settings.min_length}): {sample.name}")
                return False
            raise NotImplementedError(f'Unknown sequence filter {self.filter_settings.type}.\n')

        return False

    @staticmethod
    def _count_max_num_consecutive_invalid_frames(sample: h5py._hl.group.Group) -> float:
        """
        Counts the maximum number of consecutive invalid (cloudy/foggy/unavailable) images within an image time series.

        Args:
            sample: h5py group storing satellite imagery.

        Returns:
            float, maximum number of consecutive invalid images within the given satellite image time series.
        """

        masks_valid_obs = sample['valid_obs'][:]

        count = 0
        max_count = -math.inf

        for _, frame_status in enumerate(masks_valid_obs):
            if frame_status == 0:
                count += 1
            elif frame_status == 1 and count == 0:
                pass
            else:
                max_count = max(max_count, count)
                count = 0

        return max(max_count, count)

    @staticmethod
    def _longest_consecutive_seq(sample: h5py._hl.group.Group) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images.

        Args:
            sample:      h5py group.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """

        idx_frames = sample['idx_good_frames'][:]

        # Count number of consecutive cloud-free images
        subseq = {'start': 0, 'end': 0, 'len': 0}
        count = 1
        start = 0

        for i in range(len(idx_frames) - 1):
            if idx_frames[i] + 1 == idx_frames[i + 1]:
                end = i + 1
                count += 1
                if count > subseq['len']:
                    subseq['start'] = idx_frames[start]
                    subseq['end'] = idx_frames[end]
                    subseq['len'] = count
            else:
                start = i + 1
                count = 1

        return subseq
