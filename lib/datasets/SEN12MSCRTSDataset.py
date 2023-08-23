import datetime as dt
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
from lib.datasets.EarthNet2021Dataset import EarthNet2021Dataset

SPLITS = ['train', 'val', 'test']
CHANNEL_CONFIG = ['bgr', 'bgr-nir', 'all', 'bgr-mask', 'bgr-nir-mask', 'all-mask']


class SEN12MSCRTSDataset(EarthNet2021Dataset):
    """
    torch.utils.data.Dataset class for the SEN12MS-CR-TS dataset.

    Dataset source:
    P. Ebel, Y. Xu, M. Schmitt, and X. X. Zhu, “SEN12MS-CR-TS: A remote-sensing data set for multimodal multitemporal
    cloud removal”, IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1-14, 2022


    Args:
        root:                  str, data root directory (must contain a hdf5 file storing the SEN12MS-CR-TS dataset).
        hdf5_file:             None or str:
                                   None:   The hdf5 filename is identical to the naming convention of the data split.
                                           E.g., train.hdf5
                                   str:    hdf5 filepath w.r.t. the `root` directory.
        preprocessed:          bool, True to use preprocessed validation/test data, i.e., the temporal trimming
                               and mask generation are loaded from the hdf5 file instead of computed on-the-fly
                               (automatically switched off for training data).

        split:                 str, data split, choose among ['train', 'val', 'test'].
        channels:              str, Sentinel-2 channels to be extracted, choose among
                               ['bgr', 'bgr-nir', 'all', 'bgr-mask', 'bgr-nir-mask', 'all-mask'].
        include_S1:            bool, True to concatenate the desired Sentinel-2 channels (cf. `channels`) with
                               corresponding Sentinel-1 observations; False to load Sentinel-2 observations only.

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

                                    'min_length':             int, minimal sequence length. Removes cloud-free image
                                                              sequences consisting of less than `min_length` images.
                                    'return_valid_obs_only':  bool, True to return only those images per sequence for
                                                              which `masks_valid_obs[t]` == 1;
                                                              CONSEQUENCE: variable sequence lengths across samples.
                                    'max_num_consec_invalid': int, maximum number of consecutive invalid images per
                                                              sequence. Sequences with more than
                                                              `max_num_consec_invalid` consecutive invalid images
                                                              (cloudy/unavailable) are ignored.
                                    'max_t_sampling':         int, maximum temporal difference [days] between consecutive
                                                              (cloud-free) images. Time series which do not have at
                                                              least 'min_length' consecutive (cloud-free) images with
                                                              at most 'max_t_sampling' days between consecutive frames
                                                              will be ignored. Set to None to deactivate this filter.

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

        render_occluded_above_p:  float, mark an entire image as occluded if its cloud coverage exceeds
                                  `render_occluded_above_p` [-] (i.e., the cloud mask of the respective image is
                                  overwritten). Set `render_occluded_above_p` to None to retain the original
                                  cloud masks.
        return_cloud_prob:        bool, True to return the s2cloudless cloud probability maps ([0-1]) as `cloud_prob`.
        return_cloud_mask:        bool, True to return the s2cloudless cloud masks (0 if non-cloudy, 1 if cloudy)
                                  as `cloud_mask`.
        return_acquisition_dates: bool, True to return the acquisition dates for every Sentinel-2 observation
                                  (and Sentinel-1 if `include_S1` is True).
        augment:                  bool, True to activate data augmentation (random rotation by multiples of 90 degrees
                                  as well as random flipping along the horizontal and vertical axes), False otherwise.
        max_seq_length:           int, randomly selects a subsequence consisting of `max_seq_length` images.
                                  Image time series shorter than `max_seq_length` remain unmodified.
                                  Set `max_seq_length` to None to retain the original sequences after cloud filtering.
        ignore_seq_with_black_images: bool, True to ignore sequences with at least one black image, False otherwise.
        to_export:                bool, True to return additional parameters used in the simulation for reproducibility
                                  purposes, False otherwise.
    """

    def __init__(self,
                 root: str,
                 hdf5_file: str | Path | None = None,
                 preprocessed: bool = False,
                 split: str = 'train',
                 channels: str = 'all',
                 include_S1: bool = False,
                 filter_settings: Optional[Dict | DictConfig] = None,
                 crop_settings: Optional[Dict | DictConfig] = None,
                 pe_strategy: str = 'day-of-year',
                 mask_kwargs: Optional[Dict | DictConfig] = None,
                 render_occluded_above_p: Optional[float] = None,
                 return_cloud_prob: bool = False,
                 return_cloud_mask: bool = True,
                 return_acquisition_dates: bool = False,
                 augment: bool = False,
                 max_seq_length: Optional[int] = None,
                 ignore_seq_with_black_images: bool = False,
                 verbose: int = 0,
                 to_export: bool = False
                 ):

        if filter_settings is None:
            filter_settings = {'type': None, 'min_length': 5, 'return_valid_obs_only': False, 'max_t_sampling': None}
        if crop_settings is None:
            crop_settings = {'enabled': False, 'type': 'random', 'shape': (64, 64)}

        # -------------------------------------- Verify input parameters -------------------------------------- #

        if not os.path.exists(root):
            raise FileNotFoundError(f"Invalid `root`. Root directory does not exist: {root}")

        if split not in SPLITS:
            raise ValueError(f"Invalid `split` parameter. Choose among {SPLITS} to specify `split`.\n")

        if hdf5_file is None:
            hdf5_file = Path(root) / (split + '.hdf5')
        else:
            hdf5_file = Path(root) / hdf5_file

        if not os.path.exists(hdf5_file):
            raise FileNotFoundError(f"Cannot find the hdf5 file: {hdf5_file}")

        if isinstance(filter_settings, dict):
            filter_settings = OmegaConf.create(filter_settings)

        if isinstance(crop_settings, dict):
            crop_settings = OmegaConf.create(crop_settings)

        if isinstance(mask_kwargs, dict):
            mask_kwargs = OmegaConf.create(mask_kwargs)

        if not isinstance(filter_settings, DictConfig) or 'type' not in filter_settings or \
                filter_settings.type not in [None, 'cloud-free', 'cloud-free_consecutive'] or \
                ('min_length' in filter_settings and not isinstance(filter_settings.min_length, int)) or \
                ('max_num_consec_invalid' in filter_settings and not isinstance(
                    filter_settings.max_num_consec_invalid, int)) or \
                ('max_t_sampling' in filter_settings and (
                        not isinstance(filter_settings.max_t_sampling, int) and filter_settings.max_t_sampling is not None
                )):
            raise RuntimeError(
                "Invalid `filter_settings` parameter. Define a dictionary with the following keys and value options:\n"
                "'type': None or str,           # sequence filter type, choose from [None, 'cloud-free', "
                "'cloud-free_consecutive']\n "
                "'min_length': int,            # minimum number of frames per sequence\n"
                "'return_valid_obs_only': bool  # True to return filtered frames only, False otherwise\n "
                "'max_num_consec_invalid': int # maximum number of consecutive invalid frames per sequence\n"
                "'max_t_sampling': int          # maximum number of days between consecutive cloud-free frames\n"
            )

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

        if not isinstance(return_cloud_mask, bool):
            raise TypeError("Invalid `return_cloud_mask` parameter. Specify a boolean.")

        if not isinstance(augment, bool):
            raise TypeError("Invalid `augment` parameter. Specify a boolean.")

        if not isinstance(include_S1, bool):
            raise TypeError("Invalid `include_S1` parameter. Specify a boolean.")

        if not isinstance(return_acquisition_dates, bool):
            raise TypeError("Invalid `return_acquisition_dates` parameter. Specify a boolean.")

        # -------------------------------------- Data and split -------------------------------------- #
        self.hdf5_file = hdf5_file
        self.root = root
        self.split = split
    
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
        self.return_acquisition_dates = return_acquisition_dates

        # -------------------------------------- Channel settings -------------------------------------- #
        # Image channels and/or composites
        if channels not in CHANNEL_CONFIG:
            raise ValueError(f"Unknown channel configuration `{channels}`. Choose among {CHANNEL_CONFIG} to "
                             "specify `channels`.\n")

        self.channels = channels
        self.include_S1 = include_S1

        # Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
        # self.channels: used to extract the relevant channels from the hdf5 file
        # self.c_index_rgb and self.c_index_nir: indices of the RGB (B2, B3, B4) and NIR channels (B8), w.r.t. the
        # output of the self.__getitem__() call
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [1, 2, 3]                         # B2, B3, B4
        else:
            # self.channels in ['all', 'all-mask']
            self.num_channels = 13
            self.c_index_rgb = torch.Tensor([3, 2, 1]).long()
            self.s2_channels = list(np.arange(13))               # all 13 bands

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [7]                              # B8
        elif 'all' in self.channels:
            self.c_index_nir = torch.Tensor([7]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        if '-mask' in self.channels:
            # self.channels in ['bgr-mask', 'bgr-nir-mask', 'all-mask']
            self.num_channels += 1

        if self.include_S1:
            self.num_channels += 2

        # -------------------------------------- Spatial cropping settings -------------------------------------- #
        self.crop_settings = OmegaConf.create(crop_settings)

        # Image size
        self.image_size = crop_settings.shape if self.crop_settings.enabled else (256, 256)

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
        self.filter_settings.max_t_sampling = self.filter_settings.get('max_t_sampling', None)
        self.max_seq_length = max_seq_length

        # Get the sequence length of the (temporally trimmed) satellite image time series
        # (used to compute the size of the model, cf. write_model_structure_to_file() in lib/utils.py)
        self.seq_length = 30 if self.max_seq_length is None else self.max_seq_length

        self.ignore_seq_with_black_images = ignore_seq_with_black_images

        # -------------------------------------- Auxiliary data -------------------------------------- #
        # Save whether auxiliary data should be returned
        self.return_cloud_prob = return_cloud_prob
        self.return_cloud_mask = return_cloud_mask

        # -------------------------------------- Data augmentation -------------------------------------- #
        if self.split != 'train':
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

        # Get the paths of all data samples and optionally remove short sequences
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
        s1: Optional[Tensor] = None
        s1_dates: Optional[List[dt.date]] = None

        if not self.preprocessed or self.split == 'train':
            # a) TRAINING sequence
            # b) VALIDATION/TEST sequence which is not yet preprocessed and dumped to disk
            patch = self.f[self.paths[idx]]

            # Load the entire S2 satellite image time series, T x C x H x W
            data = torch.from_numpy(patch['S2/S2'][:, self.s2_channels, :, :].astype(np.float32))

            # Load the cloud masks, T x 1 x H x W
            data = torch.cat((data, torch.from_numpy(patch['S2/cloud_mask'][:]).float()), dim=1)

            if self.return_cloud_prob:
                # Load the cloud probability maps, T x 1 x H x W
                data = torch.cat((data, torch.from_numpy(patch['S2/cloud_prob'][:])), dim=1)

            # Optionally load the entire S1 satellite image time series, T x 2 x H x W
            if self.include_S1:
                s1 = torch.from_numpy(patch['S1/S1'][:])
                data = torch.cat((data, s1), dim=1)

            if self.augment:
                data = self.augmentation_function(data)

            if self.crop_settings.enabled:
                data = self.crop_function(data)

            # Temporally subsample/trim the sequence
            if t_sampled is None:
                t_sampled, masks_valid_obs = self._subsample_sequence(patch, seq_length=data.shape[0])
            else:
                masks_valid_obs = torch.ones(len(t_sampled), )

            # Extract the desired S2 satellite image channels and the subsampled time steps
            s2 = self._process_MS(data[:, :len(self.s2_channels), :, :])
            frames_input = s2[t_sampled, :, :, :].clone()
            frames_target = s2[t_sampled, :, :, :].clone()

            # Extract the cloud masks of the temporally trimmed S2 image sequence and optionally the cloud
            # probability maps
            cloud_mask = data[:, len(self.s2_channels), :, :].unsqueeze(1)[t_sampled, :, :, :]
            if self.return_cloud_prob:
                cc = data[:, len(self.s2_channels) + 1, :, :].unsqueeze(1)[t_sampled, :, :, :]

            if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
                cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

            # Extract the acquisition dates of the temporally trimmed S2 image sequence
            s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in patch['S2/S2_dates']]
            s2_dates = [s2_dates[t] for t in t_sampled]

            # Extract the temporally subsampled S1 satellite image sequence
            if self.include_S1:
                s1 = self._process_SAR(data[:, -2:, :, :][t_sampled, :, :, :])
                s1_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in patch['S1/S1_dates']]
                s1_dates = [s1_dates[t] for t in t_sampled]

            # Generate masks
            if self.mask_kwargs is not None:
                t_masked, frames_input, masks = self._generate_masks(patch, frames_input, cloud_mask, t_masked)
            else:
                masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))

        else:
            sample = self.f[self.paths[idx]]

            # Load the S2 satellite image time series along with their cloud masks, T x C x H x W
            # NOTE: S2 imagery is already preprocessed, i.e., no need to call self._process_MS()
            frames_target = torch.from_numpy(sample['S2/S2'][:])
            frames_input = frames_target.clone()
            masks_valid_obs = torch.ones(frames_input.shape[0], )
            cloud_mask = torch.from_numpy(sample['S2/cloud_mask'][:]).float()

            # Load the acquisition dates
            s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S2/S2_dates']]

            if self.return_cloud_prob:
                cc = torch.from_numpy(sample['S2/cloud_prob'][:])

            # Load the precomputed masks
            if self.mask_kwargs is not None:
                frames_input, masks = self._get_precomputed_masks(sample, frames_input, cloud_mask)
            else:
                masks = torch.zeros((frames_input.shape[0], 1, *frames_input.shape[-2:]))

            # Optionally load the S1 satellite image time series, T x 2 x H x W
            if self.include_S1:
                # NOTE: S1 imagery is already preprocessed, i.e., no need to call self._process_SAR()
                s1 = torch.from_numpy(sample['S1/S1'][:])
                s1_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S1/S1_dates']]

        if '-mask' in self.channels and self.mask_kwargs is not None:
            frames_input = torch.cat((frames_input, masks), dim=1)

        if self.include_S1:
            # Concatenate the (masked) S2 bands and the unmasked S1 bands
            frames_input = torch.cat((frames_input, s1), dim=1)

        # Extract the number of days since the first observation in the sequence (= temporal sampling)
        days = dataset_tools.get_position_for_positional_encoding(s2_dates, 'day-within-sequence')

        # Get positions for positional encoding
        position_days = dataset_tools.get_position_for_positional_encoding(s2_dates, self.pe_strategy)

        # Assemble output
        out = {
            'x': frames_input,  # (synthetically masked) S2 satellite image time series, (T x C x H x W), optionally including S1 bands
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
            out['cloud_prob'] = cc  # cloud probability per pixel [0-1], (T x 1 x H x W)
        if self.return_cloud_mask:
            out['cloud_mask'] = cloud_mask  # 0: non-occluded pixel, 1: occluded pixel, (T x 1 x H x W), w.r.t. `y`

        if self.return_acquisition_dates:
            # List of str, acquisition date for every Sentinel-2 (Sentinel-1) observation in the sequence
            out['S2_dates'] = [date.strftime('%Y-%m-%d') for date in s2_dates]
            if self.include_S1:
                out['S1_dates'] = [date.strftime('%Y-%m-%d') for date in s1_dates]

        if self.to_export:
            # List of str, acquisition date for every Sentinel-2 (Sentinel-1) observation in the sequence
            if 'S2_dates' not in out:
                out['S2_dates'] = [date.strftime('%Y-%m-%d') for date in s2_dates]

            if self.include_S1:
                out['S1'] = s1
                if 'S1_dates' not in out:
                    out['S1_dates'] = [date.strftime('%Y-%m-%d') for date in s1_dates]

            out['to_export'] = {'t_sampled': t_sampled}
            if self.mask_kwargs is not None and t_masked is not None:
                out['to_export']['indices_masked'] = torch.Tensor(t_masked['indices_masked'])
                if self.mask_kwargs.get('ratio_fully_masked_frames', 0) > 0:
                    out['to_export']['indices_fully_masked'] = torch.Tensor(t_masked['indices_fully_masked'])

        return out

    def _get_data_samples(self) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Returns the paths of all data samples stored in the given hdf5 file `self.f`. Sequences are excluded if
        (i)  they comprise less than `self.filter_settings.min_length` cloud-free images (if `self.filter_settings.type`
             is not None),
        (ii) or if the subsequence of cloud-free images does not include a minimum of `self.filter_settings.min_length`
             consecutive cloud-free images with at most `self.filter_settings.max_t_sampling` days between
             observations.

        Returns:
            paths:          list of str, paths of the valid data samples (patches with S2 and/or S1 satellite imagery).
            tiles2samples:  dict, mapping from tiles to valid data samples:
                                keys:    str, tile names.
                                values:  list of str, paths of the valid data samples within the respective tile.
        """

        skip_parsing = False

        if self.filter_settings is not None and self.filter_settings.type == 'cloud-free' and \
                self.filter_settings.max_t_sampling is not None and self.ignore_seq_with_black_images:

            list_paths = f'path_samples_max_t_sampling_{self.filter_settings.max_t_sampling}_wo_black_images'
            if list_paths in self.f.keys():
                paths = [path.decode("utf-8") for path in self.f[list_paths]]
                skip_parsing = True

        if not skip_parsing:
            if self.filter_settings is not None and self.filter_settings.type == 'cloud-free' and \
                    'path_samples' in self.f.keys():
                # Load the list of sample paths previously dumped to the hdf5 file
                paths = [path.decode("utf-8") for path in self.f['path_samples']]

                # Check that all samples are valid w.r.t. `self.filter_settings.max_t_sampling`
                if self.filter_settings.max_t_sampling is not None and self.split == 'train':
                    invalid = []
                    for p in paths:
                        if not self._valid_sample(self.f[p]):
                            invalid.append(p)

                    # Delete invalid samples
                    for p in invalid:
                        paths.remove(p)
            else:
                paths = []

                # Iterate over ROIs
                for roi_name, roi in self.f.items():
                    if 'ROIs' not in roi_name:
                        pass
                    else:
                        # Get the paths of all data samples belonging to the current ROI
                        paths = paths + self._get_sample_paths_from_roi(roi)

            if self.ignore_seq_with_black_images:
                # Filter out sequences with at least one black image
                invalid = []
                for p in paths:
                    s2 = torch.from_numpy(self.f[p]['S2/S2'][:].astype(np.float32))
                    if not self.preprocessed:
                        s2 = self._process_MS(s2)
                    if torch.any(s2.sum(dim=-1).sum(dim=-1).sum(dim=-1) == 0.):
                        invalid.append(p)
                for p in invalid:
                    paths.remove(p)

        # Pre-compute the mapping from tiles to valid samples
        tiles2samples = {}
        for path in paths:
            parent_name = str(Path(path).parent)

            if parent_name not in tiles2samples:
                tiles2samples[parent_name] = [path]
            else:
                tiles2samples[parent_name].append(path)

        return paths, tiles2samples

    def _get_sample_paths_from_roi(self, roi: h5py._hl.group.Group) -> List[str]:
        """
        Recursively visits every key of the h5py group `roi`. Returns a list of all keys that refer to a valid data
        sample. Definition of a valid sample: see documentation of _valid_sample().

        Args:
            roi:    h5py group.

        Returns:
            paths:  list of str, keys of valid data samples.
        """

        paths = []

        for _, tile in roi.items():
            for _, patch in tile.items():
                if self._valid_sample(patch):
                    paths.append(patch.name)

        return paths

    def _process_MS(self, img: Tensor) -> Tensor:
        """Modified from: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L33"""

        # Intensity clipping to a global unified MS intensity range
        intensity_min, intensity_max = 0, 10000
        img = torch.clamp(img, min=intensity_min, max=intensity_max)

        # Project to [0,1], preserve global intensities (across patches)
        img = self._rescale(img, intensity_min, intensity_max)
        return img

    def _process_SAR(self, img: Tensor) -> Tensor:
        """Source: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L44"""

        # Intensity clipping to a global unified SAR dB range
        dB_min, dB_max = -25, 0
        img = torch.clamp(img, min=dB_min, max=dB_max)

        # Project to [0,1], preserve global intensities (across patches)
        img = self._rescale(img, dB_min, dB_max)
        return img

    @staticmethod
    def _rescale(img: Tensor, old_min: float, old_max: float) -> Tensor:
        """Source: https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/data/dataLoader.py#L28"""
        old_range = old_max - old_min
        img = (img - old_min) / old_range
        return img

    def _sample_cloud_masks_from_tiles(self, sample: h5py._hl.group.Group, n: int, p: float = 0.1) -> torch.Tensor:
        """
        Randomly samples `n` cloud masks from a given tile.

        Args:
            sample:  h5py group
            n:      int, number of cloud masks to be sampled.
            p:      float, minimum cloud coverage [-] of the sampled cloud masks.

        Returns:
            cloud_mask:  torch.Tensor, n x 1 x H x W, sampled cloud masks.
        """

        # Extract all samples that originate from the same tile as the given input sample
        samples = self.tiles2samples[sample.parent.name]

        # Randomly sample `n` cloud masks with cloud coverage of >= p
        cloud_mask = []
        while len(cloud_mask) < n:
            # Extract the cloud masks of a randomly drawn S2 image time series, T x 1 x H x W
            seq = torch.from_numpy(self.f[random.choice(samples)]['S2/cloud_mask'][:]).float()

            if self.crop_settings.enabled:
                seq = self.crop_function(seq)

            # Compute cloud coverage per frame
            coverage = torch.mean(seq, dim=(1, 2, 3))

            indices = torch.argwhere(coverage >= p).flatten()
            if len(indices) > 0:
                cloud_mask.append(seq[np.random.choice(indices), :, :, :])

        # n x 1 x H x W
        cloud_mask = torch.stack(cloud_mask, dim=0)

        if self.render_occluded_above_p and self.render_occluded_above_p > 0.:
            cloud_mask = self._mask_images_with_cloud_coverage_above_p(cloud_mask)

        return cloud_mask

    def _subsample_sequence(self, sample: h5py._hl.group.Group, seq_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
           images is at most `self.filter_settings.max_t_sampling` days,
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

            if self.filter_settings.max_t_sampling is not None:
                subseq = self._longest_consecutive_seq_within_sampling_frequency(sample)
                masks_valid_obs[:subseq['start']] = 0
                masks_valid_obs[subseq['end'] + 1:] = 0

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
        (i)   The h5py group stores S2 satellite imagery, i.e., it has a key 'S2' that refers to a h5py Dataset.
        (ii)  The h5py group stores S1 satellite imagery, i.e., it has a key 'S1' that refers to a h5py Dataset
              (only if self.include_S1 == True).
        (iii) If cloud filtering is enabled: the satellite image time series has a minimal sequence length, where
              the minimal length is given by `self.filter_settings.min_length`.
        (iv)  If cloud filtering is enabled and `self.filter_settings.max_t_sampling` is not None: the subsequence of
              cloud-free images includes at least `self.filter_settings.min_length` consecutive cloud-free images with
              at most `self.filter_settings.max_t_sampling` days between observations.

        Args:
            sample: h5py group.

        Returns:
            bool, True if the h5py group `sample` is a valid data sample, False otherwise.
        """

        if isinstance(sample, h5py.Group) and 'S2' in sample.keys():
            if self.include_S1 and 'S1' not in sample.keys():
                return False
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

                    if self.filter_settings.max_t_sampling is not None:
                        count = self._longest_consecutive_seq_within_sampling_frequency(sample)['len']
                        if count >= self.filter_settings.min_length:
                            return True
                        if self.verbose == 1:
                            print(f"Less than {self.filter_settings.min_length} consecutive cloud-free frames with "
                                  f"temporal sampling of at most {self.filter_settings.max_t_sampling} days: "
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

                    if self.filter_settings.max_t_sampling is not None:
                        count = self._longest_consecutive_seq_within_sampling_frequency(sample)['len']
                        if count >= self.filter_settings.min_length:
                            return True
                        if self.verbose == 1:
                            print(f"Less than {self.filter_settings.min_length} consecutive cloud-free frames with "
                                  f"temporal sampling of at most {self.filter_settings.max_t_sampling} days: "
                                  f"{sample.name}")
                        return False

                    return True
                if self.verbose == 1:
                    print(f"Too short sequence ({seq_length} < {self.filter_settings.min_length}): {sample.name}")
                return False
            raise NotImplementedError(f'Unknown sequence filter {self.filter_settings.type}.\n')

        return False

    def _longest_consecutive_seq_within_sampling_frequency(self, sample: h5py._hl.group.Group) -> Dict[str, int]:
        """
        Determines the longest subsequence of consecutive cloud-free images, where the temporal sampling between
        consecutive cloud-free images does not exceed `self.filter_settings.max_t_sampling` days.

        Args:
            sample:      h5py group.

        Returns:
            subseq:      dict, the longest subsequence of valid images. The dictionary has the following key-value
                         pairs:
                            'start':  int, index of the first image of the subsequence.
                            'end':    int, index of the last image of the subsequence.
                            'len':    int, temporal length of the subsequence.
        """

        # Extract the acquisition dates of the cloud-free images within the sequence
        s2_dates = [dataset_tools.str2date(date.decode("utf-8")) for date in sample['S2/S2_dates']]
        t_cloudfree = sample['idx_good_frames'][:]
        s2_dates = [s2_dates[t] for t in t_cloudfree]

        # Count number of consecutive cloud-free images with temporal sampling of at most
        # `self.filter_settings.max_t_sampling:`
        subseq = {'start': 0, 'end': 0, 'len': 0}
        count = 1
        start = 0

        for i in range(len(s2_dates) - 1):
            if (s2_dates[i+1] - s2_dates[i]).days <= self.filter_settings.max_t_sampling:
                end = i + 1
                count += 1
                if count > subseq['len']:
                    subseq['start'] = t_cloudfree[start]
                    subseq['end'] = t_cloudfree[end]
                    subseq['len'] = count
            else:
                start = i + 1
                count = 1

        return subseq
