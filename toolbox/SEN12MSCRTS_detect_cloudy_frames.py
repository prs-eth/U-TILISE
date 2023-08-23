import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from lib.datasets import SEN12MSCRTSDataset
from lib.datasets.dataset_tools import detect_impaired_frames

parser = ArgumentParser()
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--hdf5_filename', type=str, default=None)


def main(args):

    if args.hdf5_filename is None:
        hdf5_file = os.path.join(args.root, f'{args.split}.hdf5')
    else:
        hdf5_file = os.path.join(args.root, args.hdf5_filename)
    
    dset = SEN12MSCRTSDataset(root=args.root, hdf5_file=args.hdf5_filename, split=args.split)
    paths = dset.paths
    dset.f.close()
    del dset
    
    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        # Iterate over all data samples in the given data split: detection of cloudy/foggy frames
        for path in tqdm(paths):
            patch = f[path]

            # Load the entire S2 satellite image time series, T x C x H x W
            images = torch.from_numpy(patch['S2/S2'][:].astype(np.float32))
            T = images.shape[0]

            # Load the cloud masks, T x 1 x H x W
            cloud_mask = torch.from_numpy(patch['S2/cloud_mask'][:]).float()

            # Load the cloud probability maps and convert to percentage, T x 1 x H x W
            cloud_prob = torch.from_numpy(patch['S2/cloud_prob'][:]) * 100

            # Detect impaired frames (unavailable/foggy/cloudy)
            idx_impaired_frames, _ = detect_impaired_frames(
                images, cloud_prob, cloud_mask, increased_filter_strength=True
            )

            dset = patch.create_dataset(
                'idx_impaired_frames', data=idx_impaired_frames, compression='gzip', compression_opts=9
            )

            # Indices of foggy/cloudy (but available) frames
            idx_cloudy_frames = [i for i in idx_impaired_frames if ~torch.all(torch.isnan(images[i, ...]))]
            dset = patch.create_dataset(
                'idx_cloudy_frames', data=idx_cloudy_frames, compression='gzip', compression_opts=9
            )

            # Indices of available and cloud-free frames
            idx_good_frames = [i for i in range(T) if i not in idx_impaired_frames]
            dset = patch.create_dataset(
                'idx_good_frames', data=idx_good_frames, compression='gzip', compression_opts=9
            )

            # Save a flag for every time step to indicate whether the observation is valid or not
            # (1: available and cloud-free, 0: otherwise)
            valid_obs = [1 if i in idx_good_frames else 0 for i in range(T)]
            dset = patch.create_dataset(
                'valid_obs', data=valid_obs, compression='gzip', compression_opts=9
            )

    # Retrieve valid samples: sequences with at least 5 cloud-free images, where the temporal difference between
    # consecutive cloud-free images is at most 42 days (= 6 weeks)
    dataset2 = SEN12MSCRTSDataset(
        root=args.root, hdf5_file=args.hdf5_filename, split=args.split,
        filter_settings={'type': 'cloud-free', 'min_length': 5, 'max_t_sampling': 42, 'return_valid_obs_only': False},
        verbose=1
    )
    dataset2.f.close()

    # Dump the sample paths to a list to speed up data loading later on
    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        f.create_dataset('path_samples', data=dataset2.paths)

    print('Done')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        main(parser.parse_args())
