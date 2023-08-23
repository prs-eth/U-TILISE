import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from lib.datasets.dataset_tools import detect_impaired_frames
from lib.datasets.EarthNet2021Dataset import EarthNet2021Dataset
from lib.utils import set_seed

VALIDATION_SIZE = 5000
SPLITS = ['train', 'iid', 'ood', 'extreme', 'seasonal']


class EarthNet2021_npz2hdf5(torch.utils.data.Dataset):
    """
    Args:
        root:  str, root data directory
        split: EarthNet2021 data split, ['train', 'iid', 'ood', 'extreme', 'seasonal']
        mode:  str, data split mode for the training set, ['train', 'val', 'all']
    """

    def __init__(self, root: str, split: str = 'train', mode: Optional[str] = 'train'):

        if split not in SPLITS:
            raise ValueError(f"Invalid `split`. Choose among {SPLITS} to specify `split`.\n")

        if mode not in ['train', 'val', 'all']:
            raise ValueError("Invalid `mode`. Choose among ['train', 'val', 'all'] to specify `mode`.\n")

        if split == 'train':
            self.split = split
            self.root = os.path.join(root, split)
            self.mode = mode
        else:
            self.split = split + '_test_split'
            self.root = os.path.join(root, self.split, 'context')

        # Get the paths of the data samples (including train/val split) and optionally remove short sequences
        self.tiles, self.paths = self._get_data_samples_npz()

        # Number of sequences
        self.num_samples = len(self.paths)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, Union[np.ndarray, str]]:

        # Load the data multicube: H x W x C x T
        if self.split == 'train':
            filepath = self.paths[idx]
            multicube = np.load(filepath)['highresdynamic']

        else:
            filepath = self.paths[idx]
            filepath_target = filepath.replace('context', 'target')
            multicube = np.concatenate(
                (np.load(filepath)['highresdynamic'], np.load(filepath_target)['highresdynamic']),
                axis=3
            )

        filepath = filepath.replace(self.root + '/', '').replace('context_', '')

        # Return <tile_name>/<filename>
        p = Path(filepath)
        filepath_export = str(p.parent / p.stem)

        out = {
            'highresdynamic': multicube,
            'filepath_export': filepath_export
        }

        return out

    def _get_data_samples_npz(self) -> Tuple[List[str], List[str]]:
        # Get the paths of all samples in the given data split
        tiles, paths, sample_count = self._get_data_structure()

        if self.split == 'train':
            if self.mode in ['train', 'val']:
                # Split the training set into training and validation samples (roughly 80:20 ratio).
                tiles, paths = self._get_train_val_split(tiles, paths, sample_count, return_samples=self.mode)
            else:
                # Concatenate the filepaths across all tiles
                temp = []
                for _, files in paths.items():
                    temp = temp + files
                paths = temp.copy()

        else:
            # Concatenate the filepaths across all tiles
            temp = []
            for _, files in paths.items():
                temp = temp + files
            paths = temp.copy()

        return tiles, paths

    def _get_data_structure(self) -> Tuple[List[str], Dict[str, List[str]], np.ndarray]:
        """
        Returns the filepaths of all datacube files (*.npz) stored in self.root.

        Returns:
            tiles:         list of str, tile names
            paths:         dict, dictionary with the tile names as keys, paths[tile[i]] contains a list
                           of data multicube filepaths (*.npz) of the i.th tile
            sample_count:  np.array, sample_count[i] specifies the number of data samples in tiles[i]
        """

        tiles = os.listdir(self.root)

        if 'LICENSE' in tiles:
            tiles.remove('LICENSE')
        tiles.sort()

        paths = {}
        sample_count = np.zeros((len(tiles),), dtype=int)

        for i, tile in enumerate(tiles):
            tile_path = os.path.join(self.root, tile)
            filenames = os.listdir(tile_path)
            filenames.sort()
            sample_count[i] = len(filenames)

            files = []

            for name in filenames:
                files.append(os.path.join(tile_path, name))
            paths[tile] = files

        return tiles, paths, sample_count

    @staticmethod
    def _get_train_val_split(
            tiles: List[str], paths: Dict[str, List[str]], sample_count: np.ndarray, return_samples: str = 'train'
    ) -> Tuple[List[str], List[str]]:
        """
        Splits the train split into VALIDATION_SIZE validation samples and sample_count.sum() - VALIDATION_SIZE
        training samples. All samples within one tile are either used for training or validation.

        Args:
            tiles:           list of str, tile names
            paths:           dict, dictionary with the tile names as keys, paths[tile[i]] contains a list
                             of data multicube filepaths (*.npz) of the i.th tile
            sample_count:    np.array, sample_count[i] specifies the number of data samples in tiles[i]
            return_samples:  str, 'train' or 'val', flag to return the training or validation samples

        Returns:
            tiles:           list of str, tile names of the training/validation tiles
            paths:           list of str, data multicube filepaths (*.npz) of the training/validation tiles
        """

        # Set seed to reproduce the same training/validation split
        set_seed(0)

        if return_samples not in ['train', 'val']:
            raise ValueError("Invalid train/val split identifier. Choose among ['train', 'val'].\n")

        # Total number of samples
        num_samples = sample_count.sum()

        # Convert the number of samples per tiles into probabilities
        prob = sample_count / num_samples

        num_train_samples = 0
        tiles_train = []

        # Randomly sample training tiles
        sample_from = tiles.copy()

        while num_train_samples < num_samples - VALIDATION_SIZE:
            # Sample a tile
            # (the more samples per tile, the higher its probability to be sampled as training tile)
            idx = np.random.choice(np.arange(len(sample_from)), replace=False, p=prob)
            tiles_train.append(sample_from[idx])
            num_train_samples += sample_count[idx]

            # Remove the sampled tile for the next iteration (sample without replacement)
            sample_from.pop(idx)
            sample_count = np.delete(sample_count, idx)
            prob = np.delete(prob, idx)
            prob /= prob.sum()

        # Sort the training tiles
        tiles_train.sort()

        # Use the remaining tiles as validation tiles
        tiles_val = list(set(tiles) - set(tiles_train))
        tiles_val.sort()

        # Extract the paths of the training and validation samples
        paths_train = []
        paths_val = []
        for tile in tiles:
            if tile in tiles_train:
                paths_train = paths_train + paths[tile]
            else:
                paths_val = paths_val + paths[tile]

        if return_samples == 'train':
            return tiles_train, paths_train
        return tiles_val, paths_val


def create_hdf5_group(hdf5_file: str, group: str) -> None:
    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        if not f.__contains__(group):
            f.create_group(group)


def process_npz_sample_to_hdf5(
        hdf5_file: str, directory: str, dataset: torch.utils.data.Dataset, sample_index: int
) -> None:

    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        # Load the npz sample: H x W x C x T
        data = dataset.__getitem__(sample_index)
        sample = data['highresdynamic'][:]

        p = Path(data['filepath_export'])
        filename = p.stem
        tile = p.parts[-2]

        # Create a hdf5 group per data sample
        group = os.path.join(directory, tile, filename)
        create_hdf5_group(hdf5_file, group)

        # Store the sample as hdf5 dataset
        dset = f[group].create_dataset(
            'highresdynamic', data=sample, compression='gzip', compression_opts=9
        )

        # (H x W x C x T) -> (T x C x H x W), format expected by detect_impaired_frames()
        images = torch.from_numpy(sample[:, :, :4, :]).permute(3, 2, 0, 1)
        cloud_mask = torch.from_numpy(sample[:, :, [-1], :]).permute(3, 2, 0, 1)
        T = images.shape[0]

        # Detect impaired frames (unavailable/foggy/cloudy)
        if dataset.split == 'train':
            cloud_prob = torch.from_numpy(sample[:, :, [4], :]).permute(3, 2, 0, 1)
            idx_impaired_frames, _ = detect_impaired_frames(
                images, cloud_prob, cloud_mask, increased_filter_strength=False
            )
        else:
            idx_impaired_frames, _ = detect_impaired_frames(images, None, cloud_mask)

        dset = f[group].create_dataset(
            'idx_impaired_frames', data=idx_impaired_frames, compression='gzip', compression_opts=9
        )

        # Indices of foggy/cloudy (but available) frames
        idx_cloudy_frames = [i for i in idx_impaired_frames if ~torch.all(torch.isnan(images[i, ...]))]
        dset = f[group].create_dataset(
            'idx_cloudy_frames', data=idx_cloudy_frames, compression='gzip', compression_opts=9
        )

        # Indices of available and cloud-free frames
        idx_good_frames = [i for i in range(T) if i not in idx_impaired_frames]
        dset = f[group].create_dataset(
            'idx_good_frames', data=idx_good_frames, compression='gzip', compression_opts=9
        )

        # Save a flag for every time step to indicate whether the observation is valid or not
        # (1: available and cloud-free, 0: otherwise)
        valid_obs = [1 if i in idx_good_frames else 0 for i in range(T)]
        dset = f[group].create_dataset(
            'valid_obs', data=valid_obs, compression='gzip', compression_opts=9
        )


parser = ArgumentParser()
parser.add_argument('--root_source', type=str, required=True)
parser.add_argument('--root_dest', type=str, required=True)
parser.add_argument('--split', type=str, required=True)
parser.add_argument('--mode', type=str, default='train')


def main(args):
    dataset = EarthNet2021_npz2hdf5(root=args.root_source, split=args.split, mode=args.mode)
    if args.split == 'train':
        hdf5_file = os.path.join(args.root_dest, 'train.hdf5')
    else:
        hdf5_file = os.path.join(args.root_dest, args.split + '_test_split.hdf5')

    # Create a subgroup per data split
    directory = args.mode if args.split == 'train' else args.split + '_test_split'
    #directory = os.path.join(directory, 'samples')
    create_hdf5_group(hdf5_file, directory)

    # Create a subgroup for each tile
    for tile in dataset.tiles:
        create_hdf5_group(hdf5_file, os.path.join(directory, tile))

    # Iterate over all data samples in the given data split: npz to hdf5 conversion
    for i in tqdm(range(dataset.__len__())):
        process_npz_sample_to_hdf5(hdf5_file, directory, dataset, sample_index=i)

    # Retrieve valid samples (samples with at least 5 cloud-free images) by creating a EarthNet2021 dataset instance
    # from the created hdf5 file
    dataset2 = EarthNet2021Dataset(
        root=args.root_dest, hdf5_file=Path(hdf5_file).parts[-1], split=args.split, mode=args.mode,
        filter_settings={'type': 'cloud-free', 'min_length': 5, 'return_valid_obs_only': False}
    )
    dataset2.f.close()

    # Dump the sample paths to a list to speed up data loading later on
    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        name = args.mode if args.split == 'train' else args.split + '_test_split'
        f.create_dataset(f'path_samples_{name}', data=dataset2.paths)

    print('Done')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        main(parser.parse_args())
