import math
import os
import re
import sys
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from lib import config_utils, utils
from lib.datasets import EarthNet2021Dataset, SEN12MSCRTSDataset, dataset_tools


def create_hdf5_group(hdf5_file, group):
    with h5py.File(hdf5_file, 'a', libver='latest') as f:
        if not f.__contains__(group):
            f.create_group(group)


def extract_floats_from_string(s: str):
    return list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", s)))


parser = ArgumentParser()
parser.add_argument('--config_file', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
parser.add_argument('--out_hdf5_filename', type=str, required=True)
parser.add_argument('--ratio_masked', type=float, required=False)
parser.add_argument('--split', type=str, required=False)


def main(args):

    config = config_utils.read_config(args.config_file)

    if args.ratio_masked is not None:
        config.mask.ratio_masked_frames = args.ratio_masked

    if args.split is not None:
        if args.split != 'train':
            config.data.split = args.split
            if config.data.dataset == 'earthnet2021':
                config.data.mode = None

    if config.data.dataset == 'earthnet2021':
        dataset = EarthNet2021Dataset(
            **utils.without_keys(config.data, ['dataset']),  mask_kwargs=config.mask, augment=False,
            return_cloud_prob=True, return_class_map=True, return_cloud_mask=True, to_export=True
        )

        dtype_s2 = torch.float16
        id_s2 = 'frames_target'
        id_s2_dates = ''

        id = f'_{config.data.mode}' if config.data.split == 'train' else f'{config.data.split}_test_split'

    elif config.data.dataset == 'sen12mscrts':
        dataset = SEN12MSCRTSDataset(
            **utils.without_keys(config.data, ['dataset']), mask_kwargs=config.mask, augment=False,
            return_cloud_prob=True, return_cloud_mask=True, return_acquisition_dates=True, to_export=True
        )

        dtype_s2 = torch.float32
        id_s2 = 'S2'
        id_s2_dates = 'S2_'

        id = ''
    else:
        raise NotImplementedError

    # Create the output directory and initialize the output hdf5 file
    os.makedirs(args.out_dir, exist_ok=True)
    hdf5_file = os.path.join(args.out_dir, f'{args.out_hdf5_filename}')
    f = h5py.File(hdf5_file, 'a', libver='latest')

    # Create a mask directory and a mask type identifier
    mask_dir, mask_name = dataset_tools.get_mask_sampling_id_hdf5(config.mask)
    params_query = extract_floats_from_string(mask_dir)

    utils.set_seed(config.misc.random_seed)

    # Iterate over all validation samples
    for idx in tqdm(range(dataset.__len__())):

        # Make temporal trimming deterministic across simulations
        if dataset.paths[idx] in f:
            # A previous simulation exists: load the indices of the sampled frames
            t_sampled = f[dataset.paths[idx]]['t_sampled'][:]

            # List the hdf5 groups of all previous simulations
            groups = [obj for obj in f[dataset.paths[idx]] if isinstance(f[dataset.paths[idx]][obj], h5py.Group)]
            groups_params = [extract_floats_from_string(group) for group in groups]

            # Check if a previous simulation with the same ratio of masked frames and the same ratio of fully masked
            # frames exists
            if params_query in groups_params:
                subdir = groups[groups_params.index(params_query)]

                # Load the indices of the previously sampled masked frames
                t_masked = {'indices_masked': f[dataset.paths[idx]][subdir]['t_masked'][:]}

                if len(params_query) == 2:
                    # Current simulation exhibits fully masked frames, too
                    if 't_fully_masked' in f[dataset.paths[idx]][subdir]:
                        # Load the indices of the previously sampled fully masked frames
                        t_masked['indices_fully_masked'] = f[dataset.paths[idx]][subdir]['t_fully_masked'][:]
                    else:
                        n = math.ceil(config.mask.ratio_fully_masked_frames * len(t_sampled))
                        t_masked['indices_fully_masked'] = np.random.choice(t_masked['indices_masked'], n,
                                                                            replace=False)
            else:
                groups_params = [param[0] for param in groups_params]

                if params_query[0] in groups_params:
                    subdir = groups[groups_params.index(params_query[0])]

                    # Load the indices of the previously sampled masked frames
                    t_masked = {'indices_masked': f[dataset.paths[idx]][subdir]['t_masked'][:]}

                    if len(params_query) == 2:
                        n = math.ceil(config.mask.ratio_fully_masked_frames * len(t_sampled))
                        t_masked['indices_fully_masked'] = np.random.choice(t_masked['indices_masked'], n,
                                                                            replace=False)
                else:
                    t_masked = None

            sample = dataset.__getitem__(idx, t_sampled=t_sampled, t_masked=t_masked)
        else:
            sample = dataset.__getitem__(idx)

        create_hdf5_group(hdf5_file, sample['filepath'])

        group_s2 = sample['filepath']
        group_s1 = None
        if config.data.dataset == 'sen12mscrts':
            # Create a hdf5 group for storing S2 and S1 data separately
            group_s2 = os.path.join(sample['filepath'], 'S2')
            create_hdf5_group(hdf5_file, group_s2)

            if getattr(dataset, 'include_S1'):
                group_s1 = os.path.join(sample['filepath'], 'S1')
                create_hdf5_group(hdf5_file, group_s1)

        # Sample not yet used in a previous simulation
        if id_s2 not in f[group_s2].keys():
            # Store the target S2 satellite image time series (unmasked image sequence)
            dset = f[group_s2].create_dataset(
                id_s2, data=sample['y'].type(dtype_s2), compression='gzip', compression_opts=9
            )

            # Store S2 acquisition dates per frame
            dset = f[group_s2].create_dataset(f'{id_s2_dates}dates', data=sample[f'{id_s2_dates}dates'])

            # Store associated cloud masks
            dset = f[group_s2].create_dataset(
                'cloud_mask', data=sample['cloud_mask'].type(torch.int8), compression='gzip', compression_opts=9
            )

            # Store cloud probability maps
            if getattr(dataset, 'return_cloud_prob'):
                dset = f[group_s2].create_dataset(
                    'cloud_prob', data=sample['cloud_prob'].type(dtype_s2), compression='gzip', compression_opts=9
                )

            # Store classification maps
            if getattr(dataset, 'return_class_map', False):
                dset = f[sample['filepath']].create_dataset(
                    'classification', data=sample['classification'].type(torch.int8), compression='gzip',
                    compression_opts=9
                )

            # Store indices of the sampled frames
            # (i.e., to reproduce the temporally trimmed sequence from the original sequence)
            dset = f[sample['filepath']].create_dataset(
                't_sampled', data=sample['to_export']['t_sampled'].type(torch.int8), compression='gzip',
                compression_opts=9
            )

            if group_s1 is not None:
                # Store the S1 satellite image time series associated with the target S2 satellite image time series
                dset = f[group_s1].create_dataset('S1', data=sample['S1'], compression='gzip', compression_opts=9)

                # Store S1 acquisition dates per frame
                dset = f[group_s1].create_dataset('S1_dates', data=sample['S1_dates'])

        else:
            # Sanity check
            if not torch.equal(torch.from_numpy(f[group_s2][id_s2][:]).float(), sample['y']):
                raise RuntimeError("Simulated run is inconsistent with previous simulations.")

        # Now, store the masks dedicated to this simulation run
        group = os.path.join(sample['filepath'], mask_dir)
        create_hdf5_group(hdf5_file, group)

        dset = f[group].create_dataset(
            mask_name, data=sample['masks'].type(torch.int8), compression='gzip', compression_opts=9
        )

        if config.mask.mask_type == 'simulated_clouds':
            # Store the alpha-blended input sequence
            dset = f[group].create_dataset(
                f'{id_s2}input_frames_simulated_clouds', data=sample['x'].type(dtype_s2), compression='gzip',
                compression_opts=9
            )

        if 't_masked' not in f[group]:
            # Store the indices of the masked frames
            dset = f[group].create_dataset(
                't_masked', data=sample['to_export']['indices_masked'].type(torch.int8), compression='gzip',
                compression_opts=9
            )

            if 'indices_fully_masked' in sample['to_export'].keys() and 't_fully_masked' not in f[group]:
                dset = f[group].create_dataset(
                    't_fully_masked', data=sample['to_export']['indices_fully_masked'].type(torch.int8),
                    compression='gzip', compression_opts=9
                )

    # Save sample paths
    if f'path_samples{id}' not in f:
        dset = f.create_dataset(f'path_samples{id}', data=dataset.paths)

    f.close()
    print('Done')


if __name__ == '__main__':

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(1)
    else:
        main(parser.parse_args())
