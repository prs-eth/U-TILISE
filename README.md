<p align="center">
    <h1>U-TILISE: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series</h1>
</p>

<p align="center">
    <h3 align="center"> <strong><sup>1</sup>Corinne Stucker,   <sup>2</sup>Vivien Sainte Fare Garnot,   <sup>1</sup>Konrad Schindler</strong>  </h3>
</p>

<p align="center">
    <strong><sup>1</sup> Chair of Photogrammetry and Remote Sensing, ETH Zurich</strong><br>
    <strong><sup>2</sup> Institute for Computational Science, University of Zurich</strong>
</p>

<p align="center">
    <h3 align="center">[<a href="https://arxiv.org/abs/2305.13277">ArXiv</a>]</h3>
</p>


Satellite image time series in the optical and infrared spectrum suffer from frequent data gaps due to cloud cover, cloud shadows, and temporary sensor outages. It has been a long-standing problem of remote sensing research how to best reconstruct the missing pixel values and obtain complete, cloud&#8209;free image sequences. We approach that problem from the perspective of representation learning and develop U&#8209;TILISE, an efficient neural model that is able to implicitly capture spatio-temporal patterns of the spectral intensities, and that can therefore be trained to map a cloud&#8209;masked input sequence to a cloud&#8209;free output sequence. The model consists of a convolutional *spatial encoder* that maps each individual frame of the input sequence to a latent encoding; an attention-based *temporal encoder* that captures dependencies between those per-frame encodings and lets them exchange information along the time dimension; and a convolutional *spatial decoder* that decodes the latent embeddings back into multi-spectral images. We experimentally evaluate the proposed model on EarthNet2021, a dataset of Sentinel-2 time series acquired all over Europe, and demonstrate its superior ability to reconstruct the missing pixels. Compared to a standard interpolation baseline, it increases the PSNR by 1.8 dB at previously seen locations and by 1.3 dB at unseen locations.

<image src="docs/teaser.png"/>


## Setup

### Dependencies
This code was developed using Ubuntu 22.04, Python 3.10, PyTorch 1.13, and CUDA 11.6.
For an optimal experience, we recommend creating a new conda environment and installing the required dependencies with the following commands:
```bash
conda env create -f environment.yml
conda activate u-tilise
```

After setting up the environment, establish a corresponding IPython kernel named ``, . This is necessary to execute the demo Jupyter Notebook [demo.ipynb](demo.ipynb):
```bash
ipython kernel install --user --name=u-tilise
```

### Checkpoints
You can download our pretrained model checkpoints [here](https://share.phys.ethz.ch/~pf/stuckercdata//checkpoints/).
If you prefer an automated approach, execute the script below to download and extract all checkpoints to the `./checkpoints/` directory:
```bash
bash ./scripts/download_checkpoints.sh
``` 

We provide the following model checkpoints:
* `utilise_earthnet2021.pth`: model weights for  trained on the [EarthNet2021](https://www.earthnet.tech/en21/quick-start-guide/) dataset.
* `utilise_sen12mscrts_wo_s1.pth`: model weights for U&#8209;TILISE trained on the [SEN12MS-CR-TS](https://patricktum.github.io/cloud_removal/sen12mscr/) dataset, *without* SAR guidance.
* `utilise_sen12mscrts_w_s1.pth`: model weights for U&#8209;TILISE trained on the [SEN12MS-CR-TS](https://patricktum.github.io/cloud_removal/sen12mscr/) dataset, *with* SAR guidance.


### Sample data
For demonstration purposes, we offer the preprocessed *iid* test split of the EarthNet2021 dataset. To download this data, execute the following command:
```bash
bash ./scripts/download_data_earthnet2021.sh
```

This will download and unpack the data (~ 14 GB) into the `./data/` directory. The provided data sets are:
* `earthnet_iid_test_split.hdf5`: The original 30-frames time series with actual data gaps.
* `earthnet_iid_test_split_simulation.hdf5`: The corresponding cloud&#8209;free time series with synthetically added data gaps.


## Data

### EarthNet2021
[EarthNet2021](https://www.earthnet.tech/en21/quick-start-guide/)[^1] provides Sentinel&#8209;2 satellite image time
series collected over Central and Western Europe from November 2016 to May 2020. Each time series comprises 
30 images with Level-1C top-of-atmosphere (TOA) reflectances. The images are acquired in a regular temporal interval of 
five days. Every image is composed of the four spectral bands B2 (blue), B3 (green), B4 (red), and B8 (near-infrared) 
and covers a spatial extent of 128&times;128 pixels (2.56&times;2.56 km in scene space), resampled to the resolution of 
20 m. Furthermore, the dataset includes pixel-wise cloud probability maps (training data only) and binary cloud (and cloud shadow) masks.


### SEN12MS-CR-TS
[SEN12MS-CR-TS](https://patricktum.github.io/cloud_removal/sen12mscr/)[^2] provides globally sampled Sentinel&#8209;2 satellite image time
series from 2018 with a spatial extent of 256&times;256 pixels (2.56&times;2.56 km in scene space). Each time series
contains 30 images. The images encompass all 13 spectral bands, upsampled to 10 m resolution. Furthermore, every optical image 
is paired with a spatially co-registered, temporally close (but not synchronous) C-band SAR image with two channels representing 
the $\sigma_0$ back-scatter coefficients in the VV and VH polarizations, in units of decibels (dB). Furthermore, the dataset 
includes pixel-wise cloud probabilities and binary cloud masks.

### Simulation of data gaps
To train and quantitatively assess U&#8209;TILISE's performance, we utilize gap&#8209;free (cloud&#8209;free) Sentinel&#8209;2 satellite image time series.
Our preprocessing steps involve:
1. We first identify all images with partially occluded pixels or images that are occluded/missing entirely by applying 
a threshold to the cloud probability maps (if available) or the binary cloud masks. We then remove all images with data gaps to produce 
cloud&#8209;free time series that exhibit a valid observation for every spatio-temporal location.
2. We discard time series with less than five remaining images, as we deem such sequences too short for learning spatio-temporal patterns.
3. To generate synthetic data gaps, we randomly sample real cloud masks from other acquisition times and/or locations within the same Sentinel-2 tile
   and superimpose those masks onto the gap-free time series by setting the reflectance of occluded pixels (according to the masks) to the maximum value~1.


### Custom dataset and data loader
Ensure the output of your custom data loader meets the following minimum requirements:

- `x`: (Masked) input time series, of shape $(T \times C \times H \times W)$.
- `masks`: Masks used to flag occluded/missing pixels in `x` with dimensions $(T \times 1 \times H \times W)$.
- `position_days`: Positions used for positional encoding, of shape $(T, )$.
- `y`: Target time series, of shape $(T \times C \times H \times W)$.

For visualization during training in Weights & Biases:
- Include `c_index_rgb`, `c_index_nir`, and `sample_index`.

For visualization in [demo.ipynb](demo.ipynb):
- Include `c_index_rgb`.


## Preprocessing

:warning: **Note:** If you don't intend to train U&#8209;TILISE on either the EarthNet2021 or the SEN12MS&#8209;CR&#8209;TS dataset yourself, you can skip this section.

To train U&#8209;TILISE on either the EarthNet2021 or the SEN12MS&#8209;CR&#8209;TS dataset, you will need to complete several steps:
downloading the dataset, converting it from its native format to one compatible with our data loaders, and running a simulation to 
generate cloud&#8209;free sequences with synthetically added data gaps. Here's a step-by-step guide:

**EarthNet2021**
1. **Dataset download**
   
   Obtain the dataset by following the download instructions available [here](https://www.earthnet.tech/en21/ds-download/).

2. **Data conversion**
   
   The EarthNet2021 dataset comprises two .npz files for every time series. We aggregate the data of each data split
   (i.e., train, iid, ood) into a single HDF5 file for further use. To run the npz-to-HDF5 conversion, execute the commands below:

   ```bash
   python ./toolbox/EarthNet2021_npz2hdf5.py --root_source <data_directory> --root_dest <output_directory> --split train --mode train
   python ./toolbox/EarthNet2021_npz2hdf5.py --root_source <data_directory> --root_dest <output_directory> --split train --mode val
   python ./toolbox/EarthNet2021_npz2hdf5.py --root_source <data_directory> --root_dest <output_directory> --split iid
   python ./toolbox/EarthNet2021_npz2hdf5.py --root_source <data_directory> --root_dest <output_directory> --split ood
   ```
    
   Ensure to replace `data_directory` with the root directory where you have saved your downloaded data and `output_directory` with
   your preferred destination for the HDF5 files.
   
   Upon executing the above commands, you should find the following HDF5 files in `output_directory`:
   - `train.hdf5`
   - `iid_test_split.hdf5`
   - `ood_test_split.hdf5`
   
   Besides the npz-to-HDF5 conversion, the script also stores the indices of unavailable frames and identifies all frames with partially
   or fully occluded pixels for each time series in the respective data split.


3. **Preprocessing of the validation split**

   To generate time series with artificial data gaps for training, run the following command:
   
   ```bash
   python ./toolbox/simulate_dataset.py --config_file ./data/configs/config_earthnet2021_simulation_val.yaml --out_dir <output_directory> --out_hdf5_filename earthnet2021_val_simulation.hdf5
   ```

   Make sure to set `root` in the [config_earthnet2021_simulation_val.yaml](./data/configs/config_earthnet2021_simulation_val.yaml)
   configuration file to match the `output_directory` used in step 2. Optionally, change `max_seq_length` to modify the fixed
   maximal temporal length $T$.

   This process produces a new HDF5 file, `/output_directory/earthnet2021_val_simulation.hdf5`. Among others, this file contains the
   temporally trimmed cloud&#8209;free validation time series, the corresponding time series with synthetically introduced data gaps, the masks
   used for masking, and the acquisition dates. 


4. **Preprocessing of the test splits**

   To generate time series with artificial data gaps for testing and evaluation, execute the command below:
   
   ```bash
   python ./toolbox/simulate_dataset.py --config_file ./data/configs/config_earthnet2021_simulation_test.yaml --out_dir <output_directory> --out_hdf5_filename earthnet2021_iid_test_split_simulation.hdf5
   ```

   Again, ensure to configure `root` in the [config_earthnet2021_simulation_test.yaml](./data/configs/config_earthnet2021_simulation_test.yaml)
   configuration file to match the `output_directory` used in step 2. If you wish to process the *ood* test split instead, set the
   `split` parameter to *ood* and adjust `--out_hdf5_filename` accordingly.

   > Note that `max_seq_length` in the [config_earthnet2021_simulation_test.yaml](./data/configs/config_earthnet2021_simulation_test.yaml)
   configuration file is set to None to skip the temporal trimming of the test time series.



**SEN12MS-CR-TS**
1. **Dataset download**

   Follow the provided [instructions](https://patricktum.github.io/cloud_removal/sen12mscr/) to download the dataset.

2. **Data conversion**

   To convert and aggregate the .tif files from individual acquisitions into a single HDF5 file for each data split
   (i.e., train, val, test), utilize the functionalities provided [here](https://github.com/PatrickTUM/SEN12MS-CR-TS/blob/master/util/hdf5converter/).

3. **Detection of real data gaps**

   Execute the following script to identify all frames that have partial or complete occlusions:

   ```bash
   python ./toolbox/SEN12MSCRTS_detect_cloudy_frames.py --root <data_directory> --split train
   python ./toolbox/SEN12MSCRTS_detect_cloudy_frames.py --root <data_directory> --split val
   python ./toolbox/SEN12MSCRTS_detect_cloudy_frames.py --root <data_directory> --split test
   ```

   Upon completion, the HDF5 files created in step 2 will be updated with indices corresponding to images exhibiting data gaps.

4. **Preprocessing of the validation and test splits**

   To generate time series with artificial data gaps for training and evaluation, execute the commands below:
   
   ```bash
   python ./toolbox/simulate_dataset.py --config_file ./data/configs/config_sen12mscrts_simulation_val.yaml --out_dir <output_directory> --out_hdf5_filename sen12mscrts_val_simulation.hdf5
   python ./toolbox/simulate_dataset.py --config_file ./data/configs/config_sen12mscrts_simulation_test.yaml --out_dir <output_directory> --out_hdf5_filename sen12mscrts_test_simulation.hdf5
   ```

   For detailed instructions on each parameter and step, refer to the instructions provided for the EarthNet2021 dataset above.



## Training 

To initiate training, execute the following command:
```bash
python run_train.py /path/to/config_file.yaml --save_dir <output_directory> 
```

where:

- `/path/to/config_file.yaml` is the YAML configuration file specifying all runtime arguments.
- `--save_dir` specifies the output directory.

All training hyperparameters are predefined in [default.yaml](configs/default.yaml), set to the values utilized
in the main experiments of the paper. If a parameter is specified in `/path/to/config_file.yaml`, it will override the default value.

To view all available training options, run:
```bash
python run_train.py -h
```

**Example configuration files**

We provide a collection of configuration files within the `./configs/` directory:
* `default.yaml`: Contains the default parameter settings.
* `config_earthnet2021_train.yaml`: Additional settings used to train on the EarthNet2021 dataset.
* `config_sen12mscrts_train.yaml`: Additional settings used to train on the SEN12MS&#8209;CR&#8209;TS dataset.
* `config_earthnet2021_test_simulation.yaml`: Test settings for the EarthNet2021 dataset (synthetic data gaps).
* `config_earthnet2021_test.yaml`: Test settings for the EarthNet2021 dataset (actual data gaps).

Upon execution, the `run_train.py` script combines the `default.yaml` with the provided `--config_file`, 
saving the resultant runtime configuration as a YAML file in the directory defined by `--save_dir`. As a point of reference, 
you can find an example runtime configuration at [demo_train_config.yaml](configs/demo_train_config.yaml). We will use this YAML file below to 
demonstrate the evaluation procedure.


## Evaluation

To perform the evaluation, execute the `run_eval.py` script using the following command:
```bash
python run_eval.py /path/to/config.yaml <method_name> --checkpoint <path_to_checkpoint> --test-data.data-dir <data_directory> --test-data.hdf5-file <hdf5_file_name> --test-data.split <data_split>
```

To view all available input options, run:
```bash
python run_eval.py -h
```


**Examples**

1. To evaluate the *iid* test split of the EarthNet2021 dataset, utilize the following command:
```bash
python run_eval.py ./configs/demo_train_config.yaml utilise --test-data.data-dir ./data/ --test-data.hdf5-file earthnet2021_iid_test_split_simulation.hdf5 --test-data.split iid --checkpoint ./checkpoints/utilise_earthnet2021.pth 
```

2. Similarly, to evaluate the linear interpolation baseline, run:
```bash
python run_eval.py ./configs/demo_train_config.yaml trivial --mode linear_interpolation --test-data.data-dir ./data/ --test-data.hdf5-file earthnet2021_iid_test_split_simulation.hdf5 --test-data.split iid
```


## Demo

### Inference

Check out the Jupyter Notebook [demo.ipynb](demo.ipynb). It provides a step-by-step demonstration of using U&#8209;TILISE to impute a 
given time series and visualize the associated attention masks.


## Citation
```bibtex
@article{stucker2023u,
  title={{U-TILISE}: A Sequence-to-sequence Model for Cloud Removal in Optical Satellite Time Series},
  author={Stucker, Corinne and Garnot, Vivien Sainte Fare and Schindler, Konrad},
  journal={arXiv preprint arXiv:2305.13277},
  year={2023}
}
```

## Acknowledgements
U&#8209;TILISE extends the architecture of [U-TAE](https://github.com/VSainteuf/utae-paps) to a full 3D spatio-temporal sequence-to-sequence model that preserves the temporal dimension.

We thank Vivien Sainte Fare Garnot for his efforts in open sourcing and maintaining U-TAE.[^3]


[^1]: C. Requena-Mesa, V. Benson, M. Reichstein, J. Runge, and J. Denzler, *EarthNet2021 :A large-scale dataset and challenge for earth surface forecasting as a guided video prediction task*, in IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2021, pp.1132–1142.
[^2]: P. Ebel, Y. Xu, M. Schmitt, and X. X. Zhu, *SEN12MS-CR-TS: A remote-sensing dataset for multimodal multitemporal cloud removal*, IEEE Transactions on Geoscience and Remote Sensing, vol.60, pp. 1-14, 2022.
[^3]: V.S.F. Garnot and L. Landrieu, *Panoptic segmentation of satellite image time series with convolutional temporal attention networks*, in IEEE/CVF International Conference on Computer Vision (ICCV), 2021, pp.4872–4881.
