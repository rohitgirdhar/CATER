# Baselines on CATER

Here we provide code and some basic instructions to reproduce some of the baselines on CATER.

## Setup and process the data

Generate or copy over the pre-generated data to a fast I/O directory. `unzip` the file. For the non-local code, we need to generate LMDBs. Do so by:

```bash
$ cd baselines/video-nonlocal-net
$ vim process_data/cater/gen_lmdbs.py  # <-- change the path to the lists folder for which to generate lmdb
$ python process_data/cater/gen_lmdbs.py  # <-- This will generate the LMDBs in the same folder as lists
```

## Launch non-local training pipeline

The training follows the same paradigm as the non local codebase. We provide sample `.yaml` config files in the `configs_cater` folder. To launch training, we provide a nifty little `launch.py` script. Again, you'll likely need to edit it to point it to the right paths (if needed). The pretrained model used for initialization can be downloaded from the [non-local codebase](https://github.com/facebookresearch/video-nonlocal-net#pre-trained-models-for-downloads) [[direct link](https://cmu.box.com/s/nw9hqonwz25yl86xh9q3m9yoak8fige6)]. Then, launch the training as follows:

```bash
$ vim configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml  # <-- change the first 2 lines to point to the data as processed above
$ python launch.py -c configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml
```

This would run the training and testing. The final trained output model for this run available is [here](https://cmu.box.com/s/1q767qg2b50enmo32rxk7j19xj0apga6).
Configs for other experiments can also be created based on provided config, and can be run in the same way.

### LSTM on top of Non local features

The above experiment uses the standard non-local testing paradigm: 30-crop average. However, for a task like CATER, training a temporal model on top of the clip features might make more sense than averaging the features. Hence, we also try a LSTM-based aggregation, which can be run as follows (after training the non-local model):


```bash
$ python launch.py -c configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml -t test  # <-- Test the model and store the features. These 
$ python launch.py -c configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml -t lstm  # <-- To train/test the LSTM. We saw some random variation in LSTM training so this script trains/tests the model 3 times and averages the numbers for a more stable estimate of performance
```

The expected performance for these models as in the paper is in this table (please expect some random variation across training runs):


| Expt | Config | Localize (top-1 accuracy) |
|------|--------|---------------------|
| R3D + NL, 32 frame | configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml | 28.8 |
| R3D + NL, 32 frame, LSTM | configs_cater/001_I3D_NL_localize_imagenetPretrained_32f_8SR.yaml | 45.5 |


## Tracking baseline

This code was tested using `pytorch 0.4`. Set the paths in `main.py` and run it to run the tracking baseline. It uses [DaSiamRPN](https://github.com/foolwood/DaSiamRPN) codebase with pretrained model.

```bash
$ cd tracking
$ source activate pytorch0.4  # <-- Need pytorch0.4, OpenCV (with ffmpeg) installed
$ python main.py  # <-- To run the tracking. Change parameters in the 
```


| Expt | Config | Localize (top-1 accuracy) |
|------|--------|---------------------|
| Tracking | - | 33.9 |
